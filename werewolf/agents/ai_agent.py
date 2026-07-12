import json
import logging
import re
from typing import Callable, Optional

from werewolf.agents.prompts import get_system_prompt, get_action_instruction
from werewolf.llm.provider import ModelRequest, Provider, ProviderResult
from werewolf.llm.records import (
    CallContext,
    ErrorCategory,
    UsageRecord,
    new_call_id,
)

logger = logging.getLogger("werewolf.agent")

MAX_RETRIES = 3


class AIAgent:
    """Prompt construction, response parsing, retry/fallback policy.

    Provider-agnostic: all API invocation goes through the injected
    Provider; every attempt is recorded to the injected ledger. This class
    never sees API keys or provider-specific response formats.
    """

    def __init__(
        self,
        player_id: int,
        role: str,
        team: str,
        provider: Optional[Provider] = None,
        wolf_roster: list[int] = None,
        model: str = "grok-4.3",
        show_prompts: bool = False,
        ledger=None,
        run_context: Optional[dict] = None,
        model_alias: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
    ):
        self.player_id = player_id
        self.role = role
        self.team = team
        self.provider = provider
        self.wolf_roster = wolf_roster or []
        self.model = model
        self.model_alias = model_alias
        self.reasoning_effort = reasoning_effort
        self.memory = {}
        self.show_prompts = show_prompts
        self.ledger = ledger
        self.run_context = run_context or {}
        self.system_prompt = get_system_prompt(role, player_id, wolf_roster)

        logger.debug(f"P{player_id} initialized as {role} ({team})")

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def act(
        self,
        observation: dict,
        validator: Callable[[dict, dict], tuple[bool, Optional[str]]],
        fallback_fn: Callable[[dict], dict],
        rng,
        update_memory: bool = True,
    ) -> dict:
        """update_memory=False makes the call read-only with respect to
        agent state: any updated_memory in the response is discarded.
        Used for belief assessments, which must never affect the game."""
        required_action = observation["required_action"]
        call_id = new_call_id()
        logger.debug(f"P{self.player_id} acting: {required_action}")

        if self.provider is None:
            logger.error("No LLM provider available, using fallback")
            self._record_non_api(
                observation, call_id, attempt=0,
                category=ErrorCategory.MISSING_API_KEY,
            )
            self._record_non_api(
                observation, call_id, attempt=0,
                category=ErrorCategory.FALLBACK_USED,
            )
            fallback = fallback_fn(observation)
            fallback["thought"] = (
                "No LLM provider available (missing API key or SDK) - "
                "using random action."
            )
            return fallback

        errors = []
        attempts_made = 0
        for attempt in range(1, MAX_RETRIES + 1):
            attempts_made = attempt
            logger.debug(f"P{self.player_id} attempt {attempt}/{MAX_RETRIES}")
            user_prompt = self._build_user_prompt(observation, errors)

            if self.show_prompts:
                print(f"\n{'='*60}")
                print(f"  PROMPT TO P{self.player_id} ({self.role})")
                print(f"{'='*60}")
                print(f"[SYSTEM PROMPT]\n{self.system_prompt[:500]}...")
                print(f"\n[USER PROMPT]\n{user_prompt}")
                print(f"{'='*60}\n")

            result = self.provider.complete(ModelRequest(
                model=self.model,
                system_prompt=self.system_prompt,
                user_prompt=user_prompt,
                reasoning_effort=self.reasoning_effort,
            ))
            record = self._record_from_result(
                observation, call_id, attempt, result
            )

            if not result.ok:
                logger.warning(
                    f"P{self.player_id} API failure "
                    f"({result.error_category and result.error_category.value}): "
                    f"{result.error_message}"
                )
                self._record(record)
                if result.retryable is False:
                    break  # retrying cannot help (auth, context window, ...)
                continue

            parsed, parse_method = self._parse_response(result.text)
            record.parse_method = parse_method
            if parse_method == "regex":
                logger.warning(
                    f"P{self.player_id} recovered action via regex extraction"
                )

            if parsed is None:
                record.parse_ok = False
                record.error_category = ErrorCategory.MALFORMED_JSON
                record.retryable = True
                self._record(record)
                errors.append(
                    "Your previous response was not valid JSON. "
                    "Respond with a single valid JSON object only."
                )
                logger.error(
                    f"P{self.player_id} parsing failed, "
                    f"raw[:300]: {str(result.text)[:300]}"
                )
                continue

            record.parse_ok = True
            if not parsed.get("thought"):
                parsed["thought"] = (
                    "No specific reasoning provided - making decision based "
                    "on available information."
                )

            is_valid, error = validator(observation, parsed)
            if is_valid:
                record.validation_ok = True
                record.error_category = ErrorCategory.COMPLETED
                self._record(record)
                if update_memory and "updated_memory" in parsed:
                    self.memory = parsed["updated_memory"]
                logger.debug(f"P{self.player_id} action valid")
                return parsed

            record.validation_ok = False
            record.error_category = ErrorCategory.INVALID_GAME_ACTION
            record.retryable = True
            self._record(record)
            logger.warning(f"P{self.player_id} invalid action: {error}")
            errors.append(error)

        logger.warning(
            f"P{self.player_id} using fallback after {attempts_made} attempt(s)"
        )
        self._record_non_api(
            observation, call_id, attempt=attempts_made,
            category=ErrorCategory.FALLBACK_USED,
        )
        fallback = fallback_fn(observation)
        fallback["thought"] = (
            "Not enough information or repeated errors - choosing randomly."
        )
        return fallback

    # ------------------------------------------------------------------
    # Usage recording
    # ------------------------------------------------------------------

    def _build_context(self, observation: dict) -> CallContext:
        rc = self.run_context
        return CallContext(
            game_id=rc.get("game_id", ""),
            batch_id=rc.get("batch_id"),
            trial_index=rc.get("trial_index"),
            seed=rc.get("seed"),
            round=observation.get("round", 0),
            phase=observation.get("phase", ""),
            required_action=observation.get("required_action", ""),
            player_id=self.player_id,
            player_role=self.role,
            player_team=self.team,
            prompt_version=rc.get("prompt_version"),
            model_alias=self.model_alias,
        )

    def _record_from_result(
        self, observation: dict, call_id: str, attempt: int,
        result: ProviderResult,
    ) -> UsageRecord:
        return UsageRecord(
            context=self._build_context(observation),
            provider=self.provider.name if self.provider else "none",
            requested_model=self.model,
            call_id=call_id,
            attempt=attempt,
            resolved_model=result.resolved_model,
            usage=result.usage,
            cost=result.cost,
            latency_ms=result.latency_ms,
            provider_request_id=result.provider_request_id,
            finish_reason=result.finish_reason,
            api_attempted=True,
            api_ok=result.ok,
            error_category=result.error_category,
            retryable=result.retryable,
            provider_metadata=dict(result.provider_metadata),
        )

    def _record_non_api(
        self, observation: dict, call_id: str, attempt: int,
        category: ErrorCategory,
    ) -> None:
        self._record(UsageRecord(
            context=self._build_context(observation),
            provider=self.provider.name if self.provider else "none",
            requested_model=self.model,
            call_id=call_id,
            attempt=attempt,
            api_attempted=False,
            api_ok=False,
            error_category=category,
            retryable=False,
        ))

    def _record(self, record: UsageRecord) -> None:
        if self.ledger is not None:
            self.ledger.record(record)

    # ------------------------------------------------------------------
    # Parsing (unchanged semantics; now reports which method succeeded)
    # ------------------------------------------------------------------

    def _parse_response(self, raw: str) -> tuple[Optional[dict], Optional[str]]:
        """Returns (parsed_dict, method) where method is one of
        'direct' | 'repaired' | 'regex', or (None, None) on failure."""
        content = raw.strip()
        brace = content.find("{")
        if brace != -1:
            try:
                obj, _ = json.JSONDecoder().raw_decode(content, brace)
                if isinstance(obj, dict):
                    return obj, "direct"
            except json.JSONDecodeError:
                pass
            repaired = self._repair_json(content)
            if repaired:
                obj = json.loads(repaired)
                if isinstance(obj, dict):
                    return obj, "repaired"
        extracted = self._regex_extract(raw)
        if extracted:
            return extracted, "regex"
        return None, None

    def _build_user_prompt(self, observation: dict, errors: list[str]) -> str:
        required_action = observation["required_action"]
        turn_context = observation.get("turn_context")
        action_instruction = get_action_instruction(required_action, turn_context)

        alive_list = ", ".join(f"P{p['id']}" for p in observation["alive_players"])
        prompt_parts = [
            "=== CURRENT GAME STATE ===",
            f"Round: {observation['round']}",
            f"Phase: {observation['phase']}",
            f"You are: P{observation['self']['id']} ({observation['self']['role']})",
            f"Alive players: {alive_list}",
        ]

        if observation.get("private_info"):
            private = observation["private_info"]
            if "wolf_roster" in private:
                prompt_parts.append(f"Your wolf team: {', '.join(f'P{w}' for w in private['wolf_roster'])}")
            if "last_divine_result" in private:
                result = private["last_divine_result"]
                is_wolf = "WEREWOLF" if result.get("is_werewolf") else "NOT WEREWOLF"
                prompt_parts.append(f"Last divine: P{result['target_id']} is {is_wolf}")

        if observation.get("recent_events"):
            prompt_parts.append("\n=== RECENT EVENTS ===")
            for event in observation["recent_events"][-10:]:
                prompt_parts.append(self._format_event(event))

        prompt_parts.append(f"\n=== YOUR MEMORY ===\n{json.dumps(self.memory, indent=2)}")
        prompt_parts.append(f"\n=== INSTRUCTIONS ===\n{action_instruction}")
        prompt_parts.append("\nIMPORTANT: Always explain your reasoning in the 'thought' field. If you don't have enough information, say so.")

        if errors:
            prompt_parts.append(f"\n=== PREVIOUS ERRORS (fix these!) ===")
            for i, err in enumerate(errors, 1):
                prompt_parts.append(f"{i}. {err}")

        prompt_parts.append("\nRespond with a valid JSON object only. No extra text.")

        return "\n".join(prompt_parts)

    def _format_event(self, event: dict) -> str:
        etype = event.get("type")
        payload = event.get("payload", {})
        speaker = event.get("speaker_id")
        channel = event.get("channel", "public")

        if etype == "message":
            return f"[{channel.upper()}] P{speaker}: {payload.get('text', '')}"
        elif etype == "death_announcement":
            victim_id = payload.get("victim_id")
            victim_role = payload.get("victim_role")
            cause = payload.get("cause")
            if cause == "wolf_kill":
                return f"[DEATH] P{victim_id} was killed during the night"
            else:
                return f"[DEATH] P{victim_id} was eliminated by vote ({victim_role})"
        elif etype == "vote":
            return f"[VOTE] P{payload.get('voter_id')} voted for P{payload.get('target_id')}"
        elif etype == "elimination":
            return f"[ELIMINATED] P{payload.get('eliminated_id')} ({payload.get('eliminated_role')})"
        elif etype == "divine_result":
            result = "WEREWOLF" if payload.get("is_werewolf") else "NOT WEREWOLF"
            return f"[DIVINE] P{payload.get('target_id')} is {result}"
        elif etype == "runoff_announcement":
            candidates_str = ", ".join(f"P{c}" for c in payload.get("candidates", []))
            return f"[RUNOFF] Vote tied! Runoff between: {candidates_str}"
        elif etype == "no_elimination":
            return f"[NO ELIMINATION] Runoff tied — no one is eliminated today"
        elif etype == "game_status":
            return f"[STATUS] Wolves: {payload.get('alive_wolves')}, Village: {payload.get('alive_villagers')}"
        else:
            return f"[{etype.upper()}] {json.dumps(payload)}"

    @staticmethod
    def _repair_json(text: str) -> Optional[str]:
        start = text.find("{")
        if start == -1:
            return None

        chars = []
        i = start
        in_string = False
        brace_depth = 0
        bracket_depth = 0

        while i < len(text):
            c = text[i]

            if in_string:
                if c == "\\" and i + 1 < len(text):
                    next_c = text[i + 1]
                    if next_c in '"\\bfnrtu/':
                        chars.append(c)
                        chars.append(next_c)
                    else:
                        chars.append("\\\\")
                        chars.append(next_c)
                    i += 2
                    continue
                if c == '"':
                    rest = text[i + 1:].lstrip()
                    if not rest or rest[0] in ":,}]":
                        in_string = False
                        chars.append(c)
                    else:
                        chars.append('\\"')
                    i += 1
                    continue
                if c in "\n\r":
                    chars.append("\\n")
                    i += 1
                    continue
                if c == "\t":
                    chars.append("\\t")
                    i += 1
                    continue
                chars.append(c)
                i += 1
                continue

            if c == '"':
                in_string = True
            elif c == "{":
                brace_depth += 1
            elif c == "}":
                brace_depth -= 1
            elif c == "[":
                bracket_depth += 1
            elif c == "]":
                bracket_depth -= 1

            chars.append(c)
            i += 1

            if brace_depth == 0 and bracket_depth <= 0:
                break

        if in_string:
            chars.append('"')

        joined = "".join(chars).rstrip()
        if joined.endswith(":"):
            last_quote = joined.rfind('"', 0, len(joined) - 1)
            if last_quote != -1:
                second_last = joined.rfind('"', 0, last_quote)
                if second_last != -1:
                    joined = joined[:second_last].rstrip().rstrip(",")
            else:
                joined = joined[:-1].rstrip().rstrip(",")
        elif joined.endswith(","):
            joined = joined[:-1]

        while bracket_depth > 0:
            joined += "]"
            bracket_depth -= 1
        while brace_depth > 0:
            joined += "}"
            brace_depth -= 1

        try:
            json.loads(joined)
            return joined
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _regex_extract(text: str) -> Optional[dict]:
        result = {}
        action = {}

        for key in ("vote_target", "kill_target", "divine_target"):
            m = re.search(rf'"{key}"\s*:\s*"?(\w+)"?', text)
            if m:
                action[key] = m.group(1)

        for channel in ("public", "werewolf"):
            m = re.search(
                rf'"{channel}"\s*:\s*"((?:[^"\\]|\\.)*)"',
                text,
            )
            if m:
                result.setdefault("say", {})[channel] = m.group(1)

        if not action and "say" not in result:
            return None

        if action:
            result["action"] = action
        else:
            result["action"] = None
        result.setdefault("say", None)
        result["thought"] = "[recovered from malformed response]"
        return result


def create_agents(
    players: dict,
    api_key: str = "",
    model: str = "grok-4.3",
    show_prompts: bool = False,
    provider: Optional[Provider] = None,
    ledger=None,
    run_context: Optional[dict] = None,
    model_alias: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
) -> dict[int, AIAgent]:
    """Backward-compatible factory. If no provider is injected, one is
    built from (model, api_key) via the registry; with no key or SDK the
    provider stays None and agents use the random-fallback path (preserves
    the historical no-key test behavior)."""
    if provider is None and api_key:
        from werewolf.llm.registry import build_provider, resolve
        provider = build_provider(resolve(model), api_key=api_key)

    wolf_roster = [p.id for p in players.values() if p.role == "werewolf"]

    agents = {}
    for pid, player in players.items():
        agents[pid] = AIAgent(
            player_id=pid,
            role=player.role,
            team=player.team,
            provider=provider,
            wolf_roster=wolf_roster if player.role == "werewolf" else None,
            model=model,
            show_prompts=show_prompts,
            ledger=ledger,
            run_context=run_context,
            model_alias=model_alias,
            reasoning_effort=reasoning_effort,
        )

    return agents
