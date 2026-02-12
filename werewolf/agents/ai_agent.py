import json
import logging
import os
import re
from typing import Optional, Callable

from werewolf.agents.prompts import get_system_prompt, get_action_instruction

logger = logging.getLogger("werewolf.agent")

try:
    from xai_sdk import Client
    from xai_sdk.chat import system, user
    HAS_XAI = True
except ImportError:
    HAS_XAI = False
    logger.warning("xai-sdk package not installed. Run: pip install xai-sdk")


class AIAgent:
    def __init__(
        self,
        player_id: int,
        role: str,
        team: str,
        api_key: str,
        wolf_roster: list[int] = None,
        model: str = "grok-4-1-fast",
        show_prompts: bool = False
    ):
        self.player_id = player_id
        self.role = role
        self.team = team
        self.api_key = api_key
        self.wolf_roster = wolf_roster or []
        self.model = model
        self.memory = {}
        self.show_prompts = show_prompts
        self.system_prompt = get_system_prompt(role, player_id, wolf_roster)
        
        if HAS_XAI and api_key:
            self.client = Client(
                api_key=api_key,
                timeout=120
            )
        else:
            self.client = None
            
        logger.debug(f"P{player_id} initialized as {role} ({team})")

    def act(
        self,
        observation: dict,
        validator: Callable[[dict, dict], tuple[bool, Optional[str]]],
        fallback_fn: Callable[[dict], dict],
        rng
    ) -> dict:
        if not HAS_XAI or not self.client:
            logger.error("xai-sdk not available, using fallback")
            fallback = fallback_fn(observation)
            fallback["thought"] = "xai-sdk not installed - using random action. Run: pip install xai-sdk"
            return fallback
            
        max_retries = 3
        errors = []
        required_action = observation["required_action"]
        logger.debug(f"P{self.player_id} acting: {required_action}")

        for attempt in range(max_retries):
            logger.debug(f"P{self.player_id} attempt {attempt + 1}/{max_retries}")
            try:
                user_prompt = self._build_user_prompt(observation, errors)
                
                if self.show_prompts:
                    print(f"\n{'='*60}")
                    print(f"  PROMPT TO P{self.player_id} ({self.role})")
                    print(f"{'='*60}")
                    print(f"[SYSTEM PROMPT]\n{self.system_prompt[:500]}...")
                    print(f"\n[USER PROMPT]\n{user_prompt}")
                    print(f"{'='*60}\n")
                
                logger.debug(f"P{self.player_id} prompt length: {len(user_prompt)} chars")
                
                response = self._call_grok(user_prompt)

                if response is None:
                    error_msg = "API call failed or returned invalid JSON"
                    logger.warning(f"P{self.player_id}: {error_msg}")
                    errors.append(error_msg)
                    continue

                logger.debug(f"P{self.player_id} raw response: {json.dumps(response)[:200]}...")

                if not response.get("thought"):
                    response["thought"] = "No specific reasoning provided - making decision based on available information."

                is_valid, error = validator(observation, response)
                if is_valid:
                    if "updated_memory" in response:
                        self.memory = response["updated_memory"]
                    logger.debug(f"P{self.player_id} action valid")
                    return response

                logger.warning(f"P{self.player_id} invalid action: {error}")
                errors.append(error)

            except Exception as e:
                logger.error(f"P{self.player_id} exception: {str(e)}")
                errors.append(f"Exception: {str(e)}")

        logger.warning(f"P{self.player_id} using fallback after {max_retries} failures")
        fallback = fallback_fn(observation)
        fallback["thought"] = "Not enough information or repeated errors - choosing randomly."
        return fallback

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
    def _extract_json(text: str) -> str:
        brace = text.find("{")
        if brace == -1:
            return text
        try:
            obj, end = json.JSONDecoder().raw_decode(text, brace)
            return json.dumps(obj)
        except json.JSONDecodeError:
            pass
        repaired = AIAgent._repair_json(text)
        return repaired if repaired else text

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

    def _call_grok(self, user_prompt: str) -> Optional[dict]:
        logger.debug(f"P{self.player_id} calling Grok API with model={self.model}")

        try:
            chat = self.client.chat.create(model=self.model)
            chat.append(system(self.system_prompt))
            chat.append(user(user_prompt))
            
            response = chat.sample()
            raw_content = response.content
            
            logger.debug(f"P{self.player_id} received response, tokens: {response.usage.completion_tokens if response.usage else 'N/A'}")
            
            if response.usage and hasattr(response.usage, 'reasoning_tokens') and response.usage.reasoning_tokens:
                logger.info(f"P{self.player_id} reasoning tokens: {response.usage.reasoning_tokens}")

            content = raw_content.strip()
            content = self._extract_json(content)
            logger.debug(f"P{self.player_id} parsing JSON: {content[:200]}...")

            try:
                return json.loads(content)
            except json.JSONDecodeError:
                extracted = self._regex_extract(raw_content)
                if extracted:
                    logger.warning(f"P{self.player_id} recovered action via regex extraction")
                    return extracted
                logger.error(f"P{self.player_id} all parsing failed, raw[:300]: {raw_content[:300]}")
                return None

        except Exception as e:
            logger.error(f"P{self.player_id} API error: {type(e).__name__}: {e}")
            print(f"[API Error] P{self.player_id}: {e}")
            return None


def create_agents(
    players: dict,
    api_key: str,
    model: str = "grok-4-1-fast",
    show_prompts: bool = False
) -> dict[int, AIAgent]:
    wolf_roster = [p.id for p in players.values() if p.role == "werewolf"]

    agents = {}
    for pid, player in players.items():
        agents[pid] = AIAgent(
            player_id=pid,
            role=player.role,
            team=player.team,
            api_key=api_key,
            wolf_roster=wolf_roster if player.role == "werewolf" else None,
            model=model,
            show_prompts=show_prompts
        )

    return agents
