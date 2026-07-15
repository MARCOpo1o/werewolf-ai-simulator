import random
from collections import Counter
from dataclasses import replace
from typing import Optional

from werewolf.engine.state import GameState, PlayerState
from werewolf.engine.events import (
    EVENT_SCHEMA_VERSION,
    create_message_event,
    create_thought_event,
    create_kill_event,
    create_death_announcement_event,
    create_divine_result_event,
    create_vote_event,
    create_elimination_event,
    create_runoff_announcement_event,
    create_no_elimination_event,
    create_phase_event,
    create_win_event,
    create_game_status_event,
    create_belief_snapshot_event,
)
from werewolf.engine.beliefs import (
    BELIEF_SCHEMA_VERSION,
    CHECKPOINT_POST,
    CHECKPOINT_PRE,
    parse_belief_snapshot,
)
from werewolf.engine.visibility import build_observation, update_player_seen_index
from werewolf.engine.validate import validate_action, get_fallback_action, _to_int
from werewolf.engine.logging import JSONLLogger, ConsoleTranscript
from werewolf.roles.assign import assign_roles
from werewolf.agents.ai_agent import AIAgent, create_agents
from werewolf.agents.prompts import get_prompt_version
from werewolf.engine.limits import (
    PUBLIC_MESSAGE_MAX_CHARS,
    WOLF_MESSAGE_MAX_CHARS,
    limits_dict,
    truncate_text,
)
from werewolf.llm.ledger import UsageLedger
from werewolf.llm.provider import GenerationConfig

_CODE_COMMIT = None


def get_code_commit() -> Optional[str]:
    """Short git SHA of the running code, or None outside a checkout."""
    global _CODE_COMMIT
    if _CODE_COMMIT is None:
        import pathlib
        import subprocess
        try:
            _CODE_COMMIT = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=pathlib.Path(__file__).resolve().parents[2],
                capture_output=True, text=True, timeout=5,
            ).stdout.strip() or ""
        except Exception:
            _CODE_COMMIT = ""
    return _CODE_COMMIT or None


class GameEngine:
    PHASE_ORDER = [
        "night_wolf_chat",
        "night_wolf_kill",
        "night_seer",
        "day_announce",
        "day_assess",
        "day_discuss",
        "day_vote",
    ]

    def __init__(
        self,
        n_players: int = 7,
        n_wolves: int = 2,
        n_seers: int = 1,
        seed: int = 42,
        output_dir: str = "outputs/games",
        api_key: str = "",
        model: str = "grok-4.3",
        show_all_channels: bool = True,
        show_prompts: bool = False,
        transcript_enabled: bool = True,
        provider=None,
        ledger: UsageLedger = None,
        model_alias: str = None,
        reasoning_effort: str = None,
        batch_id: str = None,
        trial_index: int = None,
        belief_snapshots: bool = True,
        generation_config: GenerationConfig = None,
        reasoning_override: str = None,
        discussion_cycles: int = 2,
        role_models: dict = None,
        role_providers: dict = None,
        allow_provider_fallback: bool = False,
    ):
        """role_models: optional {"werewolf": <alias-or-model-id>,
        "villager": ..., "seer": ...} for heterogeneous games (separates
        deception production from detection). Roles omitted fall back to
        the villager entry; providers/keys are resolved per role via the
        registry. role_providers injects pre-built providers per role
        (tests / advanced callers) and takes precedence."""
        self.n_players = n_players
        self.n_wolves = n_wolves
        self.n_seers = n_seers
        self.seed = seed
        self.output_dir = output_dir
        self.api_key = api_key
        self.model = model
        if role_models:
            unknown = set(role_models) - {"werewolf", "villager", "seer"}
            if unknown:
                raise ValueError(f"Unknown roles in role_models: {sorted(unknown)}")
            if "villager" not in role_models:
                raise ValueError('role_models requires at least a "villager" entry')
        self.belief_snapshots = belief_snapshots
        from werewolf.llm.registry import build_provider, effective_generation_config, resolve

        requested_generation = generation_config or GenerationConfig()
        normalized_reasoning = reasoning_effort
        if requested_generation.reasoning_effort is not None:
            if normalized_reasoning is not None:
                raise ValueError("Multiple legacy reasoning settings were provided")
            normalized_reasoning = requested_generation.reasoning_effort
        if reasoning_override is not None:
            if normalized_reasoning is not None and normalized_reasoning != reasoning_override:
                raise ValueError("Conflicting reasoning settings were provided")
            normalized_reasoning = reasoning_override
        self.requested_generation_config = replace(
            requested_generation, reasoning_effort=None,
        )
        self.reasoning_override = normalized_reasoning
        self.allow_provider_fallback = allow_provider_fallback
        self._closed = False
        if discussion_cycles < 1:
            raise ValueError("discussion_cycles must be >= 1")
        self.discussion_cycles = discussion_cycles

        self.rng = random.Random(seed)
        self.players = assign_roles(n_players, n_wolves, self.rng, n_seers=n_seers)

        self.state = GameState(
            seed=seed,
            round=0,
            phase="setup",
            players=self.players,
            settings={
                "n_players": n_players,
                "n_wolves": n_wolves,
                "n_seers": n_seers,
            },
            rng=self.rng
        )

        self.logger = JSONLLogger(output_dir, self.state.game_id)
        try:
            self.ledger = ledger or UsageLedger(sink=self.logger.log_llm_call)
            run_context = {
                "game_id": self.state.game_id,
                "seed": seed,
                "batch_id": batch_id,
                "trial_index": trial_index,
                "prompt_version": get_prompt_version(),
            }
        except Exception:
            self.close()
            raise
        self.role_models_resolved = None
        try:
            if role_models:
                self.agents, self.role_models_resolved = self._create_role_agents(
                    role_models, role_providers or {}, show_prompts, run_context,
                )
            else:
                spec = resolve(model_alias or model)
                self.model = spec.model
                selected_provider = provider
                if selected_provider is None and api_key:
                    build = build_provider(spec, api_key=api_key)
                    selected_provider = build.provider
                    if not build.ok and not self.allow_provider_fallback:
                        raise RuntimeError(
                            f"Provider is unavailable ({build.status.value}): "
                            f"{build.error or 'no details'}"
                        )
                if selected_provider is None and not self.allow_provider_fallback:
                    raise RuntimeError(
                        "Provider is unavailable; set allow_provider_fallback=True "
                        "only for explicit fallback tests."
                    )
                self.generation_config = effective_generation_config(
                    self.requested_generation_config, spec, self.reasoning_override,
                )
                self.agents: dict[int, AIAgent] = create_agents(
                    self.players, "", spec.model, show_prompts,
                    provider=selected_provider,
                    ledger=self.ledger,
                    run_context=run_context,
                    model_alias=model_alias,
                    generation=self.generation_config,
                )
                assignment = {
                    "alias": spec.alias,
                    "requested_model": spec.model,
                    "provider": spec.provider,
                    "registry_reasoning_default": spec.reasoning_effort,
                    "requested_reasoning_override": self.reasoning_override,
                    "effective_generation": self.generation_config.to_json_dict(),
                }
                self.role_models_resolved = {
                    role: {**assignment, "active": role != "seer" or n_seers > 0}
                    for role in ("werewolf", "villager", "seer")
                }
        except Exception:
            self.logger.close()
            self._closed = True
            raise
        try:
            self.transcript = ConsoleTranscript(
                show_all=show_all_channels,
                enabled=transcript_enabled
            )

            self._phase_index = 0
            self._pending_victim_id: Optional[int] = None

            self.logger.log_config({
                "seed": seed,
                "n_players": n_players,
                "n_wolves": n_wolves,
                "n_seers": n_seers,
                "model": self.model,
                "model_alias": model_alias,
                "prompt_version": run_context["prompt_version"],
                "batch_id": batch_id,
                "trial_index": trial_index,
                "belief_snapshots": belief_snapshots,
                "belief_schema_version": BELIEF_SCHEMA_VERSION if belief_snapshots else None,
                "event_schema_version": EVENT_SCHEMA_VERSION,
                "generation_config": self.generation_config.to_json_dict(),
                "requested_generation_config": self.requested_generation_config.to_json_dict(),
                "requested_reasoning_override": self.reasoning_override,
                "discussion_cycles": self.discussion_cycles,
                "role_models": self.role_models_resolved,
                "limits": limits_dict(),
                "code_commit": get_code_commit(),
                "game_id": self.state.game_id,
                "role_map": {
                    str(pid): {"role": p.role, "team": p.team}
                    for pid, p in sorted(self.players.items())
                },
            })
        except Exception:
            self.close()
            raise

    def _create_role_agents(
        self, role_models: dict, role_providers: dict,
        show_prompts: bool, run_context: dict,
    ) -> tuple[dict, dict]:
        """Heterogeneous agents: each role gets its own model spec and
        provider (keys resolved from that spec's env vars). Roles absent
        from role_models inherit the villager entry."""
        from werewolf.llm.registry import (
            build_provider,
            effective_generation_config,
            resolve,
        )

        specs, providers, resolved = {}, {}, {}
        provider_cache: dict = {}
        for role in ("werewolf", "villager", "seer"):
            name = role_models.get(role) or role_models["villager"]
            spec = resolve(name)
            specs[role] = spec
            if role in role_providers:
                selected = role_providers[role]
                if selected is None and not self.allow_provider_fallback:
                    raise RuntimeError(
                        f"Provider for role {role} is unavailable; "
                        "set allow_provider_fallback=True for intentional fallback."
                    )
                providers[role] = selected
            else:
                cache_key = (spec.provider, spec.model, spec.api_key_env)
                if cache_key not in provider_cache:
                    build = build_provider(spec)
                    if not build.ok and not self.allow_provider_fallback:
                        raise RuntimeError(
                            f"Provider for role {role} is unavailable "
                            f"({build.status.value}): {build.error or 'no details'}"
                        )
                    provider_cache[cache_key] = build.provider
                providers[role] = provider_cache[cache_key]
            effective = effective_generation_config(
                self.requested_generation_config, spec, self.reasoning_override,
            )
            resolved[role] = {
                "requested": name,
                "model": spec.model,
                "alias": spec.alias,
                "provider": spec.provider,
                "reasoning_effort": effective.reasoning_effort,
                "requested_model": spec.model,
                "registry_reasoning_default": spec.reasoning_effort,
                "requested_reasoning_override": self.reasoning_override,
                "effective_generation": effective.to_json_dict(),
                "active": role != "seer" or self.n_seers > 0,
            }

        wolf_roster = [p.id for p in self.players.values()
                       if p.role == "werewolf"]
        agents = {}
        for pid, player in self.players.items():
            spec = specs[player.role]
            agents[pid] = AIAgent(
                player_id=pid,
                role=player.role,
                team=player.team,
                provider=providers[player.role],
                wolf_roster=wolf_roster if player.role == "werewolf" else None,
                model=spec.model,
                show_prompts=show_prompts,
                ledger=self.ledger,
                run_context=run_context,
                model_alias=spec.alias,
                generation=effective_generation_config(
                    self.requested_generation_config,
                    spec,
                    self.reasoning_override,
                ),
            )
        # There is no single effective configuration in a heterogeneous game.
        self.generation_config = self.requested_generation_config
        return agents, resolved

    def run(self) -> str:
        self.transcript.print_role_reveal(self.players)

        while self.state.winner is None:
            self.state.round += 1
            self._run_night()

            winner = self.state.check_win_condition()
            if winner:
                self._end_game(winner)
                break

            self._run_day()

            winner = self.state.check_win_condition()
            if winner:
                self._end_game(winner)
                break

        self.close()
        return self.state.winner

    def run_next_phase(self) -> dict:
        """Run a single phase and return phase events. Used by web UI."""
        if self.state.winner is not None:
            self.close()
            return {"done": True, "winner": self.state.winner}
        while True:
            event_start_idx = len(self.state.events)
            phase_name = self.PHASE_ORDER[self._phase_index]
            should_return_phase = True

            if phase_name == "night_wolf_chat":
                self.state.round += 1
                self._set_phase("night_wolf_chat")
                self.transcript.print_phase_header(self.state.round, self.state.phase)
                self._wolf_chat()

            elif phase_name == "night_wolf_kill":
                self._set_phase("night_wolf_kill")
                self._pending_victim_id = self._wolf_kill_vote()

            elif phase_name == "night_seer":
                seer = self.state.get_seer()
                if seer and seer.alive:
                    self._set_phase("night_seer")
                    self._seer_divine(seer.id)
                else:
                    should_return_phase = False
                if self._pending_victim_id is not None:
                    self._kill_player(self._pending_victim_id, "wolf_kill")
                    self._pending_victim_id = None

            elif phase_name == "day_announce":
                self._set_phase("day_announce")
                self.transcript.print_phase_header(self.state.round, self.state.phase)
                self._log_game_status()
                winner = self.state.check_win_condition()
                if winner:
                    self._end_game(winner)

            elif phase_name == "day_assess":
                if self.belief_snapshots and self.state.winner is None:
                    # direct assignment: no public phase_change event
                    self.state.phase = "day_assess"
                    self._collect_belief_snapshots(CHECKPOINT_PRE)
                else:
                    should_return_phase = False

            elif phase_name == "day_discuss":
                self._set_phase("day_discuss")
                self._day_discussion()

            elif phase_name == "day_vote":
                self._set_phase("day_vote")
                eliminated_id = self._day_vote()
                if eliminated_id is not None:
                    self._kill_player(eliminated_id, "vote_elimination")
                self._log_game_status()
                winner = self.state.check_win_condition()
                if winner:
                    self._end_game(winner)

            self._phase_index = (self._phase_index + 1) % len(self.PHASE_ORDER)
            if not should_return_phase:
                continue

            phase_events = self.state.events[event_start_idx:]
            if self.state.winner is not None:
                self.close()
            return {
                "done": self.state.winner is not None,
                "winner": self.state.winner,
                "phase": phase_name,
                "phase_events": phase_events
            }

    def get_state_dict(self) -> dict:
        """Return serializable game state for API."""
        return {
            "game_id": self.state.game_id,
            "round": self.state.round,
            "phase": self.state.phase,
            "winner": self.state.winner,
            "settings": self.state.settings,
            "players": [
                {
                    "id": p.id,
                    "alive": p.alive,
                    "role": p.role,
                    "team": p.team,
                }
                for p in self.players.values()
            ],
            "events": self.state.events,
            "alive_wolves": len(self.state.get_alive_wolves()),
            "alive_villagers": len(self.state.get_alive_villagers()),
            "model_assignment": self.role_models_resolved,
            "observed_resolved_models": self._observed_resolved_models(),
        }

    def _observed_resolved_models(self) -> dict[str, list[str]]:
        observed: dict[str, set[str]] = {
            "werewolf": set(), "villager": set(), "seer": set(),
        }
        for record in self.ledger.records:
            if record.resolved_model and record.context.player_role in observed:
                observed[record.context.player_role].add(record.resolved_model)
        return {role: sorted(models) for role, models in observed.items()}

    def close(self) -> None:
        """Release game resources. Safe to call more than once."""
        if self._closed:
            return
        self.logger.close()
        self._closed = True

    def _run_night(self):
        self._set_phase("night_wolf_chat")
        self.transcript.print_phase_header(self.state.round, self.state.phase)
        self._wolf_chat()

        self._set_phase("night_wolf_kill")
        victim_id = self._wolf_kill_vote()

        seer = self.state.get_seer()
        if seer and seer.alive:
            self._set_phase("night_seer")
            self._seer_divine(seer.id)

        if victim_id is not None:
            self._kill_player(victim_id, "wolf_kill")

    def _run_day(self):
        self._set_phase("day_announce")
        self.transcript.print_phase_header(self.state.round, self.state.phase)
        self._log_game_status()

        if self.belief_snapshots:
            # No _set_phase(): that would log a PUBLIC phase_change event,
            # making instrumented games observably different to players.
            self.state.phase = "day_assess"
            self._collect_belief_snapshots(CHECKPOINT_PRE)

        self._set_phase("day_discuss")
        self._day_discussion()

        self._set_phase("day_vote")
        eliminated_id = self._day_vote()

        if eliminated_id is not None:
            self._kill_player(eliminated_id, "vote_elimination")

        self._log_game_status()

    def _set_phase(self, phase: str):
        self.state.phase = phase
        event = create_phase_event(self.state, phase)
        self.logger.log_event(event)

    def _wolf_chat(self):
        alive_wolves = self.state.get_alive_wolves()
        for wolf in alive_wolves:
            observation = build_observation(self.state, wolf.id, "wolf_chat")
            response = self._get_agent_action(wolf.id, observation)
            source_call_id = response.get("_source_call_id")

            if response.get("thought"):
                event = create_thought_event(
                    self.state, wolf.id, response["thought"],
                    source_call_id=source_call_id,
                )
                self.logger.log_event(event)
                self.transcript.print_event(event, self.players)

            if response.get("say") and response["say"].get("werewolf"):
                text, truncated_from = truncate_text(
                    response["say"]["werewolf"], WOLF_MESSAGE_MAX_CHARS
                )
                event = create_message_event(
                    self.state, "werewolf", wolf.id, text,
                    truncated_from=truncated_from,
                    source_call_id=source_call_id,
                )
                self.logger.log_event(event)
                self.transcript.print_event(event, self.players)

            update_player_seen_index(self.state, wolf.id)

    def _wolf_kill_vote(self) -> Optional[int]:
        alive_wolves = self.state.get_alive_wolves()
        kill_votes = {}
        source_call_ids = []

        for wolf in alive_wolves:
            observation = build_observation(self.state, wolf.id, "choose_wolf_kill")
            response = self._get_agent_action(wolf.id, observation)
            source_call_id = response.get("_source_call_id")
            if source_call_id:
                source_call_ids.append(source_call_id)

            if response.get("thought"):
                event = create_thought_event(
                    self.state, wolf.id, response["thought"],
                    source_call_id=source_call_id,
                )
                self.logger.log_event(event)
                self.transcript.print_event(event, self.players)

            action = response.get("action", {})
            target = self._coerce_player_id(action.get("kill_target"))
            if target is not None:
                kill_votes[wolf.id] = target

            update_player_seen_index(self.state, wolf.id)

        if not kill_votes:
            return None

        vote_counts = Counter(kill_votes.values())
        max_votes = max(vote_counts.values())
        candidates = [t for t, c in vote_counts.items() if c == max_votes]
        victim_id = self.rng.choice(candidates)

        event = create_kill_event(
            self.state, victim_id, kill_votes,
            source_call_ids=source_call_ids,
        )
        self.logger.log_event(event)

        return victim_id

    def _seer_divine(self, seer_id: int):
        observation = build_observation(self.state, seer_id, "seer_divine")
        response = self._get_agent_action(seer_id, observation)
        source_call_id = response.get("_source_call_id")

        if response.get("thought"):
            event = create_thought_event(
                self.state, seer_id, response["thought"],
                source_call_id=source_call_id,
            )
            self.logger.log_event(event)
            self.transcript.print_event(event, self.players)

        action = response.get("action", {})
        target_id = self._coerce_player_id(action.get("divine_target"))

        if target_id is not None and target_id in self.players:
            target = self.players[target_id]
            is_werewolf = target.role == "werewolf"
            event = create_divine_result_event(
                self.state, seer_id, target_id, is_werewolf,
                source_call_id=source_call_id,
            )
            self.logger.log_event(event)
            self.transcript.print_event(event, self.players)

        update_player_seen_index(self.state, seer_id)

    def _day_discussion(self):
        """Multi-cycle discussion with seeded, counterbalanced order.

        A single ascending-ID pass gives the first speaker no one to react
        to and the last speaker an unrebuttable word - a huge seat
        advantage that confounds adaptation measurements. Instead: the
        speaking order is shuffled deterministically per (seed, round) by
        a dedicated RNG (independent of the game RNG, so fallback behavior
        is unaffected), and even-numbered cycles reverse it, giving every
        player a chance both to react and to be rebutted."""
        alive_ids = [p.id for p in self.state.get_alive_players()]
        order_rng = random.Random(f"{self.seed}:order:{self.state.round}")
        base_order = list(alive_ids)
        order_rng.shuffle(base_order)

        for cycle in range(1, self.discussion_cycles + 1):
            order = base_order if cycle % 2 == 1 else list(reversed(base_order))
            spoken = []
            for position, player_id in enumerate(order):
                turn_context = {
                    "speaking_order": order,
                    "your_position": f"{position + 1} of {len(order)}",
                    "already_spoken": spoken.copy(),
                    "yet_to_speak": order[position + 1:],
                    "discussion_cycle": cycle,
                    "total_cycles": self.discussion_cycles,
                }
                observation = build_observation(
                    self.state, player_id, "speak_public", turn_context
                )
                response = self._get_agent_action(player_id, observation)
                source_call_id = response.get("_source_call_id")

                if response.get("thought"):
                    event = create_thought_event(
                        self.state, player_id, response["thought"],
                        source_call_id=source_call_id,
                        discussion_cycle=cycle,
                    )
                    self.logger.log_event(event)
                    self.transcript.print_event(event, self.players)

                if response.get("say") and response["say"].get("public"):
                    text, truncated_from = truncate_text(
                        response["say"]["public"], PUBLIC_MESSAGE_MAX_CHARS
                    )
                    event = create_message_event(
                        self.state, "public", player_id, text,
                        truncated_from=truncated_from,
                        meta={
                            "discussion_cycle": cycle,
                            "speaker_position": position + 1,
                        },
                        source_call_id=source_call_id,
                        discussion_cycle=cycle,
                    )
                    self.logger.log_event(event)
                    self.transcript.print_event(event, self.players)

                update_player_seen_index(self.state, player_id)
                spoken.append(player_id)

    def _collect_belief_snapshots(self, checkpoint: str):
        """One private assess_beliefs call per alive player.

        STRICTLY READ-ONLY with respect to game state: no memory update,
        no observation-cursor advance (players re-see the same events in
        their real turn), and no player-visible events. The public trace
        of an instrumented game must be identical to an uninstrumented
        one (enforced by regression test)."""
        for player in self.state.get_alive_players():
            observation = build_observation(self.state, player.id, "assess_beliefs")
            response = self._get_agent_action(
                player.id, observation, update_memory=False
            )
            source_call_id = response.get("_source_call_id")

            if response.get("thought"):
                event = create_thought_event(
                    self.state, player.id, response["thought"],
                    source_call_id=source_call_id,
                )
                self.logger.log_event(event)
                self.transcript.print_event(event, self.players)

            self._emit_belief_snapshot(
                player.id, response.get("beliefs"), checkpoint,
                source_call_id=source_call_id,
            )
            # deliberately NOT calling update_player_seen_index

    def _emit_belief_snapshot(
        self, player_id: int, raw_beliefs, checkpoint: str,
        source_call_id: Optional[str] = None,
    ):
        alive_ids = [p.id for p in self.state.get_alive_players()]
        snapshot = parse_belief_snapshot(
            raw_beliefs,
            checkpoint,
            self_id=player_id,
            alive_ids=alive_ids,
            is_wolf=self.players[player_id].role == "werewolf",
        )
        event = create_belief_snapshot_event(
            self.state, player_id, snapshot.to_payload(),
            source_call_id=source_call_id,
        )
        self.logger.log_event(event)

    def _day_vote(self) -> Optional[int]:
        alive_players = self.state.get_alive_players()
        votes = {}

        for player in alive_players:
            observation = build_observation(self.state, player.id, "vote")
            response = self._get_agent_action(player.id, observation)
            source_call_id = response.get("_source_call_id")

            if response.get("thought"):
                event = create_thought_event(
                    self.state, player.id, response["thought"],
                    source_call_id=source_call_id,
                )
                self.logger.log_event(event)
                self.transcript.print_event(event, self.players)

            if self.belief_snapshots:
                # Post-discussion snapshot rides inside the vote response;
                # a malformed one never invalidates the vote itself.
                self._emit_belief_snapshot(
                    player.id, response.get("beliefs"), CHECKPOINT_POST,
                    source_call_id=source_call_id,
                )

            action = response.get("action", {})
            target = self._coerce_player_id(action.get("vote_target"))
            if target is not None:
                votes[player.id] = target
                event = create_vote_event(
                    self.state, player.id, target,
                    source_call_id=source_call_id,
                )
                self.logger.log_event(event)
                self.transcript.print_event(event, self.players)

            update_player_seen_index(self.state, player.id)

        if not votes:
            return None

        vote_counts = Counter(votes.values())
        max_votes = max(vote_counts.values())
        candidates = [t for t, c in vote_counts.items() if c == max_votes]

        if len(candidates) > 1:
            event = create_runoff_announcement_event(
                self.state, candidates, dict(vote_counts)
            )
            self.logger.log_event(event)
            self.transcript.print_event(event, self.players)

            eliminated_id, final_vote_counts = self._runoff_vote(candidates)
            if eliminated_id is None:
                return None
        else:
            eliminated_id = candidates[0]
            final_vote_counts = dict(vote_counts)

        eliminated_role = self.players[eliminated_id].role
        event = create_elimination_event(
            self.state, eliminated_id, eliminated_role, final_vote_counts
        )
        self.logger.log_event(event)
        self.transcript.print_event(event, self.players)

        return eliminated_id

    def _runoff_vote(self, candidates: list[int]) -> tuple[Optional[int], dict]:
        alive_players = self.state.get_alive_players()
        votes = {}

        for player in alive_players:
            turn_context = {"runoff_candidates": candidates}
            observation = build_observation(
                self.state, player.id, "runoff_vote", turn_context
            )
            response = self._get_agent_action(player.id, observation)
            source_call_id = response.get("_source_call_id")

            if response.get("thought"):
                event = create_thought_event(
                    self.state, player.id, response["thought"],
                    source_call_id=source_call_id,
                )
                self.logger.log_event(event)
                self.transcript.print_event(event, self.players)

            action = response.get("action", {})
            target = self._coerce_player_id(action.get("vote_target"))
            if target is not None:
                votes[player.id] = target
                event = create_vote_event(
                    self.state, player.id, target,
                    source_call_id=source_call_id,
                )
                self.logger.log_event(event)
                self.transcript.print_event(event, self.players)

            update_player_seen_index(self.state, player.id)

        if not votes:
            event = create_no_elimination_event(self.state, candidates)
            self.logger.log_event(event)
            self.transcript.print_event(event, self.players)
            return None, {}

        vote_counts = Counter(votes.values())
        max_votes = max(vote_counts.values())
        runoff_winners = [t for t, c in vote_counts.items() if c == max_votes]

        if len(runoff_winners) == 1:
            return runoff_winners[0], dict(vote_counts)

        event = create_no_elimination_event(self.state, runoff_winners)
        self.logger.log_event(event)
        self.transcript.print_event(event, self.players)
        return None, {}

    def _kill_player(self, player_id: int, cause: str):
        player_id = self._coerce_player_id(player_id)
        if player_id is None or player_id not in self.players:
            raise ValueError(f"Cannot kill unknown player id: {player_id}")
        player = self.players[player_id]
        player.alive = False

        event = create_death_announcement_event(
            self.state, player_id, player.role, cause
        )
        self.logger.log_event(event)
        self.transcript.print_event(event, self.players)

    def _get_agent_action(
        self, player_id: int, observation: dict, update_memory: bool = True
    ) -> dict:
        agent = self.agents[player_id]

        def validator(obs, resp):
            return validate_action(obs, resp, self.state)

        def fallback(obs):
            return get_fallback_action(obs, self.rng)

        return agent.act(
            observation, validator, fallback, self.rng,
            update_memory=update_memory,
        )

    def _coerce_player_id(self, value) -> Optional[int]:
        if value is None:
            return None
        try:
            return _to_int(value)
        except (TypeError, ValueError):
            return None

    def _log_game_status(self):
        alive_wolves = len(self.state.get_alive_wolves())
        alive_villagers = len(self.state.get_alive_villagers())
        event = create_game_status_event(self.state, alive_wolves, alive_villagers)
        self.logger.log_event(event)
        self.transcript.print_event(event, self.players)

    def _end_game(self, winner: str):
        self.state.winner = winner
        remaining = [p.id for p in self.state.get_alive_players()]

        event = create_win_event(self.state, winner, remaining)
        self.logger.log_event(event)
        self.transcript.print_event(event, self.players)

        self.logger.log_usage_summary(self.ledger.game_summary())
        self.logger.log_outcome(winner, self.state.round, remaining)
