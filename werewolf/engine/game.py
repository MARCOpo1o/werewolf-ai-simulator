import random
from collections import Counter
from typing import Optional

from werewolf.engine.state import GameState, PlayerState
from werewolf.engine.events import (
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
)
from werewolf.engine.visibility import build_observation, update_player_seen_index
from werewolf.engine.validate import validate_action, get_fallback_action
from werewolf.engine.logging import JSONLLogger, ConsoleTranscript
from werewolf.roles.assign import assign_roles
from werewolf.agents.ai_agent import AIAgent, create_agents


class GameEngine:
    PHASE_ORDER = [
        "night_wolf_chat",
        "night_wolf_kill",
        "night_seer",
        "day_announce",
        "day_discuss",
        "day_vote",
    ]

    def __init__(
        self,
        n_players: int = 7,
        n_wolves: int = 2,
        seed: int = 42,
        output_dir: str = "outputs/games",
        api_key: str = "",
        model: str = "grok-4-1-fast",
        show_all_channels: bool = True,
        show_prompts: bool = False
    ):
        self.n_players = n_players
        self.n_wolves = n_wolves
        self.seed = seed
        self.output_dir = output_dir
        self.api_key = api_key
        self.model = model

        self.rng = random.Random(seed)
        self.players = assign_roles(n_players, n_wolves, self.rng)

        self.state = GameState(
            seed=seed,
            round=0,
            phase="setup",
            players=self.players,
            settings={
                "n_players": n_players,
                "n_wolves": n_wolves,
            },
            rng=self.rng
        )

        self.agents: dict[int, AIAgent] = create_agents(
            self.players, api_key, model, show_prompts
        )
        self.logger = JSONLLogger(output_dir, self.state.game_id)
        self.transcript = ConsoleTranscript(show_all=show_all_channels)

        self._phase_index = 0
        self._pending_victim_id: Optional[int] = None

        self.logger.log_config({
            "seed": seed,
            "n_players": n_players,
            "n_wolves": n_wolves,
            "model": model,
            "game_id": self.state.game_id
        })

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

        self.logger.close()
        return self.state.winner

    def run_next_phase(self) -> dict:
        """Run a single phase and return phase events. Used by web UI."""
        if self.state.winner is not None:
            return {"done": True, "winner": self.state.winner}

        event_start_idx = len(self.state.events)
        phase_name = self.PHASE_ORDER[self._phase_index]

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
        phase_events = self.state.events[event_start_idx:]

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
        }

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

            if response.get("thought"):
                event = create_thought_event(self.state, wolf.id, response["thought"])
                self.logger.log_event(event)
                self.transcript.print_event(event, self.players)

            if response.get("say") and response["say"].get("werewolf"):
                event = create_message_event(
                    self.state, "werewolf", wolf.id, response["say"]["werewolf"]
                )
                self.logger.log_event(event)
                self.transcript.print_event(event, self.players)

            update_player_seen_index(self.state, wolf.id)

    def _wolf_kill_vote(self) -> Optional[int]:
        alive_wolves = self.state.get_alive_wolves()
        kill_votes = {}

        for wolf in alive_wolves:
            observation = build_observation(self.state, wolf.id, "choose_wolf_kill")
            response = self._get_agent_action(wolf.id, observation)

            if response.get("thought"):
                event = create_thought_event(self.state, wolf.id, response["thought"])
                self.logger.log_event(event)
                self.transcript.print_event(event, self.players)

            action = response.get("action", {})
            target = action.get("kill_target")
            if target is not None:
                kill_votes[wolf.id] = target

            update_player_seen_index(self.state, wolf.id)

        if not kill_votes:
            return None

        vote_counts = Counter(kill_votes.values())
        max_votes = max(vote_counts.values())
        candidates = [t for t, c in vote_counts.items() if c == max_votes]
        victim_id = self.rng.choice(candidates)

        event = create_kill_event(self.state, victim_id, kill_votes)
        self.logger.log_event(event)

        return victim_id

    def _seer_divine(self, seer_id: int):
        observation = build_observation(self.state, seer_id, "seer_divine")
        response = self._get_agent_action(seer_id, observation)

        if response.get("thought"):
            event = create_thought_event(self.state, seer_id, response["thought"])
            self.logger.log_event(event)
            self.transcript.print_event(event, self.players)

        action = response.get("action", {})
        target_id = action.get("divine_target")

        if target_id is not None and target_id in self.players:
            target = self.players[target_id]
            is_werewolf = target.role == "werewolf"
            event = create_divine_result_event(self.state, seer_id, target_id, is_werewolf)
            self.logger.log_event(event)
            self.transcript.print_event(event, self.players)

        update_player_seen_index(self.state, seer_id)

    def _day_discussion(self):
        alive_players = self.state.get_alive_players()
        speaking_order = [p.id for p in alive_players]
        spoken = []

        for i, player in enumerate(alive_players):
            turn_context = {
                "speaking_order": speaking_order,
                "your_position": f"{i + 1} of {len(alive_players)}",
                "already_spoken": spoken.copy(),
                "yet_to_speak": speaking_order[i + 1:]
            }
            observation = build_observation(
                self.state, player.id, "speak_public", turn_context
            )
            response = self._get_agent_action(player.id, observation)

            if response.get("thought"):
                event = create_thought_event(self.state, player.id, response["thought"])
                self.logger.log_event(event)
                self.transcript.print_event(event, self.players)

            if response.get("say") and response["say"].get("public"):
                event = create_message_event(
                    self.state, "public", player.id, response["say"]["public"]
                )
                self.logger.log_event(event)
                self.transcript.print_event(event, self.players)

            update_player_seen_index(self.state, player.id)
            spoken.append(player.id)

    def _day_vote(self) -> Optional[int]:
        alive_players = self.state.get_alive_players()
        votes = {}

        for player in alive_players:
            observation = build_observation(self.state, player.id, "vote")
            response = self._get_agent_action(player.id, observation)

            if response.get("thought"):
                event = create_thought_event(self.state, player.id, response["thought"])
                self.logger.log_event(event)
                self.transcript.print_event(event, self.players)

            action = response.get("action", {})
            target = action.get("vote_target")
            if target is not None:
                votes[player.id] = target
                event = create_vote_event(self.state, player.id, target)
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

            if response.get("thought"):
                event = create_thought_event(self.state, player.id, response["thought"])
                self.logger.log_event(event)
                self.transcript.print_event(event, self.players)

            action = response.get("action", {})
            target = action.get("vote_target")
            if target is not None:
                votes[player.id] = target
                event = create_vote_event(self.state, player.id, target)
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
        player = self.players[player_id]
        player.alive = False

        event = create_death_announcement_event(
            self.state, player_id, player.role, cause
        )
        self.logger.log_event(event)
        self.transcript.print_event(event, self.players)

    def _get_agent_action(self, player_id: int, observation: dict) -> dict:
        agent = self.agents[player_id]

        def validator(obs, resp):
            return validate_action(obs, resp, self.state)

        def fallback(obs):
            return get_fallback_action(obs, self.rng)

        return agent.act(observation, validator, fallback, self.rng)

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

        self.logger.log_outcome(winner, self.state.round, remaining)
