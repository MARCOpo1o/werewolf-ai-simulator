"""Regression tests: belief instrumentation must be invisible to the game.

An instrumented game and an uninstrumented game at the same seed, with a
deterministic provider, must produce identical player-visible traces:
identical prompts for every real action, identical public events, votes,
winner, memory, and observation cursors.
"""
import json
import tempfile
import unittest

from werewolf.engine.beliefs import CHECKPOINT_PRE
from werewolf.engine.game import GameEngine
from werewolf.llm.fake_provider import FakeProvider, success_result
from tests.test_belief_snapshots_engine import full_beliefs_response


def scripted_response():
    response = full_beliefs_response()
    # include memory so the memory-mutation bug would be caught
    response["updated_memory"] = {"note": "from a real action"}
    return response


def run_game(belief_snapshots: bool):
    provider = FakeProvider(
        default=success_result(scripted_response(), cost_ticks=10)
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = GameEngine(
            n_players=4, n_wolves=1, n_seers=0, seed=23,
            output_dir=tmpdir, api_key="", provider=provider,
            transcript_enabled=False, show_all_channels=False,
            belief_snapshots=belief_snapshots,
        )
        winner = engine.run()
        with open(engine.logger.filepath, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
    return engine, provider, winner, rows


def player_visible_events(rows):
    """Events any player could ever observe (non-moderator channels)."""
    return [
        (e["type"], e["channel"], e["speaker_id"], json.dumps(e["payload"], sort_keys=True))
        for e in (r["event"] for r in rows if r["type"] == "event")
        if e["channel"] != "moderator_only"
    ]


class InstrumentationIsolationTests(unittest.TestCase):
    def setUp(self):
        self.engine_on, self.provider_on, self.winner_on, self.rows_on = run_game(True)
        self.engine_off, self.provider_off, self.winner_off, self.rows_off = run_game(False)

    def test_identical_public_trace_votes_and_winner(self):
        self.assertEqual(self.winner_on, self.winner_off)
        self.assertEqual(
            player_visible_events(self.rows_on),
            player_visible_events(self.rows_off),
        )

    def test_identical_prompts_for_all_real_actions(self):
        """Every non-assessment prompt must be byte-identical: snapshots
        may not consume events, inject memory, or add visible phases."""
        def real_action_prompts(provider):
            return [
                r.user_prompt for r in provider.requests
                if "Private Assessment" not in r.user_prompt
            ]
        self.assertEqual(
            real_action_prompts(self.provider_on),
            real_action_prompts(self.provider_off),
        )
        # sanity: the instrumented run did make assessment calls
        self.assertGreater(
            len(self.provider_on.requests), len(self.provider_off.requests)
        )

    def test_identical_final_memory_and_cursors(self):
        for pid in self.engine_on.agents:
            self.assertEqual(
                self.engine_on.agents[pid].memory,
                self.engine_off.agents[pid].memory,
                f"P{pid} memory diverged",
            )

    def test_assessment_alone_mutates_nothing(self):
        provider = FakeProvider(
            default=success_result(scripted_response(), cost_ticks=10)
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = GameEngine(
                n_players=4, n_wolves=1, n_seers=0, seed=23,
                output_dir=tmpdir, api_key="", provider=provider,
                transcript_enabled=False, show_all_channels=False,
            )
            engine.state.round = 1
            cursors_before = {
                p.id: p.last_seen_event_idx for p in engine.players.values()
            }
            engine._collect_belief_snapshots(CHECKPOINT_PRE)

            # read-only: no memory writes, no cursor movement
            for pid, agent in engine.agents.items():
                self.assertEqual(agent.memory, {}, f"P{pid} memory mutated")
            for p in engine.players.values():
                self.assertEqual(
                    p.last_seen_event_idx, cursors_before[p.id],
                    f"P{p.id} cursor advanced",
                )
            # ...but snapshots were produced
            snapshots = [e for e in engine.state.events
                         if e["type"] == "belief_snapshot"]
            self.assertEqual(len(snapshots), 4)
            # and nothing player-visible was created
            visible = [e for e in engine.state.events
                       if e["channel"] != "moderator_only"]
            self.assertEqual(visible, [])


if __name__ == "__main__":
    unittest.main()
