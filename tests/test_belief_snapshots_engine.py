import json
import tempfile
import unittest

from werewolf.engine.game import GameEngine
from werewolf.engine.validate import validate_action
from werewolf.llm.fake_provider import FakeProvider, success_result


def full_beliefs_response():
    """Static response valid as an assess_beliefs answer for ANY player
    while everyone is alive: probability keys cover all ids (the player's
    own id is reported as an 'unexpected key' problem but completeness
    over the others still holds), and second-order beliefs are included
    so wolves validate too."""
    prob_map = {"0": 0.1, "1": 0.2, "2": 0.3, "3": 0.4}
    return {
        "thought": "assessing",
        "say": None,
        "action": None,
        "beliefs": {
            "wolf_probabilities": prob_map,
            "intended_vote": None,
            "vote_confidence": 0.5,
            "most_influential_recent_speaker": None,
            "estimated_suspicion_of_me": prob_map,
        },
    }


class EngineSnapshotTests(unittest.TestCase):
    def _run(self, belief_snapshots=True):
        provider = FakeProvider(
            default=success_result(full_beliefs_response(), cost_ticks=100)
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = GameEngine(
                n_players=4, n_wolves=1, n_seers=0, seed=11,
                output_dir=tmpdir, api_key="", provider=provider,
                transcript_enabled=False, show_all_channels=False,
                belief_snapshots=belief_snapshots,
            )
            winner = engine.run()
            with open(engine.logger.filepath, encoding="utf-8") as f:
                rows = [json.loads(line) for line in f if line.strip()]
            return winner, rows, provider

    def test_snapshots_emitted_and_moderator_only(self):
        winner, rows, provider = self._run()
        self.assertIn(winner, {"wolf", "village"})

        events = [r["event"] for r in rows if r["type"] == "event"]
        snapshots = [e for e in events if e["type"] == "belief_snapshot"]
        self.assertGreater(len(snapshots), 0)
        for e in snapshots:
            self.assertEqual(e["channel"], "moderator_only")
            payload = e["payload"]
            self.assertIn(payload["checkpoint"],
                          ("pre_discussion", "post_discussion"))
            self.assertIn("valid", payload)

        # every pre-discussion snapshot from the scripted response is valid
        pre = [e for e in snapshots
               if e["payload"]["checkpoint"] == "pre_discussion"]
        self.assertTrue(all(e["payload"]["valid"] for e in pre))
        # scripted response has no vote_target -> votes fall back ->
        # post snapshots exist but are explicitly missing, never fabricated
        post = [e for e in snapshots
                if e["payload"]["checkpoint"] == "post_discussion"]
        self.assertGreater(len(post), 0)
        for e in post:
            self.assertFalse(e["payload"]["valid"])
            self.assertEqual(e["payload"]["invalid_reason"], "missing")
            self.assertEqual(e["payload"]["wolf_probabilities"], {})

        # config records the instrumentation state
        config = next(r for r in rows if r["type"] == "config")
        self.assertTrue(config["belief_snapshots"])
        self.assertEqual(config["belief_schema_version"], 1)

    def test_snapshots_never_visible_to_players(self):
        _, rows, provider = self._run()
        # No prompt sent to any player may contain snapshot internals.
        for request in provider.requests:
            self.assertNotIn("belief_snapshot", request.user_prompt)
            self.assertNotIn("estimated_suspicion_of_me\": {\"0\": 0.1",
                             request.user_prompt)

    def test_disabled_flag_removes_all_instrumentation(self):
        _, rows, provider = self._run(belief_snapshots=False)
        events = [r["event"] for r in rows if r["type"] == "event"]
        self.assertFalse(
            [e for e in events if e["type"] == "belief_snapshot"]
        )
        self.assertFalse(
            [e for e in events if e.get("payload", {}).get("phase") == "day_assess"]
        )
        for request in provider.requests:
            self.assertNotIn("Private Assessment", request.user_prompt)
        config = next(r for r in rows if r["type"] == "config")
        self.assertFalse(config["belief_snapshots"])

    def test_step_mode_includes_day_assess_phase(self):
        provider = FakeProvider(
            default=success_result(full_beliefs_response(), cost_ticks=1)
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = GameEngine(
                n_players=4, n_wolves=1, n_seers=0, seed=11,
                output_dir=tmpdir, api_key="", provider=provider,
                transcript_enabled=False, show_all_channels=False,
            )
            phases = []
            for _ in range(40):
                result = engine.run_next_phase()
                phases.append(result["phase"])
                if result["done"]:
                    break
            self.assertIn("day_assess", phases)


class SoftVoteValidationTests(unittest.TestCase):
    def test_valid_vote_with_garbage_beliefs_is_accepted(self):
        observation = {
            "required_action": "vote",
            "self": {"id": 0, "role": "villager", "team": "village"},
            "alive_players": [{"id": 0}, {"id": 1}, {"id": 2}],
            "round": 1, "phase": "day_vote",
        }
        response = {
            "thought": "t", "say": None,
            "action": {"vote_target": 2},
            "beliefs": "completely malformed garbage",
        }
        ok, error = validate_action(observation, response, None)
        self.assertTrue(ok, error)  # instrumentation never blocks the game


if __name__ == "__main__":
    unittest.main()
