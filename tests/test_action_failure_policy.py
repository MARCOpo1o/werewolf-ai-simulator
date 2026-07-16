import tempfile
import unittest

from werewolf.agents.ai_agent import ActionFailureAbort, AIAgent
from werewolf.engine.game import GameEngine, GameRoundLimitExceeded
from werewolf.llm.fake_provider import FakeProvider, error_result
from werewolf.llm.records import ErrorCategory


def observation(required_action="vote"):
    return {
        "required_action": required_action,
        "self": {"id": 1, "role": "villager"},
        "alive_players": [{"id": 1}, {"id": 2}, {"id": 3}],
        "round": 1,
        "phase": "day_vote",
    }


def always_valid(obs, parsed):
    return True, None


def fallback(obs):
    return {"action": {"vote_target": 2}, "say": None}


class AbortGamePolicyTests(unittest.TestCase):
    def test_strategic_fallback_aborts_under_abort_game(self):
        agent = AIAgent(
            player_id=1, role="villager", team="village",
            provider=FakeProvider(
                default=error_result(ErrorCategory.PROVIDER_ERROR),
            ),
            action_failure_policy="abort_game",
        )
        with self.assertRaises(ActionFailureAbort):
            agent.act(observation(), always_valid, fallback, rng=None)

    def test_missing_provider_aborts_under_abort_game(self):
        agent = AIAgent(
            player_id=1, role="villager", team="village",
            provider=None, action_failure_policy="abort_game",
        )
        with self.assertRaises(ActionFailureAbort):
            agent.act(observation(), always_valid, fallback, rng=None)

    def test_belief_assessment_fallback_never_aborts(self):
        agent = AIAgent(
            player_id=1, role="villager", team="village",
            provider=None, action_failure_policy="abort_game",
        )
        result = agent.act(
            observation("assess_beliefs"), always_valid, fallback,
            rng=None, update_memory=False,
        )
        self.assertIn("action", result)

    def test_default_policy_preserves_fallback_behavior(self):
        agent = AIAgent(
            player_id=1, role="villager", team="village", provider=None,
        )
        result = agent.act(observation(), always_valid, fallback, rng=None)
        self.assertEqual(result["action"], {"vote_target": 2})

    def test_unknown_policy_rejected(self):
        with self.assertRaises(ValueError):
            AIAgent(player_id=1, role="villager", team="village",
                    action_failure_policy="carry_on")
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                GameEngine(
                    seed=1, output_dir=tmp, api_key="",
                    transcript_enabled=False,
                    allow_provider_fallback=True,
                    action_failure_policy="carry_on",
                )


class MaxRoundsTests(unittest.TestCase):
    def test_round_limit_aborts_instead_of_fabricating_outcome(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = GameEngine(
                n_players=7, n_wolves=2, n_seers=0, seed=123,
                output_dir=tmp, api_key="",
                transcript_enabled=False, show_all_channels=False,
                allow_provider_fallback=True, belief_snapshots=False,
                max_rounds=1,
            )
            try:
                with self.assertRaises(GameRoundLimitExceeded):
                    engine.run()
            finally:
                engine.close()

    def test_generous_limit_does_not_change_games(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = GameEngine(
                n_players=7, n_wolves=2, n_seers=0, seed=123,
                output_dir=tmp, api_key="",
                transcript_enabled=False, show_all_channels=False,
                allow_provider_fallback=True, belief_snapshots=False,
                max_rounds=20,
            )
            winner = engine.run()
            self.assertIn(winner, ("wolf", "village"))

    def test_config_records_policy_and_limit(self):
        import json
        with tempfile.TemporaryDirectory() as tmp:
            engine = GameEngine(
                n_players=7, n_wolves=2, n_seers=0, seed=5,
                output_dir=tmp, api_key="",
                transcript_enabled=False, show_all_channels=False,
                allow_provider_fallback=True, belief_snapshots=False,
                max_rounds=20,
            )
            engine.close()
            with open(engine.logger.filepath, encoding="utf-8") as f:
                config = json.loads(f.readline())
            self.assertEqual(config["max_rounds"], 20)
            self.assertEqual(config["action_failure_policy"], "fallback")


if __name__ == "__main__":
    unittest.main()
