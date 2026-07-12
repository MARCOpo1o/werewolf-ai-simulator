import json
import os
import random
import tempfile
import unittest

from werewolf.agents.ai_agent import AIAgent
from werewolf.engine.game import GameEngine
from werewolf.engine.validate import get_fallback_action, validate_action
from werewolf.llm.fake_provider import (
    FakeProvider,
    error_result,
    estimated_cost_result,
    success_result,
)
from werewolf.llm.ledger import UsageLedger
from werewolf.llm.provider import ProviderResult
from werewolf.llm.records import CostSource, ErrorCategory, TokenUsage


def make_observation(required_action="vote", player_id=1):
    return {
        "required_action": required_action,
        "self": {"id": player_id, "role": "villager", "team": "village"},
        "round": 1,
        "phase": "day_vote",
        "alive_players": [{"id": 1}, {"id": 2}, {"id": 3}],
        "recent_events": [],
        "private_info": {},
    }


def make_agent(provider, ledger):
    return AIAgent(
        player_id=1,
        role="villager",
        team="village",
        provider=provider,
        model="fake-model-1",
        ledger=ledger,
        run_context={"game_id": "g_test", "seed": 7, "prompt_version": "abc123"},
    )


def run_act(agent, observation):
    rng = random.Random(0)
    return agent.act(
        observation,
        validator=lambda obs, resp: validate_action(obs, resp, None),
        fallback_fn=lambda obs: get_fallback_action(obs, rng),
        rng=rng,
    )


VALID_VOTE = {"thought": "t", "say": None, "action": {"vote_target": 2}}


class AgentRecordingTests(unittest.TestCase):
    def test_valid_response_records_exact_cost(self):
        provider = FakeProvider([success_result(VALID_VOTE, cost_ticks=37_756_000)])
        ledger = UsageLedger()
        response = run_act(make_agent(provider, ledger), make_observation())

        self.assertEqual(response["action"]["vote_target"], 2)
        records = ledger.records
        self.assertEqual(len(records), 1)
        r = records[0]
        self.assertTrue(r.api_ok and r.parse_ok and r.validation_ok)
        self.assertEqual(r.parse_method, "direct")
        self.assertEqual(r.error_category, ErrorCategory.COMPLETED)
        self.assertEqual(r.cost.ticks, 37_756_000)
        self.assertEqual(r.context.game_id, "g_test")
        self.assertEqual(r.context.prompt_version, "abc123")
        self.assertEqual(r.context.required_action, "vote")

    def test_malformed_json_still_costs_money_and_is_recorded(self):
        provider = FakeProvider([
            success_result(text="sorry, no json here at all", cost_ticks=5000),
            success_result(VALID_VOTE, cost_ticks=6000),
        ])
        ledger = UsageLedger()
        response = run_act(make_agent(provider, ledger), make_observation())

        self.assertEqual(response["action"]["vote_target"], 2)
        records = ledger.records
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0].error_category, ErrorCategory.MALFORMED_JSON)
        self.assertFalse(records[0].parse_ok)
        self.assertEqual(records[0].cost.ticks, 5000)  # billed anyway
        summary = ledger.game_summary()
        self.assertEqual(summary["cost_ticks_total"], 11000)
        self.assertEqual(summary["retries"], 1)

    def test_invalid_action_then_successful_retry_shares_call_id(self):
        invalid = {"thought": "t", "say": None, "action": {"vote_target": 99}}
        provider = FakeProvider([
            success_result(invalid, cost_ticks=100),
            success_result(VALID_VOTE, cost_ticks=200),
        ])
        ledger = UsageLedger()
        response = run_act(make_agent(provider, ledger), make_observation())

        self.assertEqual(response["action"]["vote_target"], 2)
        records = ledger.records
        self.assertEqual(records[0].error_category, ErrorCategory.INVALID_GAME_ACTION)
        self.assertFalse(records[0].validation_ok)
        self.assertTrue(records[1].validation_ok)
        self.assertEqual(records[0].call_id, records[1].call_id)
        self.assertEqual([r.attempt for r in records], [1, 2])
        # validation error is fed back to the model on retry
        self.assertIn("99", provider.requests[1].user_prompt)

    def test_retryable_provider_error_retries(self):
        provider = FakeProvider([
            error_result(ErrorCategory.RATE_LIMITED),
            success_result(VALID_VOTE, cost_ticks=100),
        ])
        ledger = UsageLedger()
        response = run_act(make_agent(provider, ledger), make_observation())
        self.assertEqual(response["action"]["vote_target"], 2)
        self.assertEqual(ledger.game_summary()["api_failures"], 1)
        self.assertEqual(
            ledger.game_summary()["errors_by_category"]["rate_limited"], 1
        )

    def test_non_retryable_error_short_circuits_to_fallback(self):
        provider = FakeProvider([
            error_result(ErrorCategory.AUTHENTICATION_ERROR),
        ])
        ledger = UsageLedger()
        response = run_act(make_agent(provider, ledger), make_observation())

        # only 1 API attempt despite MAX_RETRIES=3
        self.assertEqual(provider.calls_made, 1)
        # fallback still produced a legal action
        ok, _ = validate_action(make_observation(), response, None)
        self.assertTrue(ok)
        summary = ledger.game_summary()
        self.assertEqual(summary["calls"], 1)
        self.assertEqual(summary["fallbacks"], 1)
        self.assertEqual(
            summary["errors_by_category"]["authentication_error"], 1
        )

    def test_fallback_after_three_malformed_attempts(self):
        provider = FakeProvider(
            [success_result(text="not json", cost_ticks=10)] * 3
        )
        ledger = UsageLedger()
        response = run_act(make_agent(provider, ledger), make_observation())

        ok, _ = validate_action(make_observation(), response, None)
        self.assertTrue(ok)
        summary = ledger.game_summary()
        self.assertEqual(summary["calls"], 3)
        self.assertEqual(summary["parse_failures"], 3)
        self.assertEqual(summary["fallbacks"], 1)
        self.assertEqual(summary["cost_ticks_total"], 30)  # all attempts billed

    def test_missing_usage_fields_stay_none(self):
        result = ProviderResult(ok=True, text=json.dumps(VALID_VOTE))
        provider = FakeProvider([result])
        ledger = UsageLedger()
        run_act(make_agent(provider, ledger), make_observation())

        r = ledger.records[0]
        self.assertIsNone(r.usage.input_tokens)
        self.assertIsNone(r.cost.usd)
        summary = ledger.game_summary()
        self.assertEqual(summary["calls_missing_usage"], 1)
        self.assertFalse(summary["cost_complete"])
        self.assertIsNone(summary["cost_usd_total"])

    def test_estimated_cost_labeled_not_exact(self):
        provider = FakeProvider([estimated_cost_result(VALID_VOTE, usd=0.0002)])
        ledger = UsageLedger()
        run_act(make_agent(provider, ledger), make_observation())
        r = ledger.records[0]
        self.assertEqual(r.cost.source, CostSource.PRICING_TABLE_ESTIMATE)
        self.assertIsNone(r.cost.ticks)

    def test_reasoning_and_cached_tokens_recorded(self):
        provider = FakeProvider([success_result(
            VALID_VOTE, reasoning_tokens=345, cached_input_tokens=210,
        )])
        ledger = UsageLedger()
        run_act(make_agent(provider, ledger), make_observation())
        summary = ledger.game_summary()
        self.assertEqual(summary["tokens"]["reasoning_tokens"], 345)
        self.assertEqual(summary["tokens"]["cached_input_tokens"], 210)

    def test_regex_recovery_is_tagged(self):
        provider = FakeProvider([success_result(
            text='I will vote now! "vote_target": "2" ... final answer',
            cost_ticks=50,
        )])
        ledger = UsageLedger()
        response = run_act(make_agent(provider, ledger), make_observation())
        self.assertEqual(response["action"]["vote_target"], 2)
        self.assertEqual(ledger.records[0].parse_method, "regex")

    def test_no_provider_records_missing_key_and_fallback(self):
        ledger = UsageLedger()
        response = run_act(make_agent(None, ledger), make_observation())
        ok, _ = validate_action(make_observation(), response, None)
        self.assertTrue(ok)
        summary = ledger.game_summary()
        self.assertEqual(summary["calls"], 0)  # no paid calls
        self.assertEqual(summary["cost_usd_total"], 0.0)
        self.assertEqual(summary["errors_by_category"]["missing_api_key"], 1)
        self.assertEqual(summary["errors_by_category"]["fallback_used"], 1)

    def test_no_key_material_in_records(self):
        secret = "xai-SUPER-SECRET"
        provider = FakeProvider([success_result(VALID_VOTE)])
        ledger = UsageLedger()
        run_act(make_agent(provider, ledger), make_observation())
        for r in ledger.records:
            self.assertNotIn(secret, json.dumps(r.to_json_dict()))


class EngineIntegrationTests(unittest.TestCase):
    """Full game over a FakeProvider: every attempt lands in the JSONL log
    and the usage_summary equals the sum of llm_call records."""

    def _run_game(self, tmpdir):
        # Valid for wolf_chat/speak_public; invalid for kill/divine/vote,
        # exercising retries and fallbacks in one game.
        default = success_result(
            {"thought": "hm", "say": None, "action": None}, cost_ticks=1000,
        )
        provider = FakeProvider(default=default)
        engine = GameEngine(
            n_players=5,
            n_wolves=1,
            n_seers=0,
            seed=99,
            output_dir=tmpdir,
            api_key="",
            provider=provider,
            model_alias="fake",
            transcript_enabled=False,
            show_all_channels=False,
        )
        winner = engine.run()
        return engine, provider, winner

    def test_game_records_and_summary_consistency(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            engine, provider, winner = self._run_game(tmpdir)
            self.assertIn(winner, {"wolf", "village"})

            with open(engine.logger.filepath, encoding="utf-8") as f:
                rows = [json.loads(line) for line in f if line.strip()]

            llm_calls = [r for r in rows if r["type"] == "llm_call"]
            summaries = [r for r in rows if r["type"] == "usage_summary"]
            self.assertEqual(len(summaries), 1)
            summary = summaries[0]["usage"]

            # 1. every paid attempt is in the log
            api_lines = [r for r in llm_calls if r["api_attempted"]]
            self.assertEqual(provider.calls_made, len(api_lines))
            self.assertEqual(summary["calls"], len(api_lines))

            # 3/4. exact tick totals match the individual records
            ticks_from_lines = sum(
                r["cost"]["ticks"] for r in api_lines
                if r["cost"]["ticks"] is not None
            )
            self.assertEqual(summary["cost_ticks_total"], ticks_from_lines)
            self.assertTrue(summary["cost_complete"])

            # fallbacks happened (kill/vote actions can't succeed) and are visible
            self.assertGreater(summary["fallbacks"], 0)
            self.assertGreater(summary["validation_failures"], 0)

            # ledger in memory agrees with the file
            self.assertEqual(engine.ledger.game_summary()["calls"], summary["calls"])

            # config line carries experiment metadata
            config = next(r for r in rows if r["type"] == "config")
            self.assertEqual(config["model_alias"], "fake")
            self.assertTrue(config["prompt_version"])

    def test_no_key_game_still_completes_with_zero_paid_calls(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = GameEngine(
                n_players=5, n_wolves=1, n_seers=0, seed=123,
                output_dir=tmpdir, api_key="",
                transcript_enabled=False, show_all_channels=False,
            )
            winner = engine.run()
            self.assertIn(winner, {"wolf", "village"})
            summary = engine.ledger.game_summary()
            self.assertEqual(summary["calls"], 0)
            self.assertEqual(summary["cost_usd_total"], 0.0)
            self.assertGreater(summary["fallbacks"], 0)


if __name__ == "__main__":
    unittest.main()
