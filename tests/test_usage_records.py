import json
import unittest

from werewolf.llm.records import (
    SCHEMA_VERSION,
    TICKS_PER_USD,
    CallContext,
    CostInfo,
    CostSource,
    ErrorCategory,
    TokenUsage,
    UsageRecord,
)


def make_context(**overrides) -> CallContext:
    defaults = dict(
        game_id="game_test_1",
        round=2,
        phase="day_vote",
        required_action="vote",
        player_id=3,
        player_role="villager",
        player_team="village",
        seed=42,
    )
    defaults.update(overrides)
    return CallContext(**defaults)


class CostInfoTests(unittest.TestCase):
    def test_from_ticks_exact_conversion(self):
        cost = CostInfo.from_ticks(37_756_000)
        self.assertEqual(cost.source, CostSource.PROVIDER_REPORTED)
        self.assertEqual(cost.ticks, 37_756_000)
        self.assertAlmostEqual(cost.usd, 37_756_000 / TICKS_PER_USD)

    def test_from_ticks_rejects_bad_input(self):
        for bad in (-1, 1.5, "100", True, None):
            with self.assertRaises(ValueError):
                CostInfo.from_ticks(bad)

    def test_unavailable_has_no_fabricated_zero(self):
        cost = CostInfo.unavailable()
        self.assertEqual(cost.source, CostSource.UNAVAILABLE)
        self.assertIsNone(cost.ticks)
        self.assertIsNone(cost.usd)

    def test_estimated_requires_estimate_source(self):
        with self.assertRaises(ValueError):
            CostInfo.estimated(0.01, CostSource.PROVIDER_REPORTED)
        with self.assertRaises(ValueError):
            CostInfo.estimated(0.01, CostSource.UNAVAILABLE)
        cost = CostInfo.estimated(0.01, CostSource.PRICING_TABLE_ESTIMATE)
        self.assertEqual(cost.source, CostSource.PRICING_TABLE_ESTIMATE)
        self.assertIsNone(cost.ticks)


class UsageRecordTests(unittest.TestCase):
    def test_json_dict_shape_and_serializability(self):
        record = UsageRecord(
            context=make_context(),
            provider="xai",
            requested_model="grok-4-1-fast",
            resolved_model="grok-4.3",
            usage=TokenUsage(input_tokens=500, output_tokens=100, total_tokens=600),
            cost=CostInfo.from_ticks(12_500_000),
            api_ok=True,
            parse_ok=True,
            parse_method="direct",
            validation_ok=True,
            error_category=ErrorCategory.COMPLETED,
            latency_ms=210,
        )
        d = record.to_json_dict()
        json.dumps(d)  # must be JSON-serializable

        self.assertEqual(d["schema_version"], SCHEMA_VERSION)
        self.assertEqual(d["game_id"], "game_test_1")
        self.assertEqual(d["requested_model"], "grok-4-1-fast")
        self.assertEqual(d["resolved_model"], "grok-4.3")
        self.assertEqual(d["cost"]["ticks"], 12_500_000)
        self.assertEqual(d["cost"]["source"], "provider_reported")
        self.assertEqual(d["error_category"], "completed")
        self.assertEqual(d["usage"]["input_tokens"], 500)
        self.assertIsNone(d["usage"]["reasoning_tokens"])  # missing != 0
        self.assertTrue(d["call_id"])
        self.assertTrue(d["ts"])

    def test_missing_usage_stays_none_not_zero(self):
        record = UsageRecord(
            context=make_context(),
            provider="xai",
            requested_model="m",
        )
        d = record.to_json_dict()
        for field_name in (
            "input_tokens",
            "cached_input_tokens",
            "output_tokens",
            "reasoning_tokens",
            "total_tokens",
        ):
            self.assertIsNone(d["usage"][field_name])
        self.assertIsNone(d["cost"]["usd"])
        self.assertEqual(d["cost"]["source"], "unavailable")

    def test_metadata_scrubs_secret_keys(self):
        record = UsageRecord(
            context=make_context(),
            provider="xai",
            requested_model="m",
            provider_metadata={
                "api_key": "xai-SECRET",
                "Authorization": "Bearer xai-SECRET",
                "num_sources_used": 3,
            },
        )
        d = record.to_json_dict()
        serialized = json.dumps(d)
        self.assertNotIn("SECRET", serialized)
        self.assertEqual(d["provider_metadata"], {"num_sources_used": 3})

    def test_attempts_share_call_id_when_assigned(self):
        ctx = make_context()
        call_id = "abc123"
        first = UsageRecord(
            context=ctx, provider="xai", requested_model="m",
            call_id=call_id, attempt=1,
        )
        second = UsageRecord(
            context=ctx, provider="xai", requested_model="m",
            call_id=call_id, attempt=2,
        )
        self.assertEqual(first.call_id, second.call_id)
        self.assertEqual(second.to_json_dict()["attempt"], 2)


if __name__ == "__main__":
    unittest.main()
