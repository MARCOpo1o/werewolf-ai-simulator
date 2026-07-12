import json
import threading
import unittest

from werewolf.llm.ledger import UsageLedger
from werewolf.llm.records import (
    CallContext,
    CostInfo,
    CostSource,
    ErrorCategory,
    TokenUsage,
    UsageRecord,
)


def make_record(
    *,
    player_id=1,
    role="villager",
    phase="day_vote",
    action="vote",
    round_=1,
    attempt=1,
    ticks=None,
    est_usd=None,
    tokens=(100, 20),
    api_attempted=True,
    api_ok=True,
    parse_ok=True,
    parse_method="direct",
    validation_ok=True,
    error=None,
) -> UsageRecord:
    if ticks is not None:
        cost = CostInfo.from_ticks(ticks)
    elif est_usd is not None:
        cost = CostInfo.estimated(est_usd, CostSource.PRICING_TABLE_ESTIMATE)
    else:
        cost = CostInfo.unavailable()
    usage = TokenUsage(
        input_tokens=tokens[0] if tokens else None,
        output_tokens=tokens[1] if tokens else None,
        total_tokens=(tokens[0] + tokens[1]) if tokens else None,
    )
    return UsageRecord(
        context=CallContext(
            game_id="g1",
            round=round_,
            phase=phase,
            required_action=action,
            player_id=player_id,
            player_role=role,
            player_team="village" if role != "werewolf" else "wolf",
        ),
        provider="fake",
        requested_model="m",
        attempt=attempt,
        usage=usage,
        cost=cost,
        api_attempted=api_attempted,
        api_ok=api_ok,
        parse_ok=parse_ok if api_attempted else None,
        parse_method=parse_method if api_attempted else None,
        validation_ok=validation_ok if api_attempted else None,
        error_category=error,
    )


class LedgerRecordingTests(unittest.TestCase):
    def test_every_record_reaches_sink_in_order(self):
        seen = []
        ledger = UsageLedger(sink=seen.append)
        for i in range(5):
            ledger.record(make_record(player_id=i, ticks=1000 + i))
        self.assertEqual(len(seen), 5)
        self.assertEqual([d["player_id"] for d in seen], list(range(5)))
        for d in seen:
            json.dumps(d)

    def test_retries_counted_separately(self):
        ledger = UsageLedger()
        ledger.record(make_record(attempt=1, ticks=100, api_ok=True,
                                  parse_ok=False, validation_ok=None,
                                  error=ErrorCategory.MALFORMED_JSON))
        ledger.record(make_record(attempt=2, ticks=100))
        summary = ledger.game_summary()
        self.assertEqual(summary["calls"], 2)
        self.assertEqual(summary["retries"], 1)
        self.assertEqual(summary["parse_failures"], 1)

    def test_exact_tick_sums(self):
        ledger = UsageLedger()
        ticks = [37_756_000, 158_500, 12_500_000, 1]
        for t in ticks:
            ledger.record(make_record(ticks=t))
        summary = ledger.game_summary()
        self.assertEqual(summary["cost_ticks_total"], sum(ticks))
        self.assertTrue(summary["cost_complete"])

    def test_game_total_equals_sum_of_records(self):
        ledger = UsageLedger()
        for i in range(10):
            ledger.record(make_record(player_id=i % 3, ticks=1_000_000 * (i + 1)))
        summary = ledger.game_summary()
        records = ledger.records
        self.assertEqual(
            summary["cost_ticks_total"], sum(r.cost.ticks for r in records)
        )
        self.assertEqual(
            summary["tokens"]["total_tokens"],
            sum(r.usage.total_tokens for r in records),
        )
        by_player_total = sum(
            g["cost_ticks"] for g in summary["by_player"].values()
        )
        self.assertEqual(by_player_total, summary["cost_ticks_total"])

    def test_unavailable_cost_not_treated_as_zero(self):
        ledger = UsageLedger()
        ledger.record(make_record(ticks=5_000_000))
        ledger.record(make_record())  # cost unavailable
        summary = ledger.game_summary()
        self.assertFalse(summary["cost_complete"])
        self.assertEqual(summary["calls_with_unavailable_cost"], 1)
        # Known portion is reported; unavailable bucket has no fabricated 0.
        self.assertEqual(summary["cost_ticks_total"], 5_000_000)
        self.assertIsNone(summary["cost_by_source"]["unavailable"]["usd"])

    def test_all_cost_unknown_reports_none_not_zero(self):
        ledger = UsageLedger()
        ledger.record(make_record())
        ledger.record(make_record())
        summary = ledger.game_summary()
        self.assertIsNone(summary["cost_usd_total"])
        self.assertIsNone(summary["cost_ticks_total"])
        self.assertFalse(summary["cost_complete"])

    def test_empty_ledger_is_zero_cost(self):
        summary = UsageLedger().game_summary()
        self.assertEqual(summary["calls"], 0)
        self.assertEqual(summary["cost_usd_total"], 0.0)
        self.assertTrue(summary["cost_complete"])

    def test_estimated_and_exact_sources_kept_separate(self):
        ledger = UsageLedger()
        ledger.record(make_record(ticks=10_000_000_000))  # $1 exact
        ledger.record(make_record(est_usd=0.5))
        summary = ledger.game_summary()
        self.assertAlmostEqual(summary["cost_usd_total"], 1.5)
        self.assertEqual(
            summary["cost_by_source"]["provider_reported"]["calls"], 1
        )
        self.assertAlmostEqual(
            summary["cost_by_source"]["pricing_table_estimate"]["usd"], 0.5
        )
        # ticks total only reflects exact ticks
        self.assertEqual(summary["cost_ticks_total"], 10_000_000_000)

    def test_fallback_records_observable_and_not_counted_as_calls(self):
        ledger = UsageLedger()
        for attempt in (1, 2, 3):
            ledger.record(make_record(attempt=attempt, ticks=100, api_ok=True,
                                      parse_ok=True, validation_ok=False,
                                      error=ErrorCategory.INVALID_GAME_ACTION))
        ledger.record(make_record(api_attempted=False, api_ok=False,
                                  tokens=None,
                                  error=ErrorCategory.FALLBACK_USED))
        summary = ledger.game_summary()
        self.assertEqual(summary["calls"], 3)
        self.assertEqual(summary["fallbacks"], 1)
        self.assertEqual(summary["validation_failures"], 3)
        self.assertEqual(
            summary["errors_by_category"]["invalid_game_action"], 3
        )
        self.assertEqual(summary["errors_by_category"]["fallback_used"], 1)

    def test_breakdown_dimensions(self):
        ledger = UsageLedger()
        ledger.record(make_record(player_id=1, role="werewolf",
                                  phase="night_wolf_chat", action="wolf_chat",
                                  ticks=100))
        ledger.record(make_record(player_id=2, role="villager",
                                  phase="day_vote", action="vote", ticks=200))
        ledger.record(make_record(player_id=1, role="werewolf",
                                  phase="day_vote", action="vote", ticks=300))
        summary = ledger.game_summary()
        self.assertEqual(summary["by_player"]["1"]["calls"], 2)
        self.assertEqual(summary["by_player"]["1"]["cost_ticks"], 400)
        self.assertEqual(summary["by_role"]["werewolf"]["cost_ticks"], 400)
        self.assertEqual(summary["by_phase"]["day_vote"]["cost_ticks"], 500)
        self.assertEqual(
            summary["by_required_action"]["wolf_chat"]["cost_ticks"], 100
        )

    def test_missing_usage_counted_not_zeroed(self):
        ledger = UsageLedger()
        ledger.record(make_record(tokens=None, ticks=100))
        ledger.record(make_record(tokens=(50, 10), ticks=100))
        summary = ledger.game_summary()
        self.assertEqual(summary["calls_missing_usage"], 1)
        self.assertEqual(summary["tokens"]["total_tokens"], 60)

    def test_thread_safety_smoke(self):
        ledger = UsageLedger()

        def worker(n):
            for _ in range(100):
                ledger.record(make_record(player_id=n, ticks=1))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        summary = ledger.game_summary()
        self.assertEqual(summary["calls"], 1000)
        self.assertEqual(summary["cost_ticks_total"], 1000)


if __name__ == "__main__":
    unittest.main()
