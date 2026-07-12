import csv
import json
import os
import tempfile
import unittest

from werewolf.cli.run_trials import (
    build_batch_summary,
    run_one_trial,
    write_summary_csv,
)
from werewolf.llm.fake_provider import FakeProvider, success_result
from werewolf.llm.ledger import UsageLedger, aggregate_game_summaries
from tests.test_usage_ledger import make_record


def make_game_summary(*, ticks=None, calls=1, complete=True, **kw):
    ledger = UsageLedger()
    for i in range(calls):
        ledger.record(make_record(ticks=ticks if complete or i == 0 else None))
    return ledger.game_summary()


class AggregateSummariesTests(unittest.TestCase):
    def test_batch_totals_equal_sum_of_game_totals(self):
        ledgers = []
        for game in range(3):
            ledger = UsageLedger()
            for call in range(4):
                ledger.record(make_record(ticks=1_000_000 * (game + 1)))
            ledgers.append(ledger)
        summaries = [l.game_summary() for l in ledgers]
        batch = aggregate_game_summaries(summaries)

        self.assertEqual(batch["games"], 3)
        self.assertEqual(batch["calls"], 12)
        self.assertEqual(
            batch["cost_ticks_total"],
            sum(s["cost_ticks_total"] for s in summaries),
        )
        self.assertAlmostEqual(
            batch["cost_usd_total"],
            sum(s["cost_usd_total"] for s in summaries),
        )
        self.assertEqual(
            batch["tokens"]["total_tokens"],
            sum(s["tokens"]["total_tokens"] for s in summaries),
        )
        self.assertTrue(batch["cost_complete"])

    def test_per_game_stats(self):
        summaries = []
        # games costing 1,2,...,10 dollars (in ticks)
        for usd in range(1, 11):
            ledger = UsageLedger()
            ledger.record(make_record(ticks=usd * 10_000_000_000))
            summaries.append(ledger.game_summary())
        stats = aggregate_game_summaries(summaries)["cost_per_game"]
        self.assertEqual(stats["games_counted"], 10)
        self.assertAlmostEqual(stats["mean"], 5.5)
        self.assertAlmostEqual(stats["median"], 5.0)
        self.assertAlmostEqual(stats["p90"], 9.0)
        self.assertAlmostEqual(stats["min"], 1.0)
        self.assertAlmostEqual(stats["max"], 10.0)

    def test_incomplete_games_counted_not_zeroed(self):
        complete = UsageLedger()
        complete.record(make_record(ticks=5_000_000))
        incomplete = UsageLedger()
        incomplete.record(make_record())  # unavailable cost
        batch = aggregate_game_summaries(
            [complete.game_summary(), incomplete.game_summary()]
        )
        self.assertFalse(batch["cost_complete"])
        self.assertEqual(batch["games_with_incomplete_cost"], 1)
        # known portion reported, not zero-filled
        self.assertEqual(batch["cost_ticks_total"], 5_000_000)
        # only the costed game enters per-game stats
        self.assertEqual(batch["cost_per_game"]["games_counted"], 1)

    def test_all_unknown_cost_is_none(self):
        ledger = UsageLedger()
        ledger.record(make_record())
        batch = aggregate_game_summaries([ledger.game_summary()])
        self.assertIsNone(batch["cost_usd_total"])
        self.assertIsNone(batch["cost_per_game"])

    def test_empty_batch(self):
        batch = aggregate_game_summaries([])
        self.assertEqual(batch["games"], 0)
        self.assertEqual(batch["cost_usd_total"], 0.0)
        self.assertTrue(batch["cost_complete"])


class TrialUsagePropagationTests(unittest.TestCase):
    def _trial(self, tmpdir, i):
        default = success_result(
            {"thought": "t", "say": None, "action": None}, cost_ticks=2_000,
        )
        return run_one_trial(
            trial_index=i,
            seed=500 + i,
            n_players=4,
            n_wolves=1,
            n_seers=0,
            output_dir=tmpdir,
            api_key="",
            model="fake-model",
            quiet=True,
            provider=FakeProvider(default=default),
            model_alias="fake",
            batch_id="batch_test_1",
        )

    def test_manifest_record_carries_usage_and_batch_identity(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            record = self._trial(tmpdir, 0)
            self.assertEqual(record["batch_id"], "batch_test_1")
            self.assertEqual(record["trial_index"], 0)
            self.assertGreater(record["usage"]["calls"], 0)
            self.assertTrue(record["usage"]["cost_complete"])

            # the game log agrees with the manifest record
            with open(record["log_path"], encoding="utf-8") as f:
                rows = [json.loads(line) for line in f if line.strip()]
            file_summary = next(
                r for r in rows if r["type"] == "usage_summary"
            )["usage"]
            self.assertEqual(file_summary["calls"], record["usage"]["calls"])
            self.assertEqual(
                file_summary["cost_ticks_total"],
                record["usage"]["cost_ticks_total"],
            )
            config = next(r for r in rows if r["type"] == "config")
            self.assertEqual(config["batch_id"], "batch_test_1")
            self.assertEqual(config["trial_index"], 0)

    def test_batch_summary_equals_sum_of_trials(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            records = [self._trial(tmpdir, i) for i in range(3)]
            summary = build_batch_summary(
                records,
                run_id="batch_test_1",
                started_at="t0",
                completed_at="t1",
                trials_requested=3,
                failed_trials=0,
                config={"model": "fake-model"},
                manifest_path="m.jsonl",
            )
            self.assertEqual(summary["usage"]["games"], 3)
            self.assertEqual(
                summary["usage"]["calls"],
                sum(r["usage"]["calls"] for r in records),
            )
            self.assertEqual(
                summary["usage"]["cost_ticks_total"],
                sum(r["usage"]["cost_ticks_total"] for r in records),
            )
            self.assertIn("model_registry_snapshot", summary)

    def test_summary_csv_handles_unknown_cost_as_empty_not_zero(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            summary = build_batch_summary(
                [],
                run_id="r", started_at="t0", completed_at="t1",
                trials_requested=0, failed_trials=0,
                config={}, manifest_path="m",
            )
            # force the unknown-cost case
            summary["usage"]["cost_usd_total"] = None
            path = os.path.join(tmpdir, "s.csv")
            write_summary_csv(path, summary)
            with open(path, newline="", encoding="utf-8") as f:
                row = next(csv.DictReader(f))
            self.assertEqual(row["total_cost_usd"], "")  # empty, not "0"


if __name__ == "__main__":
    unittest.main()
