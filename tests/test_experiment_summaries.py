import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tests.test_experiment_runner import (
    boom_factory,
    make_manifest,
    quiet,
    ready_prober,
)
from werewolf.experiments.analysis_registry import (
    ANALYSIS_IMPLEMENTATIONS,
    AnalysisPolicyUnavailable,
    registry_key,
    resolve_analysis_implementation,
)
from werewolf.experiments.journal import read_journal
from werewolf.experiments.locks import analysis_lock
from werewolf.experiments.manifest import (
    experiment_dir,
    games_dir,
    journal_path,
    summary_catalog_path,
    write_manifest,
)
from werewolf.experiments.runner import run_experiment
from werewolf.experiments.snapshot import (
    SOURCE_MISSING,
    SOURCE_MODIFIED,
    SOURCE_VERIFIED,
    capture_game_sources,
    capture_lifecycle_snapshot,
)
from werewolf.experiments.summaries import (
    SummaryError,
    load_summary_catalog,
    rebuild_summary_catalog,
    summarize_experiment,
)


def fake_analyze(**kwargs):
    sources = kwargs["sources"]
    return {
        "fake": True,
        "verified_sources": sum(1 for s in sources if s.verified),
        "statuses": sorted(s.source_status for s in sources),
    }


def run_offline_experiment(tmp, **run_kwargs):
    write_manifest(tmp, make_manifest())
    run_kwargs.setdefault("health_prober", ready_prober)
    run_kwargs.setdefault("progress", quiet)
    return run_experiment(tmp, "exp1", **run_kwargs)


def summarize(tmp, **kwargs):
    kwargs.setdefault("analyze", fake_analyze)
    return summarize_experiment(tmp, "exp1", **kwargs)


class AnalysisRegistryTests(unittest.TestCase):
    def test_registry_key_derivation(self):
        contract = {
            "report_build_version": 12, "validity_policy_version": 4,
            "belief_metrics_version": 2, "aggregate_analysis_version": 1,
        }
        self.assertEqual(
            registry_key(contract),
            ("report-12", "validity-4", "belief-2", "aggregate-1"),
        )

    def test_unregistered_policy_fails_closed(self):
        contract = {
            "report_build_version": 11, "validity_policy_version": 4,
            "belief_metrics_version": 2, "aggregate_analysis_version": 1,
        }
        with self.assertRaises(AnalysisPolicyUnavailable) as ctx:
            resolve_analysis_implementation(contract)
        self.assertIn("analysis_policy_unavailable", str(ctx.exception))

    def test_registered_but_unloadable_policy_fails_closed(self):
        contract = {
            "report_build_version": 12, "validity_policy_version": 4,
            "belief_metrics_version": 2, "aggregate_analysis_version": 1,
        }
        with mock.patch.dict(ANALYSIS_IMPLEMENTATIONS, {
            registry_key(contract):
                "werewolf.experiments.aggregate:does_not_exist",
        }):
            with self.assertRaises(AnalysisPolicyUnavailable):
                resolve_analysis_implementation(contract)

    def test_registered_policy_resolves(self):
        contract = {
            "report_build_version": 12, "validity_policy_version": 4,
            "belief_metrics_version": 2, "aggregate_analysis_version": 1,
        }
        with mock.patch.dict(ANALYSIS_IMPLEMENTATIONS, {
            registry_key(contract):
                "werewolf.experiments.summaries:rebuild_summary_catalog",
        }):
            self.assertIs(
                resolve_analysis_implementation(contract),
                rebuild_summary_catalog,
            )


class SnapshotTests(unittest.TestCase):
    def test_lifecycle_snapshot_covers_only_complete_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_offline_experiment(tmp)
            journal = journal_path(tmp, "exp1")
            before = capture_lifecycle_snapshot(journal)
            with open(journal, "a", encoding="utf-8") as f:
                f.write('{"torn": "tail')
            after = capture_lifecycle_snapshot(journal)
            self.assertEqual(
                before.lifecycle_snapshot_sha256,
                after.lifecycle_snapshot_sha256,
            )
            self.assertEqual(
                before.journal_byte_length, after.journal_byte_length,
            )
            self.assertEqual(before.lifecycle_record_count,
                             after.lifecycle_record_count)
            self.assertIsNotNone(after.last_lifecycle_record_id)

    def test_sources_verified_modified_and_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_offline_experiment(tmp)
            lifecycle = capture_lifecycle_snapshot(journal_path(tmp, "exp1"))
            directory = games_dir(tmp, "exp1")
            sources = capture_game_sources(directory, lifecycle)
            self.assertEqual(len(sources), 2)
            self.assertTrue(all(s.source_status == SOURCE_VERIFIED
                                for s in sources))
            self.assertTrue(all(s.rows for s in sources))

            first, second = sources
            with open(directory / f"{first.game_id}.jsonl", "a",
                      encoding="utf-8") as f:
                f.write('{"type": "sneaky_edit"}\n')
            (directory / f"{second.game_id}.jsonl").unlink()

            sources = capture_game_sources(directory, lifecycle)
            by_id = {s.game_id: s for s in sources}
            self.assertEqual(by_id[first.game_id].source_status,
                             SOURCE_MODIFIED)
            self.assertIsNone(by_id[first.game_id].rows)
            self.assertEqual(by_id[second.game_id].source_status,
                             SOURCE_MISSING)
            self.assertIsNone(by_id[second.game_id].observed_game_sha256)

    def test_failed_attempt_sources_are_captured(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_manifest(tmp, make_manifest(seeds=(9001,)))
            run_experiment(tmp, "exp1", engine_factory=boom_factory,
                           health_prober=ready_prober, progress=quiet)
            lifecycle = capture_lifecycle_snapshot(journal_path(tmp, "exp1"))
            sources = capture_game_sources(games_dir(tmp, "exp1"), lifecycle)
            self.assertEqual(len(sources), 1)
            self.assertEqual(sources[0].record_type, "trial_failed")
            self.assertEqual(sources[0].source_status, SOURCE_MISSING)


class SummaryRevisionTests(unittest.TestCase):
    def test_summarize_creates_then_noops_on_identical_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_offline_experiment(tmp)
            first = summarize(tmp)
            self.assertTrue(first["created"])
            self.assertEqual(first["revision"], 1)
            path = Path(first["path"])
            content = path.read_bytes()

            second = summarize(tmp)
            self.assertFalse(second["created"])
            self.assertEqual(second["revision"], 1)
            self.assertEqual(second["summary_input_sha256"],
                             first["summary_input_sha256"])
            self.assertEqual(path.read_bytes(), content)

    def test_new_completed_trials_create_a_new_revision(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_manifest(tmp, make_manifest(seeds=(9001,)))
            run_experiment(tmp, "exp1", engine_factory=boom_factory,
                           health_prober=ready_prober, progress=quiet)
            first = summarize_experiment(tmp, "exp1", analyze=fake_analyze)
            self.assertEqual(first["revision"], 1)
            run_experiment(tmp, "exp1", retry_failed=True,
                           health_prober=ready_prober, progress=quiet)
            second = summarize_experiment(tmp, "exp1", analyze=fake_analyze)
            self.assertTrue(second["created"])
            self.assertEqual(second["revision"], 2)
            self.assertNotEqual(second["summary_input_sha256"],
                                first["summary_input_sha256"])

    def test_source_drift_creates_flagged_revision(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_offline_experiment(tmp)
            first = summarize(tmp)
            snapshot = read_journal(journal_path(tmp, "exp1"))
            completed = [r for r in snapshot.records
                         if r["record_type"] == "trial_completed"][0]
            log = games_dir(tmp, "exp1") / f"{completed['game_id']}.jsonl"
            with open(log, "a", encoding="utf-8") as f:
                f.write('{"type": "sneaky_edit"}\n')
            second = summarize(tmp)
            self.assertTrue(second["created"])
            self.assertEqual(second["revision"], 2)
            payload = json.loads(
                Path(tmp, "exp1", "summaries", "summary_0002.json")
                .read_text(encoding="utf-8")
            )
            statuses = [s["source_status"] for s in payload["sources"]]
            self.assertIn(SOURCE_MODIFIED, statuses)
            self.assertIn(SOURCE_VERIFIED, statuses)

    def test_analysis_runtime_change_creates_new_revision(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_offline_experiment(tmp)
            first = summarize(tmp)
            with mock.patch(
                "werewolf.experiments.summaries.analysis_runtime_hash",
                return_value="9" * 64,
            ):
                second = summarize(tmp, analysis_policy="current")
            self.assertTrue(second["created"])
            self.assertNotEqual(second["summary_input_sha256"],
                                first["summary_input_sha256"])

    def test_pinned_policy_fails_closed_when_runtime_drifted(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_offline_experiment(tmp)
            with mock.patch(
                "werewolf.experiments.summaries.analysis_runtime_hash",
                return_value="9" * 64,
            ):
                with self.assertRaises(AnalysisPolicyUnavailable) as ctx:
                    summarize(tmp)
            self.assertIn("analysis_policy_unavailable", str(ctx.exception))

    def test_current_policy_records_current_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_offline_experiment(tmp)
            with mock.patch(
                "werewolf.experiments.summaries.analysis_runtime_hash",
                return_value="9" * 64,
            ):
                result = summarize(tmp, analysis_policy="current")
            payload = json.loads(Path(result["path"]).read_text())
            self.assertEqual(payload["analysis_policy"], "current")
            self.assertEqual(
                payload["analysis_contract"]["aggregate_analysis_version"],
                1,
            )

    def test_concurrent_summarizer_is_rejected(self):
        from werewolf.experiments.locks import LockHeldError
        with tempfile.TemporaryDirectory() as tmp:
            run_offline_experiment(tmp)
            holder = analysis_lock(
                experiment_dir(tmp, "exp1"), "exp1",
            ).acquire()
            try:
                with self.assertRaises(LockHeldError):
                    summarize(tmp)
            finally:
                holder.release()

    def test_catalog_is_derived_and_rebuildable(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_offline_experiment(tmp)
            summarize(tmp)
            catalog_path = summary_catalog_path(tmp, "exp1")
            original = json.loads(catalog_path.read_text(encoding="utf-8"))
            catalog_path.unlink()
            rebuilt = load_summary_catalog(tmp, "exp1")
            self.assertEqual(rebuilt["current_revision"],
                             original["current_revision"])
            self.assertEqual(
                [r["summary_input_sha256"] for r in rebuilt["revisions"]],
                [r["summary_input_sha256"] for r in original["revisions"]],
            )

    def test_corrupt_revision_is_a_hard_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_offline_experiment(tmp)
            summarize(tmp)
            bad = Path(tmp, "exp1", "summaries", "summary_0002.json")
            bad.write_text("{not json", encoding="utf-8")
            with self.assertRaises(SummaryError):
                summarize(tmp)


if __name__ == "__main__":
    unittest.main()
