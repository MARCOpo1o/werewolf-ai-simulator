import csv
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

from tests.test_experiment_runner import (
    make_manifest, offline_engine_factory,
    quiet,
    ready_prober,
)
from werewolf.cli.experiment import main as cli_main
from werewolf.experiments.exports import (
    PROVENANCE_FIELDS,
    write_summary_exports,
)
from werewolf.experiments.index import (
    index_path,
    load_experiment_index,
    rebuild_experiment_index,
)
from werewolf.experiments.manifest import write_manifest
from werewolf.experiments.runner import run_experiment
from werewolf.experiments.summaries import summarize_experiment


def run_offline(tmp):
    write_manifest(tmp, make_manifest(game={"belief_snapshots": True}))
    run_experiment(
        tmp, "exp1", health_prober=ready_prober, progress=quiet,
        engine_factory=offline_engine_factory,
    )


def read_csv(path: Path) -> list:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


class ExportTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name
        run_offline(self.tmp)
        self.result = summarize_experiment(
            self.tmp, "exp1", exporter=write_summary_exports,
        )
        self.directory = Path(
            self.tmp, "exp1", "exports", "summary_0001",
        )

    def tearDown(self):
        self._tmp.cleanup()

    def test_all_export_files_exist(self):
        for name in ("trials.csv", "attempts.csv", "metrics.csv", "comparisons.csv",
                     "calibration.csv"):
            self.assertTrue((self.directory / name).is_file(), name)

    def test_noop_summary_rebuilds_deleted_exports(self):
        (self.directory / "metrics.csv").unlink()
        result = summarize_experiment(
            self.tmp, "exp1", exporter=write_summary_exports,
        )
        self.assertFalse(result["created"])
        self.assertTrue((self.directory / "metrics.csv").is_file())

    def test_every_row_carries_full_provenance(self):
        payload = json.loads(
            Path(self.tmp, "exp1", "summaries", "summary_0001.json")
            .read_text(encoding="utf-8")
        )
        for name in ("trials.csv", "attempts.csv", "metrics.csv",
                     "calibration.csv"):
            rows = read_csv(self.directory / name)
            self.assertTrue(rows, name)
            for row in rows:
                for column in PROVENANCE_FIELDS:
                    self.assertIn(column, row, f"{name}:{column}")
                self.assertEqual(row["experiment_id"], "exp1")
                self.assertEqual(
                    row["manifest_content_sha256"],
                    payload["manifest_content_sha256"],
                )
                self.assertEqual(row["summary_revision"], "1")
                self.assertEqual(
                    row["summary_input_sha256"],
                    payload["summary_input_sha256"],
                )
                self.assertEqual(
                    row["analysis_runtime_hash"],
                    payload["analysis_runtime_hash"],
                )

    def test_trials_export_lists_completed_games(self):
        rows = read_csv(self.directory / "trials.csv")
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(r["source_status"] == "verified"
                            for r in rows))
        self.assertTrue(all(r["winner"] in ("wolf", "village")
                            for r in rows))
        self.assertTrue(all(r["analysis_eligibility"]
                            for r in rows))
        self.assertTrue(all(r["usage_reliability"] for r in rows))

    def test_attempts_export_lists_every_operational_attempt(self):
        rows = read_csv(self.directory / "attempts.csv")
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(r["status"] == "trial_completed" for r in rows))
        self.assertTrue(all(r["source_status"] == "verified" for r in rows))
        self.assertTrue(all(r["cost_completeness"] for r in rows))

    def test_metrics_export_is_long_form(self):
        rows = read_csv(self.directory / "metrics.csv")
        metric_ids = {r["metric_id"] for r in rows}
        self.assertIn("village_win_rate", metric_ids)
        self.assertIn("retry_rate", metric_ids)
        self.assertIn("brier_post_discussion", metric_ids)
        self.assertIn(
            "clean_eligible_share_of_verified_completed", metric_ids,
        )
        views = {r["analysis_view"] for r in rows}
        self.assertEqual(
            views,
            {"all_completed", "clean_eligible",
             "completed_not_clean_eligible"},
        )

    def test_calibration_export_has_all_bins(self):
        rows = read_csv(self.directory / "calibration.csv")
        overall_post = [
            r for r in rows
            if r["analysis_view"] == "all_completed"
            and r["condition_id"] == "overall"
            and r["checkpoint"] == "post_discussion"
        ]
        self.assertEqual(len(overall_post), 10)
        self.assertEqual(overall_post[0]["bin"], "[0.0, 0.1)")
        self.assertEqual(overall_post[-1]["bin"], "[0.9, 1.0]")


class IndexTests(unittest.TestCase):
    def test_index_is_rebuilt_from_canonical_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_offline(tmp)
            summarize_experiment(tmp, "exp1",
                                 exporter=write_summary_exports)
            index = rebuild_experiment_index(tmp)
            entry = index["experiments"][0]
            self.assertEqual(entry["experiment_id"], "exp1")
            self.assertEqual(entry["status"], "ok")
            self.assertEqual(entry["scheduled_trials"], 2)
            self.assertEqual(entry["progress"]["completed"], 2)
            self.assertEqual(entry["current_summary_revision"], 1)

            index_path(tmp).unlink()
            recovered = load_experiment_index(tmp)
            self.assertEqual(
                recovered["experiments"][0]["progress"],
                entry["progress"],
            )

    def test_index_flags_journal_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_offline(tmp)
            journal = Path(tmp, "exp1", "trials.jsonl")
            with open(journal, "a", encoding="utf-8") as f:
                f.write("corrupt line\n")
            index = rebuild_experiment_index(tmp)
            entry = index["experiments"][0]
            self.assertEqual(entry["status"], "journal_integrity_error")
            self.assertTrue(entry["problems"])


class CliTests(unittest.TestCase):
    def _cli(self, *argv) -> tuple:
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = cli_main(list(argv))
        return code, out.getvalue(), err.getvalue()

    def test_create_validate_run_summarize_flow(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, out, _ = self._cli(
                "create-crossed", "--experiment-id", "cli_exp",
                "--model-a", "fast", "--model-b", "gemini_flash_lite",
                "--num-seeds", "2", "--repetitions", "1",
                "--root", tmp,
            )
            self.assertEqual(code, 0)
            self.assertIn("Created experiment cli_exp", out)

            manifest_file = Path(tmp, "cli_exp", "manifest.json")
            code, out, _ = self._cli("validate", str(manifest_file))
            self.assertEqual(code, 0, out)
            self.assertIn("Manifest is valid", out)

            index = json.loads(index_path(tmp).read_text(encoding="utf-8"))
            self.assertEqual(
                index["experiments"][0]["experiment_id"], "cli_exp",
            )
            self.assertEqual(
                index["experiments"][0]["scheduled_trials"], 8,
            )

    def test_validate_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._cli(
                "create-crossed", "--experiment-id", "cli_exp",
                "--model-a", "fast", "--model-b", "gemini_flash_lite",
                "--num-seeds", "1", "--repetitions", "1", "--root", tmp,
            )
            manifest_file = Path(tmp, "cli_exp", "manifest.json")
            manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
            manifest["description"] = "tampered"
            manifest_file.write_text(json.dumps(manifest),
                                     encoding="utf-8")
            code, out, _ = self._cli("validate", str(manifest_file))
            self.assertEqual(code, 1)
            self.assertIn("INVALID", out)

    def test_cli_run_and_summarize_offline(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_manifest(tmp, make_manifest())
            with mock.patch(
                "werewolf.experiments.runner.probe_model", ready_prober,
            ):
                code, out, err = self._cli(
                    "run", "exp1", "--root", tmp,
                )
            self.assertEqual(code, 0, err)
            self.assertIn("2 completed", out)

            code, out, err = self._cli("summarize", "exp1", "--root", tmp)
            self.assertEqual(code, 0, err)
            self.assertIn("Created summary revision 1", out)
            self.assertTrue(
                Path(tmp, "exp1", "exports", "summary_0001",
                     "metrics.csv").is_file(),
            )

            code, out, err = self._cli("summarize", "exp1", "--root", tmp)
            self.assertEqual(code, 0, err)
            self.assertIn("No-op", out)

    def test_cli_run_requires_resume_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_manifest(tmp, make_manifest())
            with mock.patch(
                "werewolf.experiments.runner.probe_model", ready_prober,
            ):
                self._cli("run", "exp1", "--root", tmp)
                code, _, err = self._cli("run", "exp1", "--root", tmp)
                self.assertEqual(code, 1)
                self.assertIn("--resume", err)


if __name__ == "__main__":
    unittest.main()
