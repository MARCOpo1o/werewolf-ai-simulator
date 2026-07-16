import tempfile
import unittest
from pathlib import Path

from tests.test_experiment_runner import make_manifest, quiet, ready_prober
from werewolf.engine.game import GameEngine
from werewolf.experiments.exports import write_summary_exports
from werewolf.experiments.manifest import write_manifest
from werewolf.experiments.runner import run_experiment
from werewolf.experiments.summaries import summarize_experiment
from werewolf.web import app as web_app


def offline_engine_factory(entry, manifest, games_directory):
    """Keep web API fixtures entirely offline even when .env has keys."""
    execution = manifest["execution_contract"]
    game = execution["game"]
    policy = execution["policies"]
    roles = execution["conditions"][entry["condition_id"]]["role_models"]
    return GameEngine(
        n_players=game["n_players"], n_wolves=game["n_wolves"],
        n_seers=game["n_seers"], seed=entry["seed"],
        output_dir=str(games_directory), api_key="",
        model=roles["villager"], role_models=roles,
        role_providers={role: None for role in roles},
        allow_provider_fallback=True,
        action_failure_policy=policy["action_failure_policy"],
        max_rounds=policy["max_rounds"],
        transcript_enabled=False, show_all_channels=False,
        belief_snapshots=game["belief_snapshots"],
        discussion_cycles=game["discussion_cycles"],
        batch_id=f"fixture/{entry['condition_id']}",
        trial_index=entry["trial_index"],
    )


class ExperimentApiTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        write_manifest(self.root, make_manifest(game={"belief_snapshots": True}))
        run_experiment(
            self.root, "exp1", health_prober=ready_prober, progress=quiet,
            engine_factory=offline_engine_factory,
        )
        summarize_experiment(
            self.root, "exp1", exporter=write_summary_exports,
        )
        self.old_root = web_app.experiment_root
        web_app.experiment_root = self.root
        web_app.app.config["TESTING"] = True
        self.client = web_app.app.test_client()

    def tearDown(self):
        web_app.experiment_root = self.old_root
        self.temp.cleanup()

    def test_lists_and_loads_derived_experiment_data(self):
        history = self.client.get("/api/experiments")
        self.assertEqual(history.status_code, 200)
        self.assertEqual(history.get_json()["experiments"][0]["experiment_id"], "exp1")

        detail = self.client.get("/api/experiments/exp1")
        self.assertEqual(detail.status_code, 200)
        body = detail.get_json()
        self.assertEqual(body["manifest"]["experiment_id"], "exp1")
        self.assertEqual(body["summary_catalog"]["current_revision"], 1)

        summary = self.client.get("/api/experiments/exp1/summaries/1")
        self.assertEqual(summary.status_code, 200)
        self.assertEqual(summary.get_json()["revision"], 1)

    def test_downloads_only_allowlisted_exports_without_caching(self):
        response = self.client.get("/api/experiments/exp1/exports/1/metrics.csv")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["Cache-Control"], "no-store")
        self.assertEqual(response.mimetype, "text/csv")
        self.assertIn("summary_input_sha256", response.get_data(as_text=True))
        response.close()

        self.assertEqual(
            self.client.get("/api/experiments/exp1/exports/1/../manifest.json").status_code,
            404,
        )

    def test_unknown_experiments_and_summaries_are_not_found(self):
        self.assertEqual(self.client.get("/api/experiments/nope").status_code, 404)
        self.assertEqual(
            self.client.get("/api/experiments/exp1/summaries/2").status_code,
            404,
        )

    def test_experiment_games_link_to_their_scoped_forensic_report(self):
        summary = self.client.get(
            "/api/experiments/exp1/summaries/1"
        ).get_json()
        game_id = summary["analysis"]["games"][0]["game_id"]
        report = self.client.get(
            f"/api/experiments/exp1/games/{game_id}/report"
        )
        self.assertEqual(report.status_code, 200)
        self.assertEqual(report.get_json()["overview"]["game_id"], game_id)

        page = self.client.get(f"/experiments/exp1/games/{game_id}")
        self.assertEqual(page.status_code, 200)
        self.assertIn(
            'data-report-api-base="/api/experiments/exp1/games"',
            page.get_data(as_text=True),
        )

    def test_pages_are_read_only_experiment_views(self):
        history = self.client.get("/experiments")
        self.assertEqual(history.status_code, 200)
        self.assertIn("Read-only benchmark history", history.get_data(as_text=True))
        self.assertIn("/static/experiments.js", history.get_data(as_text=True))

        report = self.client.get("/experiments/exp1")
        self.assertEqual(report.status_code, 200)
        self.assertIn("/static/experiment.js", report.get_data(as_text=True))
        self.assertNotIn("run", report.get_data(as_text=True).lower())
        for section in (
            "benchmark-overview", "condition-rows", "comparison-rows",
            "calibration-rows", "trial-rows", "export-links",
        ):
            self.assertIn(f'id="{section}"', report.get_data(as_text=True))

        dashboard = (
            Path(__file__).parents[1] / "werewolf" / "web" / "static"
            / "experiment.js"
        ).read_text(encoding="utf-8")
        self.assertIn("analysis-view", dashboard)
        self.assertIn("analysis_exclusion_reasons", dashboard)
        self.assertIn("scheduled_trial_outcomes", dashboard)
        self.assertIn("/experiments/${encodeURIComponent(experimentId)}/games/", dashboard)
        self.assertNotIn("method: 'POST'", dashboard)


if __name__ == "__main__":
    unittest.main()
