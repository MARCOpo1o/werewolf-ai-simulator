import json
import os
import tempfile
import unittest
from collections import Counter
from statistics import mean

from werewolf.cli.run_trials import run_one_trial, write_manifest, write_summary_csv, write_summary_json
from werewolf.engine.game import GameEngine


class IntegrationTests(unittest.TestCase):
    def test_step_mode_skips_night_seer_when_disabled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = GameEngine(
                n_players=7,
                n_wolves=2,
                n_seers=0,
                seed=321,
                output_dir=tmpdir,
                api_key="",
                transcript_enabled=False,
                show_all_channels=False,
            )

            seen_phases = []
            for _ in range(30):
                result = engine.run_next_phase()
                seen_phases.append(result["phase"])
                if result["done"]:
                    break

            self.assertNotIn("night_seer", seen_phases)
            engine.logger.close()

    def test_single_game_without_seer_logs_outcome(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = GameEngine(
                n_players=7,
                n_wolves=2,
                n_seers=0,
                seed=123,
                output_dir=tmpdir,
                api_key="",
                transcript_enabled=False,
                show_all_channels=False,
            )
            winner = engine.run()

            self.assertIn(winner, {"wolf", "village"})
            self.assertTrue(os.path.exists(engine.logger.filepath))

            with open(engine.logger.filepath, "r", encoding="utf-8") as f:
                rows = [json.loads(line) for line in f if line.strip()]

            config = next(row for row in rows if row["type"] == "config")
            outcome = next(row for row in rows if row["type"] == "outcome")

            self.assertEqual(config["n_seers"], 0)
            self.assertIn("role_map", config)
            self.assertIn(outcome["winner"], {"wolf", "village"})

    def test_mini_batch_20_produces_unique_games_and_valid_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            records = []
            for i in range(20):
                record = run_one_trial(
                    trial_index=i,
                    seed=1000 + i,
                    n_players=7,
                    n_wolves=2,
                    n_seers=0,
                    output_dir=tmpdir,
                    api_key="",
                    model="grok-4-1-fast",
                    quiet=True,
                )
                records.append(record)

            self.assertEqual(len(records), 20)

            game_ids = {r["game_id"] for r in records}
            self.assertEqual(len(game_ids), 20)

            for r in records:
                self.assertTrue(os.path.exists(r["log_path"]))

            counts = Counter(r["winner"] for r in records)
            rounds = [r["rounds"] for r in records]
            summary = {
                "trials_requested": 20,
                "trials_completed": len(records),
                "outcome_counts": {
                    "wolf": counts.get("wolf", 0),
                    "village": counts.get("village", 0),
                },
                "wolf_win_rate": counts.get("wolf", 0) / len(records),
                "village_win_rate": counts.get("village", 0) / len(records),
                "avg_rounds": mean(rounds),
                "started_at": "test-start",
                "completed_at": "test-end",
            }

            manifest_path = os.path.join(tmpdir, "manifest.jsonl")
            summary_json_path = os.path.join(tmpdir, "summary.json")
            summary_csv_path = os.path.join(tmpdir, "summary.csv")

            write_manifest(manifest_path, records)
            write_summary_json(summary_json_path, summary)
            write_summary_csv(summary_csv_path, summary)

            self.assertTrue(os.path.exists(manifest_path))
            self.assertTrue(os.path.exists(summary_json_path))
            self.assertTrue(os.path.exists(summary_csv_path))
            self.assertEqual(summary["trials_completed"], 20)
            self.assertEqual(
                summary["outcome_counts"]["wolf"] + summary["outcome_counts"]["village"],
                20,
            )


if __name__ == "__main__":
    unittest.main()
