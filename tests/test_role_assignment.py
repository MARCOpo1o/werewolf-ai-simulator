import json
import tempfile
import unittest

from werewolf.cli.run_experiment import build_conditions, run_crossed_experiment
from werewolf.engine.game import GameEngine
from werewolf.llm.fake_provider import FakeProvider, success_result


def make_provider():
    return FakeProvider(default=success_result(
        {"thought": "t", "say": None, "action": None}, cost_ticks=100,
    ))


class RoleModelAssignmentTests(unittest.TestCase):
    def _engine(self, tmpdir, wolf_provider, village_provider):
        return GameEngine(
            n_players=5, n_wolves=1, n_seers=0, seed=13,
            output_dir=tmpdir, api_key="",
            transcript_enabled=False, show_all_channels=False,
            belief_snapshots=False,
            role_models={"werewolf": "model_a", "villager": "model_b"},
            role_providers={
                "werewolf": wolf_provider,
                "villager": village_provider,
                "seer": village_provider,
            },
        )

    def test_roles_route_to_their_own_provider_and_model(self):
        wolf_provider, village_provider = make_provider(), make_provider()
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = self._engine(tmpdir, wolf_provider, village_provider)
            wolves = {pid for pid, p in engine.players.items()
                      if p.role == "werewolf"}
            engine.run()

            # every wolf agent used the wolf provider/model, and only they did
            self.assertGreater(wolf_provider.calls_made, 0)
            self.assertGreater(village_provider.calls_made, 0)
            for record in engine.ledger.records:
                if not record.api_attempted:
                    continue
                if record.context.player_id in wolves:
                    self.assertEqual(record.requested_model, "model_a")
                else:
                    self.assertEqual(record.requested_model, "model_b")
            wolf_models = {r.model for r in wolf_provider.requests}
            self.assertEqual(wolf_models, {"model_a"})
            village_models = {r.model for r in village_provider.requests}
            self.assertEqual(village_models, {"model_b"})

    def test_config_logs_resolved_role_models(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = self._engine(tmpdir, make_provider(), make_provider())
            engine.run()
            with open(engine.logger.filepath, encoding="utf-8") as f:
                config = next(
                    json.loads(line) for line in f
                    if json.loads(line)["type"] == "config"
                )
            self.assertEqual(config["role_models"]["werewolf"]["model"], "model_a")
            self.assertEqual(config["role_models"]["villager"]["model"], "model_b")
            self.assertEqual(config["role_models"]["seer"]["model"], "model_b")

    def test_villager_entry_required_and_roles_validated(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError):
                GameEngine(
                    n_players=4, n_wolves=1, n_seers=0, seed=1,
                    output_dir=tmpdir, api_key="",
                    role_models={"werewolf": "model_a"},
                    transcript_enabled=False,
                )
            with self.assertRaises(ValueError):
                GameEngine(
                    n_players=4, n_wolves=1, n_seers=0, seed=1,
                    output_dir=tmpdir, api_key="",
                    role_models={"villager": "m", "narrator": "m"},
                    transcript_enabled=False,
                )

    def test_explicit_none_role_provider_requires_fallback_opt_in(self):
        role_models = {
            "werewolf": "model_a", "villager": "model_b", "seer": "model_b",
        }
        role_providers = {role: None for role in role_models}
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(RuntimeError, "Provider for role werewolf"):
                GameEngine(
                    n_players=4, n_wolves=1, n_seers=0, seed=1,
                    output_dir=tmpdir, role_models=role_models,
                    role_providers=role_providers, transcript_enabled=False,
                )
            engine = GameEngine(
                n_players=4, n_wolves=1, n_seers=0, seed=1,
                output_dir=tmpdir, role_models=role_models,
                role_providers=role_providers, transcript_enabled=False,
                allow_provider_fallback=True,
            )
            self.assertTrue(all(agent.provider is None for agent in engine.agents.values()))
            engine.close()

    def test_inactive_seer_provider_is_never_required(self):
        provider = make_provider()
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = GameEngine(
                n_players=4, n_wolves=1, n_seers=0, seed=1,
                output_dir=tmpdir,
                role_models={
                    "werewolf": "model_a", "villager": "model_b",
                    "seer": "model_c",
                },
                role_providers={
                    "werewolf": provider, "villager": provider,
                    "seer": None,
                },
                transcript_enabled=False,
            )
            self.assertFalse(engine.role_models_resolved["seer"]["active"])
            self.assertNotIn(
                "seer", {agent.role for agent in engine.agents.values()},
            )
            engine.close()


class CrossedExperimentTests(unittest.TestCase):
    def test_condition_matrix(self):
        conditions = build_conditions("A", "B")
        self.assertEqual(set(conditions), {
            "a_homogeneous", "b_homogeneous",
            "a_wolves_b_village", "b_wolves_a_village",
        })
        self.assertEqual(conditions["a_wolves_b_village"],
                         {"werewolf": "A", "villager": "B", "seer": "B"})
        self.assertEqual(conditions["b_wolves_a_village"],
                         {"werewolf": "B", "villager": "A", "seer": "A"})
        self.assertEqual(conditions["a_homogeneous"]["villager"], "A")

    def test_mini_experiment_shares_seeds_and_roles_across_conditions(self):
        # Explicitly offline providers keep this legacy development runner
        # network-free even when local credentials are configured.
        with tempfile.TemporaryDirectory() as tmpdir:
            summary = run_crossed_experiment(
                experiment_id="test_exp",
                model_a="fast",
                model_b="gemini_flash_lite",
                seeds=[900, 901],
                repetitions=1,
                n_players=5, n_wolves=1, n_seers=0,
                output_dir=tmpdir,
                quiet=True,
                belief_snapshots=False,
                role_providers={
                    "werewolf": None,
                    "villager": None,
                    "seer": None,
                },
                allow_provider_fallback=True,
                progress=lambda *_: None,
            )

            manifest_path = summary["manifest_path"]
            with open(manifest_path, encoding="utf-8") as f:
                records = [json.loads(line) for line in f if line.strip()]
            self.assertEqual(len(records), 8)  # 4 conditions x 2 seeds x 1 rep

            # same seed -> identical role assignment in every condition
            role_maps = {}
            for record in records:
                with open(record["log_path"], encoding="utf-8") as f:
                    config = next(
                        json.loads(line) for line in f
                        if json.loads(line)["type"] == "config"
                    )
                role_maps.setdefault(record["seed"], []).append(
                    config["role_map"]
                )
            for seed, maps in role_maps.items():
                self.assertEqual(len(maps), 4)
                self.assertTrue(
                    all(m == maps[0] for m in maps),
                    f"role assignment diverged across conditions at seed {seed}",
                )

            # spec block completeness (audit item 7)
            self.assertEqual(summary["repetitions_per_seed"], 1)
            self.assertEqual(summary["seeds"], [900, 901])
            self.assertIn("belief_schema", summary["instrumentation"])
            self.assertIn("prompt_version", summary["instrumentation"])
            self.assertIn("generation_config", summary["models"])
            self.assertEqual(len(summary["conditions"]), 4)
            for cond in summary["conditions"].values():
                self.assertEqual(cond["trials_completed"], 2)


if __name__ == "__main__":
    unittest.main()
