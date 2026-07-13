import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from werewolf.engine.game import GameEngine
from werewolf.engine.logging import JSONLLogger
from werewolf.llm.fake_provider import FakeProvider, success_result
from werewolf.llm.provider import GenerationConfig, ProviderResult
from werewolf.llm.records import ErrorCategory
from werewolf.llm.registry import (
    MODEL_REGISTRY,
    ProviderBuildResult,
    ProviderBuildStatus,
    effective_generation_config,
)
from werewolf.web.services import (
    RequestValidationError,
    create_engine_from_payload,
    health_check,
    parse_game_request,
    parse_generation_settings,
)


def ready_build(provider):
    return ProviderBuildResult(
        provider=provider,
        status=ProviderBuildStatus.READY,
        required_credentials=("TEST_KEY",),
    )


class GenerationResolutionTests(unittest.TestCase):
    def test_override_then_registry_then_provider_default(self):
        base = GenerationConfig(temperature=0.0)
        reasoning = MODEL_REGISTRY["reasoning"]
        fast = MODEL_REGISTRY["fast"]
        self.assertEqual(
            effective_generation_config(base, reasoning).reasoning_effort,
            "low",
        )
        self.assertEqual(
            effective_generation_config(base, reasoning, "high").reasoning_effort,
            "high",
        )
        self.assertIsNone(
            effective_generation_config(base, fast).reasoning_effort,
        )

    def test_web_rejects_duplicate_reasoning_input(self):
        with self.assertRaises(RequestValidationError) as ctx:
            parse_generation_settings({
                "generation_config": {"reasoning_effort": "low"},
                "reasoning_override": "high",
            })
        self.assertIn("generation_config.reasoning_effort", ctx.exception.errors)

    def test_web_rejects_non_string_reasoning_and_non_finite_numbers(self):
        for override in ([], {"effort": "low"}, 1):
            with self.assertRaises(RequestValidationError) as ctx:
                parse_generation_settings({"reasoning_override": override})
            self.assertEqual(
                ctx.exception.errors["reasoning_override"]["code"], "invalid_value",
            )
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(RequestValidationError) as ctx:
                parse_generation_settings({
                    "generation_config": {"temperature": value},
                })
            self.assertEqual(
                ctx.exception.errors["generation_config.temperature"]["code"],
                "invalid_value",
            )

    def test_engine_normalizes_legacy_reasoning_inputs(self):
        provider = FakeProvider()
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = GameEngine(
                n_players=4, n_wolves=1, n_seers=0, seed=1,
                output_dir=tmpdir, provider=provider,
                transcript_enabled=False, belief_snapshots=False,
                generation_config=GenerationConfig(reasoning_effort="high"),
            )
            assignment = engine.get_state_dict()["model_assignment"]["villager"]
            self.assertEqual(assignment["requested_reasoning_override"], "high")
            self.assertEqual(
                assignment["effective_generation"]["reasoning_effort"], "high",
            )
            engine.close()

    def test_engine_rejects_conflicting_reasoning_inputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(ValueError, "Conflicting reasoning"):
                GameEngine(
                    n_players=4, n_wolves=1, n_seers=0, seed=1,
                    output_dir=tmpdir, provider=FakeProvider(),
                    reasoning_effort="low", reasoning_override="high",
                )
            with self.assertRaisesRegex(ValueError, "Multiple legacy reasoning"):
                GameEngine(
                    n_players=4, n_wolves=1, n_seers=0, seed=1,
                    output_dir=tmpdir, provider=FakeProvider(),
                    reasoning_effort="low",
                    generation_config=GenerationConfig(reasoning_effort="low"),
                )


class GameRequestTests(unittest.TestCase):
    def test_quick_request_and_defaults(self):
        parsed = parse_game_request({"model": "fast"})
        self.assertEqual(parsed.model, "fast")
        self.assertIsNone(parsed.role_models)
        self.assertEqual(parsed.discussion_cycles, 2)
        self.assertTrue(parsed.belief_snapshots)

    def test_matchup_requires_every_role(self):
        with self.assertRaises(RequestValidationError) as ctx:
            parse_game_request({
                "role_models": {"werewolf": "fast", "villager": "reasoning"},
            })
        self.assertEqual(ctx.exception.errors["role_models"]["code"], "incomplete_roles")

    def test_inactive_seer_inherits_villager(self):
        parsed = parse_game_request({
            "n_seers": 0,
            "role_models": {
                "werewolf": "reasoning",
                "villager": "gemini_flash_lite",
                "seer": "claude_sonnet",
            },
        })
        self.assertEqual(parsed.role_models["seer"], "gemini_flash_lite")

    def test_only_selectable_aliases_are_accepted(self):
        with self.assertRaises(RequestValidationError):
            parse_game_request({"model": "grok-4.3"})

    def test_provider_errors_are_attributed_to_every_affected_role(self):
        missing = ProviderBuildResult(
            status=ProviderBuildStatus.MISSING_CREDENTIAL,
            required_credentials=("GROK_API_KEY",),
        )
        with mock.patch("werewolf.web.services.build_provider", return_value=missing):
            with self.assertRaises(RequestValidationError) as ctx:
                create_engine_from_payload({
                    "role_models": {
                        "werewolf": "fast", "villager": "fast", "seer": "fast",
                    },
                })
        self.assertEqual(set(ctx.exception.errors), {
            "role_models.werewolf", "role_models.villager", "role_models.seer",
        })
        self.assertTrue(all(
            error["code"] == "missing_key" for error in ctx.exception.errors.values()
        ))

    def test_valid_matchup_is_constructed_with_prebuilt_providers(self):
        provider = FakeProvider()
        engine = object()
        with mock.patch("werewolf.web.services.build_provider", return_value=ready_build(provider)), \
             mock.patch("werewolf.web.services.GameEngine", return_value=engine) as engine_class:
            result = create_engine_from_payload({
                "n_seers": 1,
                "role_models": {
                    "werewolf": "reasoning", "villager": "fast", "seer": "gpt_nano",
                },
                "reasoning_override": "high",
            })
        self.assertIs(result, engine)
        kwargs = engine_class.call_args.kwargs
        self.assertEqual(kwargs["reasoning_override"], "high")
        self.assertEqual(set(kwargs["role_providers"]), {"werewolf", "villager", "seer"})


class HealthCheckTests(unittest.TestCase):
    def test_ready_check_is_exactly_one_direct_provider_call(self):
        provider = FakeProvider(results=[success_result(
            {"health": "ok"}, resolved_model="grok-4.3",
        )])
        with mock.patch("werewolf.web.services.build_provider", return_value=ready_build(provider)):
            body, status = health_check("fast", {})
        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "ready")
        self.assertEqual(provider.calls_made, 1)
        self.assertTrue(body["checks"]["json_valid"])
        self.assertTrue(body["checks"]["model_match"])

    def test_adjusted_check_reports_detected_drop(self):
        result = success_result({"health": "ok"}, resolved_model="grok-4.3")
        result.provider_metadata["generation_dropped"] = ["top_p"]
        provider = FakeProvider(results=[result])
        with mock.patch("werewolf.web.services.build_provider", return_value=ready_build(provider)):
            body, _ = health_check("fast", {
                "generation_config": {"top_p": 0.9},
            })
        self.assertEqual(body["status"], "adjusted")
        self.assertEqual(body["generation_dropped"], ["top_p"])

    def test_malformed_json_fails_with_matching_model(self):
        provider = FakeProvider(results=[success_result(
            text="not-json", resolved_model="gpt-5.4-nano-2026-03-17",
        )])
        with mock.patch("werewolf.web.services.build_provider", return_value=ready_build(provider)):
            body, _ = health_check("gpt_nano", {})
        self.assertEqual(body["status"], "failed")
        self.assertFalse(body["checks"]["json_valid"])
        self.assertTrue(body["checks"]["model_match"])

    def test_wrong_json_shapes_and_objects_fail_contract(self):
        for text in ('[]', '"hello"', '{"error": "model unavailable"}'):
            provider = FakeProvider(results=[success_result(
                text=text, resolved_model="grok-4.3",
            )])
            with mock.patch("werewolf.web.services.build_provider", return_value=ready_build(provider)):
                body, _ = health_check("fast", {})
            self.assertEqual(body["status"], "failed", text)
            self.assertFalse(body["checks"]["json_valid"], text)

    def test_unreported_model_is_adjusted_with_warning(self):
        provider = FakeProvider(results=[success_result(
            {"health": "ok"}, resolved_model=None,
        )])
        with mock.patch("werewolf.web.services.build_provider", return_value=ready_build(provider)):
            body, _ = health_check("fast", {})
        self.assertEqual(body["status"], "adjusted")
        self.assertEqual(body["checks"]["model_identity"], "unreported")
        self.assertIsNone(body["checks"]["model_match"])
        self.assertTrue(body["warnings"])

    def test_mismatched_model_fails_independently(self):
        provider = FakeProvider(results=[success_result(
            {"health": "ok"}, resolved_model="gpt-5.4-preview",
        )])
        with mock.patch("werewolf.web.services.build_provider", return_value=ready_build(provider)):
            body, _ = health_check("gpt_nano", {})
        self.assertEqual(body["status"], "failed")
        self.assertTrue(body["checks"]["json_valid"])
        self.assertEqual(body["checks"]["model_identity"], "mismatched")

    def test_missing_key_is_distinct_from_provider_unavailable(self):
        missing = ProviderBuildResult(
            status=ProviderBuildStatus.MISSING_CREDENTIAL,
            required_credentials=("GROK_API_KEY",),
        )
        unavailable = ProviderBuildResult(
            status=ProviderBuildStatus.DEPENDENCY_UNAVAILABLE,
            error="SDK is not installed",
        )
        with mock.patch("werewolf.web.services.build_provider", return_value=missing):
            missing_body, _ = health_check("fast", {})
        with mock.patch("werewolf.web.services.build_provider", return_value=unavailable):
            unavailable_body, _ = health_check("fast", {})
        self.assertEqual(missing_body["status"], "missing_key")
        self.assertEqual(unavailable_body["status"], "provider_unavailable")


class EngineLifecycleTests(unittest.TestCase):
    def test_close_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = GameEngine(
                n_players=4, n_wolves=1, n_seers=0, seed=4,
                output_dir=tmpdir, provider=FakeProvider(),
                transcript_enabled=False, belief_snapshots=False,
            )
            engine.close()
            engine.close()
            self.assertTrue(engine.logger.file.closed)

    def test_homogeneous_fallback_requires_explicit_opt_in(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(RuntimeError, "allow_provider_fallback"):
                GameEngine(
                    n_players=4, n_wolves=1, n_seers=0, seed=4,
                    output_dir=tmpdir, provider=None, api_key="",
                    transcript_enabled=False, belief_snapshots=False,
                )
            engine = GameEngine(
                n_players=4, n_wolves=1, n_seers=0, seed=4,
                output_dir=tmpdir, provider=None, api_key="",
                transcript_enabled=False, belief_snapshots=False,
                allow_provider_fallback=True,
            )
            self.assertTrue(all(agent.provider is None for agent in engine.agents.values()))
            engine.close()

    def test_model_alias_controls_homogeneous_request_model(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = GameEngine(
                n_players=4, n_wolves=1, n_seers=0, seed=4,
                output_dir=tmpdir, model="grok-4.3",
                model_alias="gemini_flash", provider=FakeProvider(),
                transcript_enabled=False, belief_snapshots=False,
            )
            expected = MODEL_REGISTRY["gemini_flash"].model
            self.assertEqual(engine.model, expected)
            self.assertTrue(all(agent.model == expected for agent in engine.agents.values()))
            self.assertEqual(
                engine.get_state_dict()["model_assignment"]["villager"]["requested_model"],
                expected,
            )
            engine.close()

    def test_late_initialization_failure_closes_logger(self):
        closed = []
        original_close = JSONLLogger.close

        def tracked_close(logger):
            closed.append(logger)
            original_close(logger)

        with tempfile.TemporaryDirectory() as tmpdir, \
             mock.patch("werewolf.engine.game.JSONLLogger.close", autospec=True, side_effect=tracked_close):
            with mock.patch(
                "werewolf.engine.game.JSONLLogger.log_config",
                side_effect=RuntimeError("log failure"),
            ):
                with self.assertRaisesRegex(RuntimeError, "log failure"):
                    GameEngine(
                        n_players=4, n_wolves=1, n_seers=0, seed=4,
                        output_dir=tmpdir, provider=FakeProvider(),
                        transcript_enabled=False, belief_snapshots=False,
                    )
            self.assertEqual(len(closed), 1)
            self.assertTrue(closed[0].file.closed)

    def test_web_phase_completion_closes_logger_after_outcome(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = GameEngine(
                n_players=4, n_wolves=1, n_seers=0, seed=4,
                output_dir=tmpdir, provider=FakeProvider(),
                transcript_enabled=False, belief_snapshots=False,
            )
            villagers = [p for p in engine.players.values() if p.team == "village"]
            for player in villagers[:-1]:
                player.alive = False
            engine.state.round = 1
            engine._phase_index = engine.PHASE_ORDER.index("day_announce")
            result = engine.run_next_phase()
            self.assertTrue(result["done"])
            self.assertEqual(result["winner"], "wolf")
            self.assertTrue(engine.logger.file.closed)
            self.assertGreater(engine.ledger.game_summary()["calls"], -1)

    def test_role_assignment_logs_per_role_effective_generation(self):
        provider = FakeProvider(default=success_result({}))
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = GameEngine(
                n_players=4, n_wolves=1, n_seers=0, seed=4,
                output_dir=tmpdir, transcript_enabled=False,
                belief_snapshots=False,
                role_models={
                    "werewolf": "reasoning", "villager": "fast", "seer": "fast",
                },
                role_providers={role: provider for role in ("werewolf", "villager", "seer")},
            )
            state = engine.get_state_dict()
            self.assertEqual(
                state["model_assignment"]["werewolf"]["effective_generation"]["reasoning_effort"],
                "low",
            )
            self.assertIsNone(
                state["model_assignment"]["villager"]["effective_generation"]["reasoning_effort"],
            )
            self.assertFalse(state["model_assignment"]["seer"]["active"])
            engine.close()


class WebApiTests(unittest.TestCase):
    def setUp(self):
        with mock.patch("dotenv.load_dotenv"):
            self.webapp = importlib.import_module("werewolf.web.app")
        self.webapp.app.config.update(TESTING=True)
        self.client = self.webapp.app.test_client()
        self.webapp.game_engine = None

    def tearDown(self):
        if self.webapp.game_engine is not None:
            self.webapp.game_engine.close()
        self.webapp.game_engine = None

    def test_catalog_is_ordered_and_secret_free(self):
        response = self.client.get("/api/models")
        self.assertEqual(response.status_code, 200)
        models = response.get_json()["models"]
        self.assertEqual(models[0]["alias"], "fast")
        self.assertNotIn("api_key_env", models[0])
        self.assertNotIn("key", " ".join(models[0].keys()).replace("key_configured", ""))

    def test_setup_page_contains_quick_matchup_and_custom_controls(self):
        html = self.client.get("/").get_data(as_text=True)
        self.assertIn("Model Matchup", html)
        self.assertIn("Custom settings", html)
        self.assertIn("health-check-btn", html)
        javascript = (
            Path(__file__).parents[1] / "werewolf" / "web" / "static" / "app.js"
        ).read_text(encoding="utf-8")
        self.assertIn("healthRelevantControls", javascript)
        self.assertIn("Number.isInteger", javascript)

    def test_new_game_rejects_malformed_and_non_object_json(self):
        malformed = self.client.post(
            "/api/new", data="{broken", content_type="application/json",
        )
        self.assertEqual(malformed.status_code, 400)
        self.assertEqual(malformed.get_json()["errors"]["request"]["code"], "invalid_json")
        empty = self.client.post("/api/new")
        self.assertEqual(empty.status_code, 400)
        self.assertEqual(empty.get_json()["errors"]["request"]["code"], "invalid_json")
        for value in (None, False, [], 0, "text"):
            response = self.client.post(
                "/api/new", data=json.dumps(value), content_type="application/json",
            )
            self.assertEqual(response.status_code, 400, value)
            self.assertEqual(response.get_json()["errors"]["request"]["code"], "invalid_type")

    def test_health_check_rejects_malformed_and_non_object_json(self):
        malformed = self.client.post(
            "/api/models/fast/health-check",
            data="{broken", content_type="application/json",
        )
        self.assertEqual(malformed.status_code, 400)
        self.assertEqual(malformed.get_json()["errors"]["request"]["code"], "invalid_json")
        response = self.client.post("/api/models/fast/health-check", json=[])
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["errors"]["request"]["code"], "invalid_type")

    def test_failed_creation_preserves_active_engine(self):
        old_engine = mock.Mock()
        self.webapp.game_engine = old_engine
        with mock.patch.object(
            self.webapp, "create_engine_from_payload",
            side_effect=RequestValidationError({"model": {"code": "missing_key", "message": "missing"}}),
        ):
            response = self.client.post("/api/new", json={"model": "fast"})
        self.assertEqual(response.status_code, 400)
        self.assertIs(self.webapp.game_engine, old_engine)
        old_engine.close.assert_not_called()
        self.webapp.game_engine = None

    def test_successful_creation_swaps_then_closes_old_engine(self):
        old_engine = mock.Mock()
        new_engine = mock.Mock()
        new_engine.get_state_dict.return_value = {"game_id": "new"}
        self.webapp.game_engine = old_engine
        with mock.patch.object(self.webapp, "create_engine_from_payload", return_value=new_engine):
            response = self.client.post("/api/new", json={"model": "fast"})
        self.assertEqual(response.status_code, 200)
        self.assertIs(self.webapp.game_engine, new_engine)
        old_engine.close.assert_called_once_with()
        self.webapp.game_engine = None

    def test_state_serialization_failure_preserves_old_game_and_closes_new(self):
        old_engine = mock.Mock()
        new_engine = mock.Mock()
        new_engine.get_state_dict.side_effect = RuntimeError("cannot serialize")
        self.webapp.game_engine = old_engine
        with mock.patch.object(self.webapp, "create_engine_from_payload", return_value=new_engine):
            response = self.client.post("/api/new", json={"model": "fast"})
        self.assertEqual(response.status_code, 500)
        self.assertIs(self.webapp.game_engine, old_engine)
        new_engine.close.assert_called_once_with()
        old_engine.close.assert_not_called()
        self.webapp.game_engine = None

    def test_old_game_close_failure_does_not_invalidate_new_game(self):
        old_engine = mock.Mock()
        old_engine.close.side_effect = RuntimeError("close failed")
        new_engine = mock.Mock()
        new_engine.get_state_dict.return_value = {"game_id": "new"}
        self.webapp.game_engine = old_engine
        with mock.patch.object(self.webapp, "create_engine_from_payload", return_value=new_engine):
            response = self.client.post("/api/new", json={"model": "fast"})
        self.assertEqual(response.status_code, 200)
        self.assertIs(self.webapp.game_engine, new_engine)
        old_engine.close.assert_called_once_with()
        self.webapp.game_engine = None


if __name__ == "__main__":
    unittest.main()
