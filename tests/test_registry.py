import json
import os
import unittest
from unittest import mock

from werewolf.llm.registry import (
    MODEL_REGISTRY,
    ModelSpec,
    get_api_key,
    registry_snapshot,
    resolved_model_matches,
    selectable_models,
    resolve,
)


class ResolveTests(unittest.TestCase):
    def test_aliases_point_at_current_models(self):
        # fast omits reasoning_effort: xai-sdk 1.17.0 only accepts
        # 'low'/'high' client-side, so "none" broke every call.
        self.assertEqual(resolve("fast").model, "grok-4.3")
        self.assertIsNone(resolve("fast").reasoning_effort)
        self.assertEqual(resolve("reasoning").model, "grok-4.3")
        self.assertEqual(resolve("reasoning").reasoning_effort, "low")
        self.assertEqual(resolve("fast").provider, "xai")

    def test_full_model_id_passthrough(self):
        spec = resolve("grok-4.3")
        self.assertIsNone(spec.alias)
        self.assertEqual(spec.model, "grok-4.3")
        self.assertEqual(spec.provider, "xai")

    def test_gemini_aliases_route_to_litellm(self):
        # Regression: a gemini alias/slug must never reach the xAI provider
        # (live run sent gemini/... to xAI -> 'Model not found' fallbacks).
        for name in ("gemini_flash_lite", "gemini_flash",
                     "gemini/gemini-3.5-flash"):
            spec = resolve(name)
            self.assertEqual(spec.provider, "litellm", name)
            self.assertEqual(spec.api_key_env, ("GEMINI_API_KEY",), name)

    def test_claude_and_openai_aliases_route_to_litellm(self):
        cases = {
            "claude_haiku": ("anthropic/claude-haiku-4-5-20251001",
                             ("ANTHROPIC_API_KEY",)),
            "claude_sonnet": ("anthropic/claude-sonnet-5",
                              ("ANTHROPIC_API_KEY",)),
            "gpt_nano": ("openai/gpt-5.4-nano-2026-03-17",
                         ("OPENAI_API_KEY",)),
            "gpt_luna": ("openai/gpt-5.6-luna", ("OPENAI_API_KEY",)),
        }
        for alias, (model, key_env) in cases.items():
            spec = resolve(alias)
            self.assertEqual(spec.provider, "litellm", alias)
            self.assertEqual(spec.model, model, alias)
            self.assertEqual(spec.api_key_env, key_env, alias)

    def test_prefixed_ids_route_to_litellm(self):
        self.assertEqual(resolve("anthropic/claude-x").provider, "litellm")
        self.assertEqual(
            resolve("anthropic/claude-x").api_key_env, ("ANTHROPIC_API_KEY",)
        )
        # bare IDs keep historical xAI behavior
        self.assertEqual(resolve("grok-4.5").provider, "xai")

    def test_build_provider_dispatches_by_spec(self):
        try:
            import litellm  # noqa: F401
        except ImportError:
            self.skipTest("litellm not installed")
        from werewolf.llm.litellm_provider import LiteLLMProvider
        from werewolf.llm.registry import build_provider
        result = build_provider(resolve("gemini_flash"), api_key="test-key")
        self.assertTrue(result.ok)
        self.assertIsInstance(result.provider, LiteLLMProvider)
        missing = build_provider(resolve("gemini_flash"), api_key="")
        self.assertFalse(missing.ok)
        self.assertEqual(missing.status.value, "missing_credential")

    def test_registry_specs_are_frozen(self):
        with self.assertRaises(Exception):
            MODEL_REGISTRY["fast"].model = "other"

    def test_provider_build_statuses_and_secret_redaction(self):
        from werewolf.llm.registry import ProviderBuildStatus, build_provider
        spec = resolve("gemini_flash")
        with mock.patch(
            "werewolf.llm.litellm_provider.LiteLLMProvider",
            side_effect=RuntimeError("litellm is not installed"),
        ):
            dependency = build_provider(spec, api_key="secret-value")
        self.assertEqual(dependency.status, ProviderBuildStatus.DEPENDENCY_UNAVAILABLE)

        with mock.patch(
            "werewolf.llm.litellm_provider.LiteLLMProvider",
            side_effect=ValueError("initialization rejected secret-value"),
        ):
            failed = build_provider(spec, api_key="secret-value")
        self.assertEqual(failed.status, ProviderBuildStatus.INITIALIZATION_FAILED)
        self.assertNotIn("secret-value", failed.error)


class ApiKeyTests(unittest.TestCase):
    def test_lookup_order_matches_legacy(self):
        # Legacy get_api_key(): GROK_API_KEY first, then XAI_API_KEY.
        spec = resolve("fast")
        with mock.patch.dict(os.environ, {"GROK_API_KEY": "k-grok",
                                          "XAI_API_KEY": "k-xai"}):
            self.assertEqual(get_api_key(spec), "k-grok")
        with mock.patch.dict(os.environ, {"XAI_API_KEY": "k-xai"}, clear=False):
            os.environ.pop("GROK_API_KEY", None)
            self.assertEqual(get_api_key(spec), "k-xai")

    def test_empty_env_returns_empty_string(self):
        spec = resolve("fast")
        with mock.patch.dict(os.environ, {"GROK_API_KEY": "", "XAI_API_KEY": ""}):
            self.assertEqual(get_api_key(spec), "")

    def test_no_key_material_in_spec_or_snapshot(self):
        secret = "xai-VERY-SECRET-VALUE"
        with mock.patch.dict(os.environ, {"GROK_API_KEY": secret}):
            spec = resolve("fast")
            self.assertNotIn(secret, repr(spec))
            snapshot = json.dumps(registry_snapshot())
            self.assertNotIn(secret, snapshot)
            # Snapshot names the env var, not its value.
            self.assertIn("GROK_API_KEY", snapshot)


class SnapshotTests(unittest.TestCase):
    def test_snapshot_covers_all_aliases_and_serializes(self):
        snapshot = registry_snapshot()
        self.assertEqual(set(snapshot.keys()), set(MODEL_REGISTRY.keys()))
        json.dumps(snapshot)
        for entry in snapshot.values():
            self.assertIn("provider", entry)
            self.assertIn("model", entry)
            self.assertIn("api_key_env", entry)


class CatalogTests(unittest.TestCase):
    def test_selectable_catalog_has_complete_factual_metadata(self):
        specs = selectable_models()
        self.assertTrue(specs)
        self.assertEqual(specs, sorted(specs, key=lambda s: (s.sort_order, s.alias)))
        for spec in specs:
            self.assertTrue(spec.alias)
            self.assertTrue(spec.display_name)
            self.assertTrue(spec.family)
            self.assertTrue(spec.description)
            self.assertIn(spec.speed_tier, {"fast", "medium", "slow"})
            self.assertIn(spec.cost_tier, {"low", "medium", "high"})

    def test_resolved_model_matching_is_explicit(self):
        spec = resolve("gpt_nano")
        self.assertTrue(resolved_model_matches(spec, spec.model))
        self.assertTrue(resolved_model_matches(spec, "gpt-5.4-nano-2026-03-17"))
        self.assertFalse(resolved_model_matches(spec, "gpt-5.4"))
        self.assertFalse(resolved_model_matches(spec, "gpt-5.4-preview"))
        self.assertFalse(resolved_model_matches(spec, None))


if __name__ == "__main__":
    unittest.main()
