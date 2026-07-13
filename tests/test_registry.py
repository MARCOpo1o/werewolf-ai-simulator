import json
import os
import unittest
from unittest import mock

from werewolf.llm.registry import (
    MODEL_REGISTRY,
    ModelSpec,
    get_api_key,
    registry_snapshot,
    resolve,
)


class ResolveTests(unittest.TestCase):
    def test_aliases_point_at_current_models(self):
        # Deliberately updated 2026-07: grok-4-1-fast* retired 2026-05-15;
        # aliases now target grok-4.3 with explicit reasoning effort.
        self.assertEqual(resolve("fast").model, "grok-4.3")
        self.assertEqual(resolve("fast").reasoning_effort, "none")
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
            "gpt_mini": ("openai/gpt-4o-mini", ("OPENAI_API_KEY",)),
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
        provider = build_provider(resolve("gemini_flash"), api_key="test-key")
        self.assertIsInstance(provider, LiteLLMProvider)
        self.assertIsNone(build_provider(resolve("gemini_flash"), api_key=""))

    def test_registry_specs_are_frozen(self):
        with self.assertRaises(Exception):
            MODEL_REGISTRY["fast"].model = "other"


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


if __name__ == "__main__":
    unittest.main()
