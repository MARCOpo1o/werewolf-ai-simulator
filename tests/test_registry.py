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
    def test_aliases_match_legacy_presets(self):
        # Must mirror MODEL_PRESETS in cli/run_game.py until the deliberate
        # model-upgrade commit.
        self.assertEqual(resolve("fast").model, "grok-4-1-fast")
        self.assertEqual(resolve("reasoning").model, "grok-4-1-fast-reasoning")
        self.assertEqual(resolve("fast").provider, "xai")

    def test_full_model_id_passthrough(self):
        spec = resolve("grok-4.3")
        self.assertIsNone(spec.alias)
        self.assertEqual(spec.model, "grok-4.3")
        self.assertEqual(spec.provider, "xai")

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
