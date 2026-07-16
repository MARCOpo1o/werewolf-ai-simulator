import re
import unittest

from werewolf.agents.prompts import get_prompt_version
from werewolf.experiments.conditions import (
    ConditionError,
    build_crossed_conditions,
    condition_models,
    model_catalog,
    normalize_conditions,
)
from werewolf.experiments.profiles import (
    PromptProfileError,
    resolve_prompt_profile,
    verify_prompt_profile,
)

HEX64 = re.compile(r"^[0-9a-f]{64}$")


class PromptProfileTests(unittest.TestCase):
    def test_baseline_profile_pins_current_prompts(self):
        profile = resolve_prompt_profile("baseline_v1")
        self.assertEqual(profile["name"], "baseline_v1")
        self.assertEqual(
            profile["prompt_source_version"], get_prompt_version()
        )
        hashes = profile["rendered_prompt_hashes"]
        for key in (
            "system_werewolf", "system_seer", "system_villager",
            "limits_notice", "action_vote", "action_speak_public",
            "action_assess_beliefs", "action_runoff_vote",
            "action_wolf_chat", "action_choose_wolf_kill",
            "action_seer_divine",
        ):
            self.assertIn(key, hashes)
            self.assertRegex(hashes[key], HEX64)

    def test_profile_resolution_is_deterministic(self):
        self.assertEqual(
            resolve_prompt_profile("baseline_v1"),
            resolve_prompt_profile("baseline_v1"),
        )

    def test_unknown_profile_rejected(self):
        with self.assertRaises(PromptProfileError):
            resolve_prompt_profile("experimental_v9")

    def test_verify_detects_drifted_render_hashes(self):
        pinned = resolve_prompt_profile("baseline_v1")
        self.assertEqual(verify_prompt_profile(pinned), [])
        tampered = dict(pinned)
        tampered["rendered_prompt_hashes"] = dict(
            pinned["rendered_prompt_hashes"], system_seer="0" * 64,
        )
        drift = verify_prompt_profile(tampered)
        self.assertTrue(any("system_seer" in d for d in drift))

    def test_verify_detects_source_change(self):
        pinned = dict(resolve_prompt_profile("baseline_v1"))
        pinned["prompt_source_version"] = "deadbeef0000"
        drift = verify_prompt_profile(pinned)
        self.assertTrue(any("prompt source changed" in d for d in drift))


class ConditionTests(unittest.TestCase):
    def test_crossed_helper_materializes_producer_target_matrix(self):
        conditions = build_crossed_conditions("fast", "gemini_flash_lite")
        self.assertEqual(
            set(conditions),
            {"a_homogeneous", "b_homogeneous",
             "a_wolves_b_village", "b_wolves_a_village"},
        )
        self.assertEqual(conditions["a_homogeneous"]["role_models"], {
            "werewolf": "fast", "villager": "fast", "seer": "fast",
        })
        self.assertEqual(conditions["a_wolves_b_village"]["role_models"], {
            "werewolf": "fast",
            "villager": "gemini_flash_lite",
            "seer": "gemini_flash_lite",
        })
        self.assertEqual(conditions["b_wolves_a_village"]["role_models"], {
            "werewolf": "gemini_flash_lite",
            "villager": "fast",
            "seer": "fast",
        })

    def test_crossed_helper_requires_distinct_models(self):
        with self.assertRaises(ConditionError):
            build_crossed_conditions("fast", "fast")

    def test_arbitrary_explicit_conditions_supported(self):
        conditions = normalize_conditions({
            "mixed_wolves": {
                "role_models": {
                    "werewolf": "fast",
                    "villager": "claude_haiku",
                    "seer": "gpt_nano",
                },
                "description": "three-model condition",
            },
        })
        self.assertEqual(
            condition_models(conditions),
            ["claude_haiku", "fast", "gpt_nano"],
        )

    def test_invalid_conditions_rejected(self):
        for bad in (
            {},  # empty
            {"Bad ID": {"role_models": {
                "werewolf": "fast", "villager": "fast", "seer": "fast"}}},
            {"c1": {"role_models": {"werewolf": "fast"}}},  # missing roles
            {"c1": {"role_models": {
                "werewolf": "fast", "villager": "fast", "seer": "fast",
                "witch": "fast"}}},  # unsupported role
            {"c1": {"role_models": {
                "werewolf": "fast", "villager": None, "seer": "fast"}}},
            {"c1": {"role_models": {
                "werewolf": "fast", "villager": "fast", "seer": "fast"},
                "extra": 1}},
        ):
            with self.assertRaises(ConditionError, msg=repr(bad)):
                normalize_conditions(bad)

    def test_model_catalog_reports_provider_mapping(self):
        conditions = build_crossed_conditions("fast", "gemini_flash_lite")
        catalog = model_catalog(conditions)
        self.assertEqual(set(catalog), {"fast", "gemini_flash_lite"})
        self.assertEqual(catalog["fast"]["provider"], "xai")
        self.assertEqual(catalog["fast"]["requested_model"], "grok-4.3")
        self.assertEqual(
            catalog["gemini_flash_lite"]["provider"], "litellm"
        )
        self.assertEqual(
            catalog["gemini_flash_lite"]["api_key_env"], ["GEMINI_API_KEY"]
        )


if __name__ == "__main__":
    unittest.main()
