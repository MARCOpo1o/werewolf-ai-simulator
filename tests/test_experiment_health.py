import unittest

from werewolf.experiments.conditions import build_crossed_conditions
from werewolf.experiments.health import (
    adjustment_fingerprint,
    evaluate_health_record,
    probe_model,
    unique_health_targets,
)
from werewolf.llm.fake_provider import (
    FakeProvider,
    error_result,
    success_result,
)
from werewolf.llm.records import ErrorCategory


def targets_for(model_a="fast", model_b="gemini_flash_lite",
                generation=None):
    return unique_health_targets(
        build_crossed_conditions(model_a, model_b),
        generation or {"max_output_tokens": 4096},
    )


class HealthTargetTests(unittest.TestCase):
    def test_one_target_per_model_generation_fingerprint(self):
        targets = targets_for()
        self.assertEqual(
            sorted(t["model_name"] for t in targets),
            ["fast", "gemini_flash_lite"],
        )
        self.assertEqual(
            len({t["health_fingerprint"] for t in targets}), 2,
        )

    def test_generation_changes_fingerprint(self):
        base = targets_for(generation={"max_output_tokens": 4096})
        changed = targets_for(generation={"max_output_tokens": 2048})
        base_fp = {t["model_name"]: t["health_fingerprint"] for t in base}
        changed_fp = {t["model_name"]: t["health_fingerprint"]
                      for t in changed}
        for model in base_fp:
            self.assertNotEqual(base_fp[model], changed_fp[model])

    def test_registry_reasoning_default_reaches_effective_generation(self):
        targets = targets_for(model_b="gemini_flash")
        gemini = next(t for t in targets
                      if t["model_name"] == "gemini_flash")
        self.assertEqual(
            gemini["effective_generation"]["reasoning_effort"], "low",
        )


class ProbeTests(unittest.TestCase):
    def _target(self):
        return next(t for t in targets_for() if t["model_name"] == "fast")

    def test_ready_probe(self):
        provider = FakeProvider([success_result(
            {"health": "ok"}, resolved_model="grok-4.3",
        )])
        record = probe_model(self._target(), provider=provider)
        self.assertEqual(record["status"], "ready")
        self.assertEqual(record["adjustments"]["model_identity"], "matched")
        self.assertEqual(record["cost_completeness"], "provider_reported")
        self.assertIsNone(record["sanitized_error"])
        self.assertNotIn("adjustment_fingerprint", record)

    def test_adjusted_probe_carries_fingerprint(self):
        result = success_result({"health": "ok"}, resolved_model="grok-4.3")
        result.provider_metadata = {"generation_dropped": ["provider_seed"]}
        record = probe_model(self._target(),
                             provider=FakeProvider([result]))
        self.assertEqual(record["status"], "adjusted")
        self.assertEqual(
            record["adjustment_fingerprint"],
            adjustment_fingerprint(
                health_fingerprint=record["health_fingerprint"],
                generation_dropped=["provider_seed"],
                generation_adjusted=[],
                model_identity="matched",
                resolved_model="grok-4.3",
            ),
        )

    def test_mismatched_model_fails(self):
        record = probe_model(
            self._target(),
            provider=FakeProvider([success_result(
                {"health": "ok"}, resolved_model="grok-2-mini",
            )]),
        )
        self.assertEqual(record["status"], "failed")
        self.assertEqual(
            record["adjustments"]["model_identity"], "mismatched",
        )

    def test_api_error_fails_but_preserves_cost(self):
        record = probe_model(
            self._target(),
            provider=FakeProvider([error_result(
                ErrorCategory.PROVIDER_ERROR, cost_ticks=5_000_000,
            )]),
        )
        self.assertEqual(record["status"], "failed")
        self.assertIsNotNone(record["sanitized_error"])
        self.assertIsNotNone(record["cost"])
        self.assertEqual(record["cost_completeness"], "provider_reported")


class HealthPolicyTests(unittest.TestCase):
    def _adjusted_record(self):
        result = success_result({"health": "ok"}, resolved_model="grok-4.3")
        result.provider_metadata = {"generation_dropped": ["provider_seed"]}
        target = next(t for t in targets_for() if t["model_name"] == "fast")
        return probe_model(target, provider=FakeProvider([result]))

    def test_ready_always_accepted(self):
        target = next(t for t in targets_for() if t["model_name"] == "fast")
        record = probe_model(target, provider=FakeProvider([
            success_result({"health": "ok"}, resolved_model="grok-4.3"),
        ]))
        self.assertIsNone(evaluate_health_record(
            record, predeclared_fingerprints=[], allow_adjusted=False,
        ))

    def test_undeclared_adjustment_blocks_even_with_flag(self):
        record = self._adjusted_record()
        reason = evaluate_health_record(
            record, predeclared_fingerprints=[], allow_adjusted=True,
        )
        self.assertIn("not predeclared", reason)

    def test_declared_adjustment_requires_flag(self):
        record = self._adjusted_record()
        declared = [record["adjustment_fingerprint"]]
        reason = evaluate_health_record(
            record, predeclared_fingerprints=declared, allow_adjusted=False,
        )
        self.assertIn("--allow-adjusted-health", reason)
        self.assertIsNone(evaluate_health_record(
            record, predeclared_fingerprints=declared, allow_adjusted=True,
        ))

    def test_changed_adjustment_blocks(self):
        record = self._adjusted_record()
        other = adjustment_fingerprint(
            health_fingerprint=record["health_fingerprint"],
            generation_dropped=["temperature"],
            generation_adjusted=[],
            model_identity="matched",
            resolved_model="grok-4.3",
        )
        self.assertIsNotNone(evaluate_health_record(
            record, predeclared_fingerprints=[other], allow_adjusted=True,
        ))

    def test_same_adjustment_for_another_health_target_is_not_authorized(self):
        record = self._adjusted_record()
        other_target = adjustment_fingerprint(
            health_fingerprint="f" * 64,
            generation_dropped=["provider_seed"],
            generation_adjusted=[],
            model_identity="matched",
            resolved_model="grok-4.3",
        )
        self.assertIsNotNone(evaluate_health_record(
            record,
            predeclared_fingerprints=[other_target],
            allow_adjusted=True,
        ))

    def test_resolved_model_is_part_of_adjustment_authorization(self):
        record = self._adjusted_record()
        other_resolution = adjustment_fingerprint(
            health_fingerprint=record["health_fingerprint"],
            generation_dropped=["provider_seed"],
            generation_adjusted=[],
            model_identity="matched",
            resolved_model="grok-4.3-redirect",
        )
        self.assertIsNotNone(evaluate_health_record(
            record,
            predeclared_fingerprints=[other_resolution],
            allow_adjusted=True,
        ))

    def test_failed_blocks(self):
        target = next(t for t in targets_for() if t["model_name"] == "fast")
        record = probe_model(target, provider=FakeProvider([
            error_result(ErrorCategory.RATE_LIMITED),
        ]))
        self.assertIsNotNone(evaluate_health_record(
            record, predeclared_fingerprints=[], allow_adjusted=True,
        ))


if __name__ == "__main__":
    unittest.main()
