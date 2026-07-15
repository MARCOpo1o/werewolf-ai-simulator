import unittest

from werewolf.evaluation.stats import bootstrap_ci, paired_bootstrap_diff
from werewolf.evaluation.validity import (
    classify_game,
    summarize_validity,
)


def llm_call(action="vote", *, api_attempted=True, error=None,
             parse_method="direct", validation_ok=True,
             requested="m-1", resolved="m-1"):
    return {
        "type": "llm_call",
        "required_action": action,
        "api_attempted": api_attempted,
        "error_category": error,
        "parse_method": parse_method if api_attempted else None,
        "validation_ok": validation_ok if api_attempted else None,
        "requested_model": requested,
        "resolved_model": resolved if api_attempted else None,
    }


def snapshot_event(valid=True):
    return {"type": "event", "event": {
        "type": "belief_snapshot", "round": 1, "speaker_id": 0,
        "channel": "moderator_only",
        "payload": {
            "checkpoint": "pre_discussion", "valid": valid,
            "wolf_probabilities": {"1": 0.5},
        },
    }}


def rows(calls, config=None, events=()):
    return [config or {"type": "config", "belief_snapshots": False},
            *calls, *events]


class ValidityGateTests(unittest.TestCase):
    def test_clean_game(self):
        result = classify_game(rows([llm_call(), llm_call("speak_public")]))
        self.assertTrue(result["clean"])
        self.assertEqual(result["violations"], {})

    def test_fallback_vote_is_dirty(self):
        result = classify_game(rows([
            llm_call("vote", api_attempted=False, error="fallback_used"),
        ]))
        self.assertFalse(result["clean"])
        self.assertEqual(result["violations"], {"fallback_vote": 1})

    def test_assessment_fallback_is_not_a_violation(self):
        result = classify_game(rows([
            llm_call("assess_beliefs", api_attempted=False,
                     error="fallback_used"),
        ]))
        self.assertTrue(result["clean"])

    def test_regex_recovered_strategic_action_is_dirty(self):
        result = classify_game(rows([
            llm_call("vote", parse_method="regex"),
        ]))
        self.assertEqual(result["violations"], {"regex_recovered_action": 1})

    def test_context_and_unknown_model_failures_are_dirty(self):
        result = classify_game(rows([
            llm_call(error="context_window_exceeded"),
            llm_call(error="unknown_model"),
        ]))
        self.assertEqual(result["violations"], {
            "context_window_exceeded": 1, "unknown_model": 1,
        })

    def test_resolved_model_mismatch_is_dirty(self):
        # silent redirect: requested a retired slug, served another family
        result = classify_game(rows([
            llm_call(requested="grok-4-1-fast", resolved="grok-4.3"),
        ]))
        self.assertEqual(result["violations"], {"resolved_model_mismatch": 1})
        # same family with provider prefix is fine
        clean = classify_game(rows([
            llm_call(requested="gemini/gemini-3.1-flash-lite",
                     resolved="gemini-3.1-flash-lite"),
        ]))
        self.assertTrue(clean["clean"])
        near_name = classify_game(rows([
            llm_call(requested="gpt-5.4", resolved="gpt-5.4-nano"),
        ]))
        self.assertEqual(
            near_name["violations"], {"resolved_model_mismatch": 1},
        )

    def test_low_snapshot_coverage_is_dirty(self):
        config = {"type": "config", "belief_snapshots": True}
        events = [snapshot_event(valid=True)] * 18 + [snapshot_event(False)] * 2
        result = classify_game(rows([llm_call()], config, events))  # 90%
        self.assertEqual(result["violations"], {"low_snapshot_coverage": 1})
        ok = classify_game(
            rows([llm_call()], config, [snapshot_event(True)] * 20)
        )
        self.assertTrue(ok["clean"])

    def test_truthy_snapshot_valid_string_is_not_clean(self):
        config = {"type": "config", "belief_snapshots": True}
        result = classify_game(rows(
            [llm_call()], config, [snapshot_event(valid="false")],
        ))
        self.assertEqual(result["violations"], {"low_snapshot_coverage": 1})

    def test_invalid_probability_makes_claimed_valid_snapshot_dirty(self):
        config = {"type": "config", "belief_snapshots": True}
        event = snapshot_event(valid=True)
        event["event"]["payload"]["wolf_probabilities"].update({
            "2": True,
        })
        result = classify_game(rows([llm_call()], config, [event]))
        self.assertEqual(result["violations"], {"low_snapshot_coverage": 1})

    def test_enabled_instrumentation_with_zero_snapshots_is_dirty(self):
        config = {"type": "config", "belief_snapshots": True}
        result = classify_game(rows([llm_call()], config, []))
        self.assertEqual(
            result["violations"], {"missing_snapshot_instrumentation": 1},
        )

    def test_summarize_validity(self):
        summary = summarize_validity([
            {"clean": True, "violations": {}},
            {"clean": False, "violations": {"fallback_vote": 2}},
            {"clean": False, "violations": {"fallback_vote": 1,
                                            "regex_recovered_action": 3}},
        ])
        self.assertEqual(summary["games"], 3)
        self.assertEqual(summary["clean_games"], 1)
        self.assertEqual(summary["dirty_games"], 2)
        self.assertEqual(summary["violations_by_type"],
                         {"fallback_vote": 3, "regex_recovered_action": 3})


class BootstrapTests(unittest.TestCase):
    def test_constant_values_give_point_interval(self):
        result = bootstrap_ci({s: [0.5, 0.5] for s in range(10)})
        self.assertAlmostEqual(result["estimate"], 0.5)
        self.assertAlmostEqual(result["ci_low"], 0.5)
        self.assertAlmostEqual(result["ci_high"], 0.5)
        self.assertEqual(result["n_seeds"], 10)

    def test_deterministic_and_ordered(self):
        data = {s: [float(s % 3)] for s in range(12)}
        a = bootstrap_ci(data, rng_seed=1)
        b = bootstrap_ci(data, rng_seed=1)
        self.assertEqual(a, b)
        self.assertLessEqual(a["ci_low"], a["estimate"])
        self.assertLessEqual(a["estimate"], a["ci_high"])

    def test_repetitions_collapse_to_seed_mean(self):
        # one seed with reps [0,1] must weigh like a single 0.5
        result = bootstrap_ci({1: [0.0, 1.0]})
        self.assertAlmostEqual(result["estimate"], 0.5)

    def test_empty_returns_none(self):
        self.assertIsNone(bootstrap_ci({}))
        self.assertIsNone(paired_bootstrap_diff({1: [1.0]}, {2: [1.0]}))

    def test_paired_diff_identical_conditions_is_zero(self):
        data = {s: [float(s % 2)] for s in range(10)}
        result = paired_bootstrap_diff(data, dict(data))
        self.assertAlmostEqual(result["estimate"], 0.0)
        self.assertAlmostEqual(result["ci_low"], 0.0)
        self.assertAlmostEqual(result["ci_high"], 0.0)
        self.assertEqual(result["n_common_seeds"], 10)

    def test_paired_diff_constant_shift(self):
        a = {s: [0.8] for s in range(8)}
        b = {s: [0.3] for s in range(8)}
        result = paired_bootstrap_diff(a, b)
        self.assertAlmostEqual(result["estimate"], 0.5)
        self.assertAlmostEqual(result["ci_low"], 0.5)
        self.assertAlmostEqual(result["ci_high"], 0.5)


if __name__ == "__main__":
    unittest.main()
