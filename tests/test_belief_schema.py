import unittest

from werewolf.engine.beliefs import (
    CHECKPOINT_PRE,
    coerce_probability,
    coerce_prob_map,
    parse_belief_snapshot,
    validate_assess_beliefs,
)


class CoerceProbabilityTests(unittest.TestCase):
    def test_valid_forms(self):
        self.assertEqual(coerce_probability(0.65), 0.65)
        self.assertEqual(coerce_probability(1), 1.0)
        self.assertEqual(coerce_probability(0), 0.0)
        self.assertEqual(coerce_probability("0.4"), 0.4)
        self.assertEqual(coerce_probability(1.0000001), 1.0)  # drift clamped

    def test_invalid_forms(self):
        for bad in (
            True, False, 65, -0.5, "high", None, [0.5],
            float("nan"), float("inf"),
        ):
            self.assertIsNone(coerce_probability(bad), bad)


class CoerceProbMapTests(unittest.TestCase):
    def test_tolerant_keys(self):
        parsed, problems = coerce_prob_map(
            {"P1": 0.2, "2": "0.5", 3: 0.9}, {1, 2, 3}
        )
        self.assertEqual(parsed, {1: 0.2, 2: 0.5, 3: 0.9})
        self.assertEqual(problems, [])

    def test_problems_reported_not_guessed(self):
        parsed, problems = coerce_prob_map(
            {"1": 0.2, "junk": 0.5, "2": "very likely", "9": 0.1}, {1, 2, 3}
        )
        self.assertEqual(parsed, {1: 0.2})
        self.assertEqual(len(problems), 4)  # bad key, bad prob, unexpected, missing

    def test_non_dict(self):
        parsed, problems = coerce_prob_map([0.1, 0.2], {1})
        self.assertEqual(parsed, {})
        self.assertTrue(problems)


class ParseSnapshotTests(unittest.TestCase):
    ALIVE = [0, 1, 2, 3]

    def test_valid_villager_snapshot(self):
        snapshot = parse_belief_snapshot(
            {
                "wolf_probabilities": {"1": 0.1, "2": 0.7, "3": 0.2},
                "intended_vote": 2,
                "vote_confidence": 0.8,
                "most_influential_recent_speaker": 3,
                "estimated_suspicion_of_me": None,
            },
            CHECKPOINT_PRE, self_id=0, alive_ids=self.ALIVE, is_wolf=False,
        )
        self.assertTrue(snapshot.valid)
        self.assertEqual(snapshot.intended_vote, 2)
        payload = snapshot.to_payload()
        self.assertEqual(payload["wolf_probabilities"]["2"], 0.7)
        self.assertTrue(payload["valid"])

    def test_missing_and_malformed(self):
        missing = parse_belief_snapshot(
            None, CHECKPOINT_PRE, 0, self.ALIVE, False)
        self.assertFalse(missing.valid)
        self.assertEqual(missing.invalid_reason, "missing")

        malformed = parse_belief_snapshot(
            "garbage", CHECKPOINT_PRE, 0, self.ALIVE, False)
        self.assertFalse(malformed.valid)
        self.assertEqual(malformed.invalid_reason, "malformed")

    def test_partial_keeps_what_parsed(self):
        snapshot = parse_belief_snapshot(
            {"wolf_probabilities": {"1": 0.4}},  # 2, 3 missing
            CHECKPOINT_PRE, 0, self.ALIVE, False,
        )
        self.assertFalse(snapshot.valid)
        self.assertEqual(snapshot.invalid_reason, "partial")
        self.assertEqual(snapshot.wolf_probabilities, {1: 0.4})

    def test_wolf_requires_second_order_beliefs(self):
        base = {"wolf_probabilities": {"1": 0.0, "2": 0.5, "3": 0.5}}
        no_suspicion = parse_belief_snapshot(
            base, CHECKPOINT_PRE, 0, self.ALIVE, is_wolf=True)
        self.assertFalse(no_suspicion.valid)

        with_suspicion = parse_belief_snapshot(
            {**base, "estimated_suspicion_of_me": {"1": 0.3, "2": 0.6, "3": 0.1}},
            CHECKPOINT_PRE, 0, self.ALIVE, is_wolf=True)
        self.assertTrue(with_suspicion.valid)
        self.assertEqual(with_suspicion.estimated_suspicion_of_me[2], 0.6)

    def test_villager_never_requires_second_order(self):
        snapshot = parse_belief_snapshot(
            {"wolf_probabilities": {"1": 0.1, "2": 0.2, "3": 0.3}},
            CHECKPOINT_PRE, 0, self.ALIVE, is_wolf=False)
        self.assertTrue(snapshot.valid)
        self.assertIsNone(snapshot.estimated_suspicion_of_me)


class ValidateAssessTests(unittest.TestCase):
    def _observation(self, role="villager"):
        return {
            "required_action": "assess_beliefs",
            "self": {"id": 0, "role": role, "team": "village"},
            "alive_players": [{"id": 0}, {"id": 1}, {"id": 2}],
            "round": 1, "phase": "day_assess",
        }

    def test_strict_accept(self):
        ok, error = validate_assess_beliefs(self._observation(), {
            "beliefs": {"wolf_probabilities": {"1": 0.5, "2": 0.5}},
        })
        self.assertTrue(ok, error)

    def test_strict_reject_with_actionable_error(self):
        ok, error = validate_assess_beliefs(self._observation(), {
            "beliefs": {"wolf_probabilities": {"1": 0.5}},
        })
        self.assertFalse(ok)
        self.assertIn("P2", error)  # names the missing player


if __name__ == "__main__":
    unittest.main()
