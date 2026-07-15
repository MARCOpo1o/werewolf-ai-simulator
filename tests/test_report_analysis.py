import unittest

from werewolf.reporting.analysis import (
    build_belief_analysis,
    build_decision_analysis,
    build_manipulation_signals,
)


CONFIG = {
    "belief_schema_version": 1,
    "role_map": {
        "0": {"role": "villager", "team": "village"},
        "1": {"role": "villager", "team": "village"},
        "2": {"role": "werewolf", "team": "wolf"},
    },
}


def snapshot(event_id, player, checkpoint, probs, *, suspicion=None, influential=None):
    return {
        "id": event_id, "event_id": f"evt_{event_id:06d}",
        "source_line": event_id + 2, "round": 1, "phase": "day_assess",
        "type": "belief_snapshot", "channel": "moderator_only",
        "speaker_id": player, "source_call_id": f"call-{event_id}",
        "discussion_cycle": None,
        "payload": {
            "schema_version": 1, "checkpoint": checkpoint,
            "wolf_probabilities": {str(k): v for k, v in probs.items()},
            "intended_vote": 2, "vote_confidence": 0.7,
            "most_influential_recent_speaker": influential,
            "estimated_suspicion_of_me": (
                {str(k): v for k, v in suspicion.items()} if suspicion else None
            ),
            "valid": True,
        },
    }


def belief_timeline():
    return [
        snapshot(0, 0, "pre_discussion", {1: 0.2, 2: 0.7}),
        snapshot(1, 1, "pre_discussion", {0: 0.6, 2: 0.3}),
        snapshot(2, 2, "pre_discussion", {0: 0.0, 1: 0.0},
                 suspicion={0: 0.5, 1: 0.4}),
        {
            "id": 3, "event_id": "evt_000003", "source_line": 5,
            "round": 1, "phase": "day_discuss", "type": "message",
            "channel": "public", "speaker_id": 2, "source_call_id": "call-3",
            "discussion_cycle": 1, "payload": {"text": "Consider P1 instead."},
        },
        snapshot(4, 0, "post_discussion", {1: 0.8, 2: 0.3}, influential=2),
        snapshot(5, 1, "post_discussion", {0: 0.2, 2: 0.9}),
        snapshot(6, 2, "post_discussion", {0: 0.0, 1: 0.0},
                 suspicion={0: 0.4, 1: 0.8}),
        {
            "id": 7, "event_id": "evt_000007", "source_line": 9,
            "round": 1, "phase": "day_vote", "type": "vote",
            "channel": "public", "speaker_id": 0, "source_call_id": "call-7",
            "discussion_cycle": None,
            "payload": {"voter_id": 0, "target_id": 1},
        },
        {
            "id": 8, "event_id": "evt_000008", "source_line": 10,
            "round": 1, "phase": "day_vote", "type": "vote",
            "channel": "public", "speaker_id": 1, "source_call_id": "call-8",
            "discussion_cycle": None,
            "payload": {"voter_id": 1, "target_id": 2},
        },
    ]


class BeliefReportTests(unittest.TestCase):
    def test_probability_accuracy_and_revisions_are_descriptive(self):
        beliefs = build_belief_analysis(CONFIG, belief_timeline())
        self.assertTrue(beliefs["available"])
        self.assertEqual(beliefs["summary"]["harmful_revisions"], 1)
        self.assertEqual(beliefs["summary"]["beneficial_revisions"], 1)
        contribution = next(
            item for item in beliefs["trajectories"]
            if item["observer_id"] == 0 and item["target_id"] == 2
            and item["checkpoint"] == "pre_discussion"
        )
        self.assertAlmostEqual(contribution["squared_error"], 0.09)
        self.assertAlmostEqual(contribution["brier_score_contribution"], 0.09)
        self.assertNotIn("calibration", str(beliefs).lower())

    def test_brier_label_requires_known_schema(self):
        timeline = belief_timeline()
        timeline[0]["payload"]["schema_version"] = 99
        beliefs = build_belief_analysis(CONFIG, timeline)
        item = next(
            item for item in beliefs["trajectories"]
            if item["event_id"] == "evt_000000"
        )
        self.assertIsNone(item["brier_score_contribution"])
        self.assertFalse(item["brier_schema_applicable"])

    def test_manipulation_signals_are_noncausal_and_source_only(self):
        timeline = belief_timeline()
        beliefs = build_belief_analysis(CONFIG, timeline)
        signals = build_manipulation_signals(CONFIG, timeline, beliefs)
        self.assertTrue(signals["available"])
        self.assertFalse(signals["causal"])
        self.assertEqual(signals["candidate_episodes"][0]["observer_id"], 0)
        self.assertEqual(
            signals["candidate_episodes"][0]["most_influential_recent_speaker"], 2,
        )
        self.assertEqual(len(signals["wolf_suspicion_awareness"]), 4)


class DecisionReportTests(unittest.TestCase):
    def call(self, call_id, line, action="speak_public"):
        return {
            "type": "llm_call", "call_id": call_id, "source_line": line,
            "player_id": 0, "player_role": "villager", "round": 1,
            "phase": "day_discuss", "required_action": action,
            "attempt": 1, "api_attempted": True, "api_ok": True,
            "parse_method": "direct", "validation_ok": True,
            "error_category": "completed", "requested_model": "m",
            "resolved_model": "m",
        }

    def message(self):
        return {
            "event_id": "evt_000001", "source_line": 5, "round": 1,
            "phase": "day_discuss", "type": "message", "channel": "public",
            "speaker_id": 0, "source_call_id": None, "payload": {"text": "x"},
        }

    def test_unique_legacy_candidate_is_inferred(self):
        timeline = [self.message()]
        result = build_decision_analysis(timeline, [self.call("call-a", 4)])
        self.assertEqual(timeline[0]["source_call_id"], "call-a")
        self.assertEqual(timeline[0]["link_quality"], "inferred")
        self.assertEqual(result["attempt_groups"][0]["attempt_count"], 1)

    def test_multiple_legacy_candidates_are_ambiguous(self):
        timeline = [self.message()]
        calls = [self.call("call-a", 3), self.call("call-b", 4)]
        build_decision_analysis(timeline, calls)
        self.assertIsNone(timeline[0]["source_call_id"])
        self.assertEqual(timeline[0]["link_quality"], "ambiguous")


if __name__ == "__main__":
    unittest.main()
