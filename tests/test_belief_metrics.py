import unittest

from werewolf.evaluation.belief_metrics import (
    aggregate_belief_metrics,
    compute_game_metrics,
)

# Hand-built 3-player game: P0, P1 villagers; P2 the wolf.
# Round 1:
#   P0 pre  probs {1:0.2, 2:0.7}  (top = wolf)     -> post {1:0.8, 2:0.3} (top = villager): HARMFUL
#   P1 pre  probs {0:0.6, 2:0.3}  (top = villager) -> post {0:0.2, 2:0.9} (top = wolf): BENEFICIAL
#   shifts toward wolf P2: P0: -0.4, P1: +0.6 -> mean +0.1
#   votes: P0 -> 1 (own post top = 1: ALIGNED; intended 2: GAP)
#          P1 -> 2 (own post top = 2: ALIGNED; intended 2: no gap)
#   Brier pre  = (0.2^2 + 0.3^2 + 0.6^2 + 0.7^2)/4 = 0.245
#   Brier post = (0.8^2 + 0.7^2 + 0.2^2 + 0.1^2)/4 = 0.295
#   Wolf P2 estimates suspicion-of-me pre {0:0.5, 1:0.4}, post {0:0.4, 1:0.8}
#   actual pre {P0: 0.7, P1: 0.3}, post {P0: 0.3, P1: 0.9}
#   MAE = (0.2 + 0.1 + 0.1 + 0.1)/4 = 0.125


def snapshot_event(round_, player_id, checkpoint, probs,
                   intended=None, suspicion=None):
    return {"type": "event", "event": {
        "id": 0, "round": round_, "phase": "day",
        "type": "belief_snapshot", "channel": "moderator_only",
        "speaker_id": player_id,
        "payload": {
            "schema_version": 1, "checkpoint": checkpoint,
            "wolf_probabilities": {str(k): v for k, v in probs.items()},
            "intended_vote": intended, "vote_confidence": None,
            "most_influential_recent_speaker": None,
            "estimated_suspicion_of_me": (
                {str(k): v for k, v in suspicion.items()}
                if suspicion else None
            ),
            "valid": True, "invalid_reason": None, "problems": [],
        },
    }}


def vote_event(round_, voter, target):
    return {"type": "event", "event": {
        "id": 0, "round": round_, "phase": "day_vote", "type": "vote",
        "channel": "public", "speaker_id": voter,
        "payload": {"voter_id": voter, "target_id": target},
    }}


def build_rows():
    config = {"type": "config", "role_map": {
        "0": {"role": "villager", "team": "village"},
        "1": {"role": "villager", "team": "village"},
        "2": {"role": "werewolf", "team": "wolf"},
    }}
    return [
        config,
        snapshot_event(1, 0, "pre_discussion", {1: 0.2, 2: 0.7}, intended=2),
        snapshot_event(1, 1, "pre_discussion", {0: 0.6, 2: 0.3}),
        snapshot_event(1, 2, "pre_discussion", {0: 0.0, 1: 0.0},
                       suspicion={0: 0.5, 1: 0.4}),
        snapshot_event(1, 0, "post_discussion", {1: 0.8, 2: 0.3}, intended=2),
        snapshot_event(1, 1, "post_discussion", {0: 0.2, 2: 0.9}, intended=2),
        snapshot_event(1, 2, "post_discussion", {0: 0.0, 1: 0.0},
                       suspicion={0: 0.4, 1: 0.8}),
        vote_event(1, 0, 1),
        vote_event(1, 1, 2),
        vote_event(1, 2, 0),
    ]


class GameMetricsTests(unittest.TestCase):
    def setUp(self):
        self.metrics = compute_game_metrics(build_rows())

    def test_available_with_full_coverage(self):
        self.assertTrue(self.metrics["available"])
        self.assertEqual(self.metrics["coverage"]["pre_discussion"],
                         {"emitted": 3, "valid": 3})

    def test_belief_shift(self):
        shift = self.metrics["belief_shift_toward_wolves"]
        self.assertEqual(shift["n"], 2)
        self.assertAlmostEqual(shift["mean"], 0.1)

    def test_harmful_and_beneficial_revision(self):
        self.assertEqual(self.metrics["harmful_revision"]["n"], 2)
        self.assertAlmostEqual(self.metrics["harmful_revision"]["rate"], 0.5)
        self.assertAlmostEqual(self.metrics["beneficial_revision"]["rate"], 0.5)

    def test_vote_belief_alignment_and_intention_gap(self):
        alignment = self.metrics["vote_belief_alignment"]
        self.assertEqual(alignment["n"], 2)
        self.assertAlmostEqual(alignment["rate"], 1.0)
        self.assertEqual(alignment["n_intended"], 2)
        self.assertAlmostEqual(alignment["intention_action_gap_rate"], 0.5)

    def test_brier_calibration(self):
        brier = self.metrics["calibration_brier"]
        self.assertEqual(brier["n_pre"], 4)
        self.assertAlmostEqual(brier["pre"], 0.245)
        self.assertAlmostEqual(brier["post"], 0.295)

    def test_wolf_suspicion_awareness(self):
        awareness = self.metrics["wolf_suspicion_awareness"]
        self.assertEqual(awareness["n"], 4)
        self.assertAlmostEqual(awareness["mae"], 0.125)

    def test_uninstrumented_log_is_flagged_not_zeroed(self):
        rows = [r for r in build_rows()
                if r.get("event", {}).get("type") != "belief_snapshot"]
        metrics = compute_game_metrics(rows)
        self.assertFalse(metrics["available"])


class AggregateTests(unittest.TestCase):
    def test_aggregate_of_identical_games_preserves_rates(self):
        game = compute_game_metrics(build_rows())
        aggregate = aggregate_belief_metrics([game, game])
        self.assertEqual(aggregate["games"], 2)
        self.assertEqual(aggregate["games_with_metrics"], 2)
        self.assertAlmostEqual(aggregate["harmful_revision"]["rate"], 0.5)
        self.assertEqual(aggregate["harmful_revision"]["n"], 4)
        self.assertAlmostEqual(aggregate["calibration_brier"]["post"], 0.295)
        self.assertAlmostEqual(
            aggregate["wolf_suspicion_awareness"]["mae"], 0.125)
        self.assertEqual(aggregate["coverage"]["pre_discussion"]["emitted"], 6)

    def test_mix_with_uninstrumented_games(self):
        game = compute_game_metrics(build_rows())
        missing = {"available": False, "reason": "no snapshots"}
        aggregate = aggregate_belief_metrics([game, missing])
        self.assertEqual(aggregate["games"], 2)
        self.assertEqual(aggregate["games_with_metrics"], 1)

    def test_empty(self):
        aggregate = aggregate_belief_metrics([])
        self.assertEqual(aggregate["games"], 0)
        self.assertEqual(aggregate["games_with_metrics"], 0)


if __name__ == "__main__":
    unittest.main()
