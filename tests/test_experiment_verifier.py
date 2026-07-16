import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from werewolf.engine.game import GameEngine
from werewolf.experiments.verifier import (
    reconcile_attempt_source,
    verify_terminal_completion,
)

GAME_RULES = {"n_players": 7, "n_wolves": 2, "n_seers": 1}


def run_offline_game(tmpdir, seed=404) -> tuple:
    engine = GameEngine(
        n_players=7, n_wolves=2, n_seers=1, seed=seed,
        output_dir=tmpdir, api_key="",
        transcript_enabled=False, show_all_channels=False,
        allow_provider_fallback=True, belief_snapshots=False,
    )
    winner = engine.run()
    return engine.state.game_id, Path(engine.logger.filepath), winner


def rewrite(path: Path, rows: list) -> None:
    path.write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8",
    )


def load_rows(path: Path) -> list:
    return [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines() if line.strip()]


class VerifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.game_id, cls.log_path, cls.winner = run_offline_game(
            cls._tmp.name,
        )
        cls.rows = load_rows(cls.log_path)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def verify(self, rows, game_id=None, rules=GAME_RULES) -> dict:
        data = "".join(json.dumps(r) + "\n" for r in rows).encode("utf-8")
        return verify_terminal_completion(
            data, expected_game_id=game_id or self.game_id,
            game_rules=rules,
        )

    def test_completed_offline_game_verifies(self):
        result = self.verify(self.rows)
        self.assertTrue(result["complete"], result["reasons"])
        self.assertEqual(result["outcome"]["winner"], self.winner)

    def test_game_id_mismatch_blocks_recovery(self):
        result = self.verify(self.rows, game_id="game_999_1_ff")
        self.assertFalse(result["complete"])
        self.assertFalse(result["checks"]["game_id_match"])

    def test_missing_outcome_blocks_recovery(self):
        rows = [r for r in self.rows if r.get("type") != "outcome"]
        result = self.verify(rows)
        self.assertFalse(result["complete"])
        self.assertFalse(result["checks"]["single_usable_outcome"])

    def test_explicit_failure_abort_is_classified(self):
        rows = [r for r in self.rows if r.get("type") != "outcome"]
        rows.append({
            "type": "abort", "abort_schema_version": 1,
            "reason": "ActionFailureAbort", "round": 2,
            "phase": "day_vote",
        })
        result = self.verify(rows)
        self.assertFalse(result["complete"])
        self.assertEqual(result["terminal_abort"], {
            "reason": "ActionFailureAbort", "classification": "failed",
        })

    def test_operator_abort_remains_interrupted(self):
        rows = [r for r in self.rows if r.get("type") != "outcome"]
        rows.append({
            "type": "abort", "abort_schema_version": 1,
            "reason": "operator_interrupt", "round": 2,
            "phase": "day_vote",
        })
        result = self.verify(rows)
        self.assertEqual(
            result["terminal_abort"]["classification"], "interrupted",
        )

    def test_duplicate_outcome_blocks_recovery(self):
        outcome = next(r for r in self.rows if r.get("type") == "outcome")
        result = self.verify(self.rows + [outcome])
        self.assertFalse(result["complete"])

    def test_flipped_winner_fails_victory_predicate(self):
        rows = [dict(r) for r in self.rows]
        outcome = next(r for r in rows if r.get("type") == "outcome")
        outcome["winner"] = (
            "village" if outcome["winner"] == "wolf" else "wolf"
        )
        result = self.verify(rows)
        self.assertFalse(result["complete"])
        self.assertFalse(result["checks"]["victory_predicate"])

    def test_missing_usage_summary_blocks_recovery(self):
        rows = [r for r in self.rows if r.get("type") != "usage_summary"]
        result = self.verify(rows)
        self.assertFalse(result["complete"])
        self.assertFalse(result["checks"]["usage_summary_present"])

    def test_game_rule_mismatch_blocks_recovery(self):
        result = self.verify(
            self.rows, rules={"n_players": 8, "n_wolves": 2, "n_seers": 1},
        )
        self.assertFalse(result["complete"])
        self.assertFalse(result["checks"]["game_rules_match"])

    def test_unparseable_lines_are_tolerated_when_terminal_state_parses(self):
        data = self.log_path.read_bytes() + b'{"torn": \n'
        result = verify_terminal_completion(
            data, expected_game_id=self.game_id, game_rules=GAME_RULES,
        )
        self.assertTrue(result["complete"])
        self.assertEqual(result["unparseable_lines"], 1)

    def test_verifier_is_independent_of_pr2_analysis_code(self):
        source = (
            Path("werewolf/experiments/verifier.py")
            .read_text(encoding="utf-8")
        )
        for forbidden in ("werewolf.reporting", "werewolf.evaluation"):
            self.assertNotIn(forbidden, source)


class ReconcileSourceTests(unittest.TestCase):
    def test_missing_log_reports_missing_source(self):
        result = reconcile_attempt_source(
            "/nonexistent/game.jsonl",
            expected_game_id="game_1_1_aa", game_rules=GAME_RULES,
        )
        self.assertEqual(result["source_status"], "missing_game_log")
        self.assertIsNone(result["recorded_game_sha256"])
        self.assertIsNone(result["verification"])

    def test_hash_and_verification_use_the_same_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            game_id, log_path, _ = run_offline_game(tmp, seed=505)
            result = reconcile_attempt_source(
                log_path, expected_game_id=game_id, game_rules=GAME_RULES,
            )
            self.assertEqual(result["source_status"], "recorded")
            self.assertEqual(
                result["recorded_game_sha256"],
                hashlib.sha256(log_path.read_bytes()).hexdigest(),
            )
            self.assertTrue(result["verification"]["complete"])

    def test_incomplete_log_is_hashed_but_not_recoverable(self):
        with tempfile.TemporaryDirectory() as tmp:
            game_id, log_path, _ = run_offline_game(tmp, seed=606)
            rows = load_rows(log_path)
            rewrite(log_path, [r for r in rows if r.get("type") != "outcome"])
            result = reconcile_attempt_source(
                log_path, expected_game_id=game_id, game_rules=GAME_RULES,
            )
            self.assertEqual(result["source_status"], "recorded")
            self.assertIsNotNone(result["recorded_game_sha256"])
            self.assertFalse(result["verification"]["complete"])


if __name__ == "__main__":
    unittest.main()
