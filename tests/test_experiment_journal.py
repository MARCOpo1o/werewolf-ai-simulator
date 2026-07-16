import json
import tempfile
import unittest
from pathlib import Path

from werewolf.experiments.journal import (
    HEALTH_CHECK,
    RECORD_SCHEMA_VERSION,
    SOURCE_MISSING,
    SOURCE_RECORDED,
    TRIAL_COMPLETED,
    TRIAL_FAILED,
    TRIAL_INTERRUPTED,
    TRIAL_STARTED,
    JournalError,
    JournalIntegrityError,
    JournalWriter,
    read_journal,
    replay,
    sanitize_error,
    validate_record,
)

MANIFEST_SHA = "1" * 64
CONTRACT_SHA = "2" * 64
GAME_SHA = "3" * 64


def attempt_fields(trial_id="trial_a", attempt_id="att_1",
                   attempt_number=1, **overrides) -> dict:
    fields = {
        "trial_id": trial_id,
        "attempt_id": attempt_id,
        "attempt_number": attempt_number,
        "trial_index": 0,
        "scheduler_position": 0,
        "condition_id": "a_homogeneous",
        "seed": 5000,
        "repetition": 0,
        "game_id": "game_5000_1_aa",
    }
    fields.update(overrides)
    return fields


def completed_fields(**overrides) -> dict:
    fields = attempt_fields(**{
        k: v for k, v in overrides.items() if k in attempt_fields()
    })
    fields.update({
        "recorded_game_sha256": GAME_SHA,
        "source_status": SOURCE_RECORDED,
        "winner": "village",
        "rounds": 3,
    })
    for key, value in overrides.items():
        if key not in attempt_fields():
            fields[key] = value
    return fields


def make_writer(tmp) -> JournalWriter:
    return JournalWriter(
        Path(tmp) / "trials.jsonl",
        manifest_content_sha256=MANIFEST_SHA,
        execution_contract_sha256=CONTRACT_SHA,
    )


class RecordValidationTests(unittest.TestCase):
    def _envelope(self, record_type: str, fields: dict) -> dict:
        return {
            "record_id": "r1",
            "record_schema_version": RECORD_SCHEMA_VERSION,
            "record_type": record_type,
            "recorded_at": "2026-07-15T00:00:00+00:00",
            "execution_session_id": "s1",
            "manifest_content_sha256": MANIFEST_SHA,
            "execution_contract_sha256": CONTRACT_SHA,
            **fields,
        }

    def test_valid_trial_started(self):
        record = self._envelope(TRIAL_STARTED, attempt_fields())
        self.assertEqual(validate_record(record), [])

    def test_non_trial_records_reject_trial_fields(self):
        record = self._envelope("execution_session_started", {
            "pid": 1, "hostname": "h", "execution_runtime_hash": "x",
            "resume": False, "runnable_trials": 4,
            "trial_id": "",  # fake empty trial field
        })
        errors = validate_record(record)
        self.assertTrue(any("outside its schema" in e for e in errors))

    def test_terminal_records_require_source_consistency(self):
        bad = self._envelope(TRIAL_COMPLETED, completed_fields())
        bad["source_status"] = SOURCE_MISSING  # but sha present
        self.assertTrue(validate_record(bad))
        bad2 = self._envelope(TRIAL_COMPLETED, completed_fields())
        bad2["recorded_game_sha256"] = None  # but status says recorded
        self.assertTrue(validate_record(bad2))
        good = self._envelope(TRIAL_INTERRUPTED, attempt_fields())
        good["recorded_game_sha256"] = None
        good["source_status"] = SOURCE_MISSING
        self.assertEqual(validate_record(good), [])

    def test_health_check_schema(self):
        record = self._envelope(HEALTH_CHECK, {
            "health_fingerprint": "f" * 64,
            "model_alias": "fast",
            "requested_model": "grok-4.3",
            "provider": "xai",
            "effective_generation": {"max_output_tokens": 4096},
            "status": "ready",
            "adjustments": {"generation_dropped": [],
                            "generation_adjusted": []},
            "latency_ms": 120,
            "cost": {"usd": 0.001},
            "cost_completeness": "provider_reported",
            "sanitized_error": None,
        })
        self.assertEqual(validate_record(record), [])
        record["status"] = "sideways"
        self.assertTrue(validate_record(record))

    def test_unknown_record_type_rejected(self):
        record = self._envelope("mystery", {})
        self.assertTrue(validate_record(record))


class JournalWriterTests(unittest.TestCase):
    def test_attempt_lifecycle_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            writer = make_writer(tmp)
            writer.session_started(
                execution_runtime_hash="h", resume=False, runnable_trials=1,
            )
            writer.append(TRIAL_STARTED, attempt_fields())
            writer.append(TRIAL_COMPLETED, completed_fields())
            writer.session_finished(
                completed_trials=1, failed_trials=0, interrupted_trials=0,
            )
            snapshot = read_journal(writer.path)
            self.assertEqual(len(snapshot.records), 4)
            state = replay(snapshot.records)
            self.assertTrue(state.trials["trial_a"].completed)
            self.assertEqual(state.trials["trial_a"].attempt_count, 1)

    def test_terminal_without_start_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            writer = make_writer(tmp)
            with self.assertRaises(JournalError):
                writer.append(TRIAL_COMPLETED, completed_fields())

    def test_double_terminal_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            writer = make_writer(tmp)
            writer.append(TRIAL_STARTED, attempt_fields())
            writer.append(TRIAL_FAILED, {
                **attempt_fields(),
                "recorded_game_sha256": None,
                "source_status": SOURCE_MISSING,
                "sanitized_error": "boom",
            })
            with self.assertRaises(JournalError):
                writer.append(TRIAL_COMPLETED, completed_fields())

    def test_failed_then_completed_counts_one_trial_three_attempts(self):
        with tempfile.TemporaryDirectory() as tmp:
            writer = make_writer(tmp)
            for attempt in (1, 2):
                writer.append(TRIAL_STARTED, attempt_fields(
                    attempt_id=f"att_{attempt}", attempt_number=attempt,
                ))
                writer.append(TRIAL_FAILED, {
                    **attempt_fields(
                        attempt_id=f"att_{attempt}", attempt_number=attempt,
                    ),
                    "recorded_game_sha256": GAME_SHA,
                    "source_status": SOURCE_RECORDED,
                    "sanitized_error": "provider_error",
                })
            writer.append(TRIAL_STARTED, attempt_fields(
                attempt_id="att_3", attempt_number=3,
            ))
            writer.append(TRIAL_COMPLETED, {
                **completed_fields(),
                "attempt_id": "att_3", "attempt_number": 3,
            })
            trial = writer.state.trials["trial_a"]
            self.assertTrue(trial.completed)
            self.assertEqual(trial.attempt_count, 3)
            self.assertEqual(trial.failed_attempts, 2)

    def test_started_after_completion_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            writer = make_writer(tmp)
            writer.append(TRIAL_STARTED, attempt_fields())
            writer.append(TRIAL_COMPLETED, completed_fields())
            with self.assertRaises(JournalError):
                writer.append(TRIAL_STARTED, attempt_fields(
                    attempt_id="att_2", attempt_number=2,
                ))

    def test_wrong_attempt_number_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            writer = make_writer(tmp)
            with self.assertRaises(JournalError):
                writer.append(TRIAL_STARTED, attempt_fields(
                    attempt_number=2,
                ))

    def test_duplicate_attempt_id_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            writer = make_writer(tmp)
            writer.append(TRIAL_STARTED, attempt_fields(trial_id="t1"))
            writer.append(TRIAL_INTERRUPTED, {
                **attempt_fields(trial_id="t1"),
                "recorded_game_sha256": None,
                "source_status": SOURCE_MISSING,
            })
            with self.assertRaises(JournalError):
                writer.append(TRIAL_STARTED, attempt_fields(trial_id="t2"))

    def test_writer_resumes_state_from_disk(self):
        with tempfile.TemporaryDirectory() as tmp:
            writer = make_writer(tmp)
            writer.append(TRIAL_STARTED, attempt_fields())
            resumed = make_writer(tmp)
            self.assertIsNotNone(resumed.state.trials["trial_a"].open_attempt)
            resumed.append(TRIAL_INTERRUPTED, {
                **attempt_fields(),
                "recorded_game_sha256": None,
                "source_status": SOURCE_MISSING,
            })

    def test_writer_rejects_foreign_manifest_journal(self):
        with tempfile.TemporaryDirectory() as tmp:
            writer = make_writer(tmp)
            writer.append(TRIAL_STARTED, attempt_fields())
            with self.assertRaises(JournalIntegrityError):
                JournalWriter(
                    writer.path,
                    manifest_content_sha256="9" * 64,
                    execution_contract_sha256=CONTRACT_SHA,
                )

    def test_torn_tail_is_truncated_and_appendable(self):
        with tempfile.TemporaryDirectory() as tmp:
            writer = make_writer(tmp)
            writer.append(TRIAL_STARTED, attempt_fields())
            with open(writer.path, "a", encoding="utf-8") as f:
                f.write('{"record_id": "torn')  # crash mid-append
            snapshot = read_journal(writer.path)
            self.assertTrue(snapshot.truncated_tail)
            self.assertEqual(len(snapshot.records), 1)
            repaired = make_writer(tmp)
            repaired.append(TRIAL_INTERRUPTED, {
                **attempt_fields(),
                "recorded_game_sha256": None,
                "source_status": SOURCE_MISSING,
            })
            final = read_journal(repaired.path)
            self.assertFalse(final.truncated_tail)
            self.assertEqual(len(final.records), 2)

    def test_mid_file_corruption_is_an_integrity_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            writer = make_writer(tmp)
            writer.append(TRIAL_STARTED, attempt_fields())
            with open(writer.path, "a", encoding="utf-8") as f:
                f.write("not json\n")
                f.write(json.dumps({"also": "not a valid record"}) + "\n")
            with self.assertRaises(JournalIntegrityError):
                read_journal(writer.path)


class SanitizeErrorTests(unittest.TestCase):
    def test_bounds_and_redaction(self):
        text = sanitize_error(
            "boom sk-abcdefghijklmnop secret\nline2 " + "x" * 400
        )
        self.assertNotIn("sk-abcdefghijklmnop", text)
        self.assertIn("[REDACTED]", text)
        self.assertNotIn("\n", text)
        self.assertLessEqual(len(text), 300)


if __name__ == "__main__":
    unittest.main()
