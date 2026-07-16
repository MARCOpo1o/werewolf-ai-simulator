"""Lifecycle journal: the canonical execution record of an experiment.

`trials.jsonl` is append-only. Every record shares a common envelope
(identity, time, session, and the manifest/execution-contract hashes it
ran under) and is then validated against a record-type-specific schema:
non-trial records never carry fake empty trial fields.

Scheduled-trial state is derived, never stored: a trial that failed
twice and later completed is one completed scheduled trial and three
operational attempts. Legal attempt transitions are exactly

    trial_started -> trial_completed
    trial_started -> trial_failed
    trial_started -> trial_interrupted

The journal tolerates a torn final line (crash mid-append) and nothing
else: earlier corruption is an integrity error, not something to skip.
"""
from __future__ import annotations

import json
import os
import re
import socket
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

RECORD_SCHEMA_VERSION = 1

TRIAL_STARTED = "trial_started"
TRIAL_COMPLETED = "trial_completed"
TRIAL_FAILED = "trial_failed"
TRIAL_INTERRUPTED = "trial_interrupted"
HEALTH_CHECK = "health_check"
SESSION_STARTED = "execution_session_started"
SESSION_FINISHED = "execution_session_finished"
SESSION_ABORTED = "execution_session_aborted"

TERMINAL_ATTEMPT_TYPES = frozenset(
    {TRIAL_COMPLETED, TRIAL_FAILED, TRIAL_INTERRUPTED}
)
ATTEMPT_TYPES = frozenset({TRIAL_STARTED}) | TERMINAL_ATTEMPT_TYPES

SOURCE_RECORDED = "recorded"
SOURCE_MISSING = "missing_game_log"

_ENVELOPE_FIELDS = (
    "record_id", "record_schema_version", "record_type", "recorded_at",
    "execution_session_id", "manifest_content_sha256",
    "execution_contract_sha256",
)

_ATTEMPT_FIELDS = (
    "trial_id", "attempt_id", "attempt_number", "trial_index",
    "scheduler_position", "condition_id", "seed", "repetition", "game_id",
)

_HEX64 = re.compile(r"^[0-9a-f]{64}$")

# record_type -> (required fields, optional fields) beyond the envelope.
_RECORD_FIELDS = {
    TRIAL_STARTED: (frozenset(_ATTEMPT_FIELDS), frozenset()),
    TRIAL_COMPLETED: (
        frozenset(_ATTEMPT_FIELDS)
        | {"recorded_game_sha256", "source_status", "winner", "rounds"},
        frozenset({"recovered", "verifier"}),
    ),
    TRIAL_FAILED: (
        frozenset(_ATTEMPT_FIELDS)
        | {"recorded_game_sha256", "source_status", "sanitized_error"},
        frozenset({"error_category"}),
    ),
    TRIAL_INTERRUPTED: (
        frozenset(_ATTEMPT_FIELDS)
        | {"recorded_game_sha256", "source_status"},
        frozenset({"reason", "verifier"}),
    ),
    HEALTH_CHECK: (
        frozenset({
            "health_fingerprint", "model_alias", "requested_model",
            "provider", "effective_generation", "status", "adjustments",
            "latency_ms", "cost", "cost_completeness", "sanitized_error",
        }),
        frozenset({"resolved_model", "adjustment_fingerprint"}),
    ),
    SESSION_STARTED: (
        frozenset({
            "pid", "hostname", "execution_runtime_hash", "resume",
            "runnable_trials",
        }),
        frozenset({"retry_failed", "allow_adjusted_health"}),
    ),
    SESSION_FINISHED: (
        frozenset({
            "completed_trials", "failed_trials", "interrupted_trials",
        }),
        frozenset(),
    ),
    SESSION_ABORTED: (frozenset({"reason"}), frozenset()),
}

_HEALTH_STATUSES = frozenset(
    {"ready", "adjusted", "failed", "missing_key", "provider_unavailable"}
)


class JournalError(ValueError):
    pass


class JournalIntegrityError(JournalError):
    """The journal contains corruption before its final line."""


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_record_id() -> str:
    return uuid.uuid4().hex


def sanitize_error(exc_or_text, limit: int = 300) -> str:
    """Bounded, single-line error description. Never include prompts,
    responses, or credentials — callers pass exception summaries only."""
    text = str(exc_or_text).replace("\n", " ").replace("\r", " ").strip()
    text = re.sub(r"\b(sk-[A-Za-z0-9_\-]{8,}|xai-[A-Za-z0-9_\-]{8,}|"
                  r"AIza[A-Za-z0-9_\-]{10,})", "[REDACTED]", text)
    return text[:limit] or (
        exc_or_text.__class__.__name__
        if isinstance(exc_or_text, BaseException) else "unknown_error"
    )


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def validate_record(record) -> list:
    """Envelope plus record-type-specific validation. Returns errors."""
    if not isinstance(record, dict):
        return ["record must be a JSON object"]
    errors = []
    for name in _ENVELOPE_FIELDS:
        if name not in record:
            errors.append(f"missing envelope field: {name}")
    if record.get("record_schema_version") != RECORD_SCHEMA_VERSION:
        errors.append(
            f"record_schema_version must be {RECORD_SCHEMA_VERSION}"
        )
    record_type = record.get("record_type")
    if record_type not in _RECORD_FIELDS:
        errors.append(f"unknown record_type: {record_type!r}")
        return errors
    for name in ("record_id", "recorded_at", "execution_session_id"):
        if name in record and not isinstance(record[name], str):
            errors.append(f"{name} must be a string")
    for name in ("manifest_content_sha256", "execution_contract_sha256"):
        value = record.get(name)
        if name in record and (
            not isinstance(value, str) or not _HEX64.match(value)
        ):
            errors.append(f"{name} must be a 64-char hex digest")

    required, optional = _RECORD_FIELDS[record_type]
    body = set(record) - set(_ENVELOPE_FIELDS)
    missing = required - body
    if missing:
        errors.append(
            f"{record_type} missing fields: {sorted(missing)}"
        )
    unknown = body - required - optional
    if unknown:
        # Fake empty trial fields on non-trial records land here too.
        errors.append(
            f"{record_type} has fields outside its schema: {sorted(unknown)}"
        )

    if record_type in ATTEMPT_TYPES and not missing:
        for name in ("attempt_number", "trial_index", "scheduler_position",
                     "seed", "repetition"):
            if not _is_int(record[name]):
                errors.append(f"{name} must be an integer")
        if _is_int(record.get("attempt_number")) \
                and record["attempt_number"] < 1:
            errors.append("attempt_number must be >= 1")
        for name in ("trial_id", "attempt_id", "condition_id", "game_id"):
            if not isinstance(record[name], str) or not record[name]:
                errors.append(f"{name} must be a non-empty string")
    if record_type in TERMINAL_ATTEMPT_TYPES and not missing:
        sha = record["recorded_game_sha256"]
        status = record["source_status"]
        if status not in (SOURCE_RECORDED, SOURCE_MISSING):
            errors.append(
                f"source_status must be {SOURCE_RECORDED!r} or "
                f"{SOURCE_MISSING!r}"
            )
        if status == SOURCE_MISSING and sha is not None:
            errors.append(
                "recorded_game_sha256 must be null when the game log "
                "is missing"
            )
        if status == SOURCE_RECORDED and (
            not isinstance(sha, str) or not _HEX64.match(sha)
        ):
            errors.append(
                "recorded_game_sha256 must be a 64-char hex digest "
                "when a game log was recorded"
            )
    if record_type == HEALTH_CHECK and not missing:
        if record["status"] not in _HEALTH_STATUSES:
            errors.append(
                f"health status must be one of {sorted(_HEALTH_STATUSES)}"
            )
        if not isinstance(record["adjustments"], dict):
            errors.append("adjustments must be an object")
    if record_type == SESSION_STARTED and not missing:
        if not _is_int(record["pid"]):
            errors.append("pid must be an integer")
        if not isinstance(record["resume"], bool):
            errors.append("resume must be a boolean")
    if record_type == SESSION_FINISHED and not missing:
        for name in ("completed_trials", "failed_trials",
                     "interrupted_trials"):
            if not _is_int(record[name]):
                errors.append(f"{name} must be an integer")
    return errors


# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------

@dataclass
class JournalSnapshot:
    """Complete-line view of a journal (crash-tolerant tail handling)."""
    records: list
    byte_length: int          # bytes covered by complete parsed lines
    truncated_tail: bool      # a torn final line was ignored

    @property
    def last_record_id(self) -> Optional[str]:
        return self.records[-1]["record_id"] if self.records else None


def read_journal(path) -> JournalSnapshot:
    path = Path(path)
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        return JournalSnapshot(records=[], byte_length=0, truncated_tail=False)
    records = []
    offset = 0
    truncated = False
    while offset < len(data):
        newline = data.find(b"\n", offset)
        if newline == -1:
            truncated = True  # torn tail: crash mid-append
            break
        line = data[offset:newline]
        if line.strip():
            try:
                record = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, ValueError) as exc:
                raise JournalIntegrityError(
                    f"Corrupt journal line at byte {offset} of {path}: {exc}"
                )
            errors = validate_record(record)
            if errors:
                raise JournalIntegrityError(
                    f"Invalid journal record at byte {offset} of {path}: "
                    + "; ".join(errors)
                )
            records.append(record)
        offset = newline + 1
    return JournalSnapshot(
        records=records, byte_length=offset, truncated_tail=truncated,
    )


# --------------------------------------------------------------------------
# Replay: derive attempt/trial state
# --------------------------------------------------------------------------

@dataclass
class TrialState:
    trial_id: str
    attempts: list = field(default_factory=list)

    @property
    def attempt_count(self) -> int:
        return len(self.attempts)

    @property
    def completed(self) -> bool:
        return any(
            a["terminal"] and a["terminal"]["record_type"] == TRIAL_COMPLETED
            for a in self.attempts
        )

    @property
    def open_attempt(self) -> Optional[dict]:
        for attempt in self.attempts:
            if attempt["terminal"] is None:
                return attempt
        return None

    @property
    def failed_attempts(self) -> int:
        return sum(
            1 for a in self.attempts
            if a["terminal"]
            and a["terminal"]["record_type"] in (TRIAL_FAILED,
                                                 TRIAL_INTERRUPTED)
        )


@dataclass
class ReplayState:
    trials: dict = field(default_factory=dict)
    sessions: dict = field(default_factory=dict)
    health_checks: list = field(default_factory=list)

    def trial(self, trial_id: str) -> TrialState:
        if trial_id not in self.trials:
            self.trials[trial_id] = TrialState(trial_id=trial_id)
        return self.trials[trial_id]

    def apply(self, record: dict) -> None:
        record_type = record["record_type"]
        if record_type == SESSION_STARTED:
            self.sessions[record["execution_session_id"]] = {
                "started": record, "ended": None,
            }
        elif record_type in (SESSION_FINISHED, SESSION_ABORTED):
            session = self.sessions.get(record["execution_session_id"])
            if session is None:
                raise JournalError(
                    f"{record_type} for unknown session "
                    f"{record['execution_session_id']}"
                )
            session["ended"] = record
        elif record_type == HEALTH_CHECK:
            self.health_checks.append(record)
        elif record_type == TRIAL_STARTED:
            trial = self.trial(record["trial_id"])
            if trial.completed:
                raise JournalError(
                    f"trial_started for already-completed trial "
                    f"{record['trial_id']}"
                )
            if trial.open_attempt is not None:
                raise JournalError(
                    f"trial_started while attempt "
                    f"{trial.open_attempt['started']['attempt_id']} of "
                    f"trial {record['trial_id']} is still open"
                )
            if record["attempt_number"] != trial.attempt_count + 1:
                raise JournalError(
                    f"attempt_number {record['attempt_number']} for trial "
                    f"{record['trial_id']} (expected "
                    f"{trial.attempt_count + 1})"
                )
            if any(a["started"]["attempt_id"] == record["attempt_id"]
                   for t in self.trials.values() for a in t.attempts):
                raise JournalError(
                    f"duplicate attempt_id {record['attempt_id']}"
                )
            trial.attempts.append({"started": record, "terminal": None})
        elif record_type in TERMINAL_ATTEMPT_TYPES:
            trial = self.trials.get(record["trial_id"])
            open_attempt = trial.open_attempt if trial else None
            if (
                open_attempt is None
                or open_attempt["started"]["attempt_id"]
                != record["attempt_id"]
            ):
                raise JournalError(
                    f"{record_type} for attempt {record['attempt_id']} of "
                    f"trial {record['trial_id']} without a matching open "
                    "trial_started"
                )
            if record["game_id"] != open_attempt["started"]["game_id"]:
                raise JournalError(
                    f"{record_type} game_id does not match trial_started "
                    f"for attempt {record['attempt_id']}"
                )
            open_attempt["terminal"] = record
        else:  # pragma: no cover - validate_record already rejects
            raise JournalError(f"unknown record_type {record_type}")


def replay(records) -> ReplayState:
    state = ReplayState()
    for record in records:
        state.apply(record)
    return state


# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------

class JournalWriter:
    """Append-only writer that validates schemas and attempt transitions
    against the already-persisted journal before every append."""

    def __init__(
        self,
        path,
        *,
        manifest_content_sha256: str,
        execution_contract_sha256: str,
        execution_session_id: Optional[str] = None,
    ):
        self.path = Path(path)
        self.manifest_content_sha256 = manifest_content_sha256
        self.execution_contract_sha256 = execution_contract_sha256
        self.execution_session_id = execution_session_id or uuid.uuid4().hex
        snapshot = read_journal(self.path)
        for record in snapshot.records:
            if record["manifest_content_sha256"] != manifest_content_sha256:
                raise JournalIntegrityError(
                    "Journal records were written under a different "
                    "manifest_content_sha256; refusing to append."
                )
        self._state = replay(snapshot.records)
        if snapshot.truncated_tail:
            # A crash tore the final append mid-line. The record was never
            # durable, so truncating back to the last complete line loses
            # nothing and keeps the journal parseable.
            with open(self.path, "r+b") as f:
                f.truncate(snapshot.byte_length)

    @property
    def state(self) -> ReplayState:
        return self._state

    def append(self, record_type: str, fields: dict) -> dict:
        record = {
            "record_id": new_record_id(),
            "record_schema_version": RECORD_SCHEMA_VERSION,
            "record_type": record_type,
            "recorded_at": utc_now_iso(),
            "execution_session_id": self.execution_session_id,
            "manifest_content_sha256": self.manifest_content_sha256,
            "execution_contract_sha256": self.execution_contract_sha256,
            **fields,
        }
        errors = validate_record(record)
        if errors:
            raise JournalError(
                f"Refusing invalid {record_type} record: " + "; ".join(errors)
            )
        # Transition check against a copy-free replay: apply raises on
        # illegal transitions before anything is persisted.
        self._state.apply(record)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, sort_keys=True) + "\n"
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())
        return record

    # Convenience appenders -------------------------------------------------

    def session_started(self, *, execution_runtime_hash: str, resume: bool,
                        runnable_trials: int, **optional) -> dict:
        return self.append(SESSION_STARTED, {
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "execution_runtime_hash": execution_runtime_hash,
            "resume": resume,
            "runnable_trials": runnable_trials,
            **optional,
        })

    def session_finished(self, *, completed_trials: int, failed_trials: int,
                         interrupted_trials: int) -> dict:
        return self.append(SESSION_FINISHED, {
            "completed_trials": completed_trials,
            "failed_trials": failed_trials,
            "interrupted_trials": interrupted_trials,
        })

    def session_aborted(self, reason: str) -> dict:
        return self.append(SESSION_ABORTED, {
            "reason": sanitize_error(reason),
        })
