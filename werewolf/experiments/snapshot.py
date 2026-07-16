"""Source snapshots for aggregate analysis.

Summarization must describe exactly the evidence it read. The lifecycle
snapshot hashes the byte prefix of trials.jsonl covered by complete
records; each game source is read exactly ONCE, hashed from those bytes,
and parsed from those same bytes — never hashed and then reopened.

Every terminal attempt's current source hash is compared against the
`recorded_game_sha256` journaled when the attempt ended:

- match                 -> "verified"
- mismatch / appeared   -> "source_modified_after_completion"
- file gone             -> "missing_game_log"

Drifted or missing evidence is analytically ineligible; for failed and
interrupted attempts the same check keeps operational-cost totals from
silently changing — such evidence is excluded from authoritative totals
and reported as incomplete.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from werewolf.experiments.canonical import sha256_bytes
from werewolf.experiments.journal import (
    TERMINAL_ATTEMPT_TYPES,
    JournalSnapshot,
    parse_journal_bytes,
    replay,
)

SOURCE_VERIFIED = "verified"
SOURCE_MODIFIED = "source_modified_after_completion"
SOURCE_MISSING = "missing_game_log"


@dataclass
class LifecycleSnapshot:
    records: list
    journal_byte_length: int
    lifecycle_record_count: int
    lifecycle_snapshot_sha256: str
    last_lifecycle_record_id: Optional[str]

    def meta(self) -> dict:
        return {
            "last_lifecycle_record_id": self.last_lifecycle_record_id,
            "journal_byte_length": self.journal_byte_length,
            "lifecycle_record_count": self.lifecycle_record_count,
            "lifecycle_snapshot_sha256": self.lifecycle_snapshot_sha256,
        }


def capture_lifecycle_snapshot(journal_path) -> LifecycleSnapshot:
    """Capture only complete journal lines (a torn tail is invisible)."""
    path = Path(journal_path)
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        data = b""
    parsed: JournalSnapshot = parse_journal_bytes(data, source=str(path))
    prefix = data[:parsed.byte_length]
    return LifecycleSnapshot(
        records=parsed.records,
        journal_byte_length=parsed.byte_length,
        lifecycle_record_count=len(parsed.records),
        lifecycle_snapshot_sha256=sha256_bytes(prefix),
        last_lifecycle_record_id=parsed.last_record_id,
    )


@dataclass
class GameSource:
    """One terminal attempt's evidence, captured with a single read."""
    trial_id: str
    attempt_id: str
    attempt_number: int
    record_type: str
    condition_id: str
    seed: int
    repetition: int
    game_id: str
    recorded_game_sha256: Optional[str]
    observed_game_sha256: Optional[str]
    source_status: str
    terminal_record: dict = field(repr=False)
    data: Optional[bytes] = field(default=None, repr=False)
    rows: Optional[list] = field(default=None, repr=False)

    @property
    def verified(self) -> bool:
        return self.source_status == SOURCE_VERIFIED

    def identity(self) -> dict:
        """Entry for summary_input_sha256's sorted_game_sources."""
        return {
            "trial_id": self.trial_id,
            "attempt_id": self.attempt_id,
            "game_id": self.game_id,
            "recorded_game_sha256": self.recorded_game_sha256,
            "observed_game_sha256": self.observed_game_sha256,
            "source_status": self.source_status,
        }


def _parse_game_rows(data: bytes) -> list:
    rows = []
    for line in data.split(b"\n"):
        if not line.strip():
            continue
        try:
            row = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def capture_game_sources(games_dir, lifecycle: LifecycleSnapshot) -> list:
    """One GameSource per terminal attempt, ordered by (trial_id,
    attempt_id). Reads each game log exactly once."""
    state = replay(lifecycle.records)
    sources = []
    for trial in state.trials.values():
        for attempt in trial.attempts:
            terminal = attempt["terminal"]
            if terminal is None or (
                terminal["record_type"] not in TERMINAL_ATTEMPT_TYPES
            ):
                continue
            recorded = terminal["recorded_game_sha256"]
            path = Path(games_dir) / f"{terminal['game_id']}.jsonl"
            try:
                data = path.read_bytes()
            except FileNotFoundError:
                data = None
            if data is None:
                observed = None
                status = SOURCE_MISSING
                rows = None
            else:
                observed = sha256_bytes(data)
                if recorded is None or observed != recorded:
                    # Includes logs that appeared after being journaled
                    # missing: evidence changed after completion.
                    status = SOURCE_MODIFIED
                    rows = None
                else:
                    status = SOURCE_VERIFIED
                    rows = _parse_game_rows(data)
            if data is None and recorded is None:
                # Journaled missing and still missing: consistent, but
                # there is simply no evidence to analyze.
                status = SOURCE_MISSING
            sources.append(GameSource(
                trial_id=terminal["trial_id"],
                attempt_id=terminal["attempt_id"],
                attempt_number=terminal["attempt_number"],
                record_type=terminal["record_type"],
                condition_id=terminal["condition_id"],
                seed=terminal["seed"],
                repetition=terminal["repetition"],
                game_id=terminal["game_id"],
                recorded_game_sha256=recorded,
                observed_game_sha256=observed,
                source_status=status,
                terminal_record=terminal,
                data=data if status == SOURCE_VERIFIED else None,
                rows=rows,
            ))
    sources.sort(key=lambda s: (s.trial_id, s.attempt_id))
    return sources
