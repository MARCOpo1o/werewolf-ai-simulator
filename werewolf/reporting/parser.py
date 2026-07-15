"""Tolerant, provenance-preserving JSONL game-log parser."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class ParseWarning:
    code: str
    message: str
    source_line: Optional[int] = None

    def to_json_dict(self) -> dict:
        return {
            "code": self.code,
            "message": self.message,
            "source_line": self.source_line,
        }


@dataclass
class ParsedGameLog:
    path: Path
    sha256: str
    source_size: int
    source_mtime_ns: int
    rows: list[dict] = field(default_factory=list)
    config: Optional[dict] = None
    events: list[dict] = field(default_factory=list)
    llm_calls: list[dict] = field(default_factory=list)
    usage_summary: Optional[dict] = None
    outcome: Optional[dict] = None
    warnings: list[ParseWarning] = field(default_factory=list)
    record_counts: dict[str, int] = field(default_factory=dict)


def parse_game_log(path: str | Path) -> ParsedGameLog:
    path = Path(path)
    data = path.read_bytes()
    stat = path.stat()
    parsed = ParsedGameLog(
        path=path,
        sha256=hashlib.sha256(data).hexdigest(),
        source_size=len(data),
        source_mtime_ns=stat.st_mtime_ns,
    )
    counts: dict[str, int] = {}
    for source_line, raw in enumerate(data.splitlines(), 1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            parsed.warnings.append(ParseWarning(
                "malformed_json_line", f"Invalid JSON: {exc}", source_line,
            ))
            continue
        if not isinstance(row, dict):
            parsed.warnings.append(ParseWarning(
                "non_object_record", "JSONL record is not an object", source_line,
            ))
            continue
        row_type = row.get("type")
        if not isinstance(row_type, str):
            row_type = "unknown"
            parsed.warnings.append(ParseWarning(
                "missing_record_type", "Record has no string type", source_line,
            ))
        counts[row_type] = counts.get(row_type, 0) + 1
        wrapped = {"source_line": source_line, "record": row}
        parsed.rows.append(wrapped)

        if row_type == "config":
            if parsed.config is None:
                parsed.config = row
            else:
                parsed.warnings.append(ParseWarning(
                    "duplicate_config", "Additional config record ignored", source_line,
                ))
        elif row_type == "event":
            event = row.get("event")
            if isinstance(event, dict):
                parsed.events.append({**event, "source_line": source_line})
            else:
                parsed.warnings.append(ParseWarning(
                    "malformed_event", "Event record does not contain an object", source_line,
                ))
        elif row_type == "llm_call":
            parsed.llm_calls.append({**row, "source_line": source_line})
        elif row_type == "usage_summary":
            usage = row.get("usage")
            if isinstance(usage, dict):
                if parsed.usage_summary is not None:
                    parsed.warnings.append(ParseWarning(
                        "duplicate_usage_summary",
                        "Latest usage summary replaces an earlier summary",
                        source_line,
                    ))
                parsed.usage_summary = usage
            else:
                parsed.warnings.append(ParseWarning(
                    "malformed_usage_summary", "Usage summary is not an object", source_line,
                ))
        elif row_type == "outcome":
            if parsed.outcome is not None:
                parsed.warnings.append(ParseWarning(
                    "duplicate_outcome", "Latest outcome replaces an earlier outcome",
                    source_line,
                ))
            parsed.outcome = row

    parsed.record_counts = counts
    if parsed.config is None:
        parsed.warnings.append(ParseWarning(
            "missing_config", "No canonical config record was found",
        ))
    else:
        configured_id = parsed.config.get("game_id")
        if configured_id and configured_id != path.stem:
            parsed.warnings.append(ParseWarning(
                "game_id_mismatch",
                (
                    f"Config game_id {configured_id!r} does not match "
                    f"canonical filename game_id {path.stem!r}"
                ),
            ))
    return parsed


__all__ = ["ParsedGameLog", "ParseWarning", "parse_game_log"]
