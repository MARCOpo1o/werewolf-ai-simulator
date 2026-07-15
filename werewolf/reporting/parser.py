"""Tolerant, provenance-preserving JSONL game-log parser."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from werewolf.engine.beliefs import inspect_recorded_probability_map
from werewolf.json_safety import (
    as_mapping,
    nonnegative_finite_number,
    nonnegative_int,
)


_TOKEN_FIELDS = (
    "input_tokens", "cached_input_tokens", "output_tokens",
    "reasoning_tokens", "total_tokens",
)


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
                if "role_map" in row and not isinstance(row.get("role_map"), dict):
                    parsed.warnings.append(ParseWarning(
                        "malformed_role_map", "Config role_map is not an object",
                        source_line,
                    ))
                if "role_models" in row and not isinstance(row.get("role_models"), dict):
                    parsed.warnings.append(ParseWarning(
                        "malformed_role_models",
                        "Config role_models is not an object", source_line,
                    ))
            else:
                parsed.warnings.append(ParseWarning(
                    "duplicate_config", "Additional config record ignored", source_line,
                ))
        elif row_type == "event":
            event = row.get("event")
            if isinstance(event, dict):
                parsed.events.append({**event, "source_line": source_line})
                if "payload" in event and not isinstance(event.get("payload"), dict):
                    parsed.warnings.append(ParseWarning(
                        "malformed_event_payload",
                        "Event payload is not an object", source_line,
                    ))
                payload = as_mapping(event.get("payload"))
                if event.get("type") == "belief_snapshot" and payload:
                    if (
                        "valid" in payload
                        and not isinstance(payload.get("valid"), bool)
                    ):
                        parsed.warnings.append(ParseWarning(
                            "invalid_belief_valid_flag",
                            "Belief valid flag is not a JSON boolean",
                            source_line,
                        ))
                    for field_name, allow_none in (
                        ("wolf_probabilities", False),
                        ("estimated_suspicion_of_me", True),
                    ):
                        if field_name not in payload:
                            continue
                        _, probabilities_valid = inspect_recorded_probability_map(
                            payload.get(field_name), allow_none=allow_none,
                        )
                        if not probabilities_valid:
                            parsed.warnings.append(ParseWarning(
                                "invalid_belief_probability",
                                f"Belief {field_name} contains an invalid "
                                "player ID or probability value",
                                source_line,
                            ))
            else:
                parsed.warnings.append(ParseWarning(
                    "malformed_event", "Event record does not contain an object", source_line,
                ))
        elif row_type == "llm_call":
            parsed.llm_calls.append({**row, "source_line": source_line})
            raw_usage = row.get("usage")
            if raw_usage is not None and not isinstance(raw_usage, dict):
                parsed.warnings.append(ParseWarning(
                    "malformed_llm_usage", "LLM usage is not an object", source_line,
                ))
            else:
                usage = as_mapping(raw_usage)
                invalid_tokens = [
                    field for field in _TOKEN_FIELDS
                    if field in usage and usage[field] is not None
                    and nonnegative_int(usage[field]) is None
                ]
                if invalid_tokens:
                    parsed.warnings.append(ParseWarning(
                        "invalid_llm_tokens",
                        "Negative, non-integer, or non-finite token values: "
                        + ", ".join(invalid_tokens),
                        source_line,
                    ))
            raw_cost = row.get("cost")
            if raw_cost is not None and not isinstance(raw_cost, dict):
                parsed.warnings.append(ParseWarning(
                    "malformed_llm_cost", "LLM cost is not an object", source_line,
                ))
            else:
                cost = as_mapping(raw_cost)
                invalid_cost = [
                    field for field in ("usd", "ticks")
                    if field in cost and cost[field] is not None
                    and nonnegative_finite_number(cost[field]) is None
                ]
                if invalid_cost:
                    parsed.warnings.append(ParseWarning(
                        "invalid_llm_cost",
                        "Negative or non-finite cost values: "
                        + ", ".join(invalid_cost),
                        source_line,
                    ))
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
                terminal_tokens = as_mapping(usage.get("tokens"))
                invalid_tokens = [
                    field for field in _TOKEN_FIELDS
                    if field in terminal_tokens
                    and terminal_tokens[field] is not None
                    and nonnegative_int(terminal_tokens[field]) is None
                ]
                if invalid_tokens:
                    parsed.warnings.append(ParseWarning(
                        "invalid_terminal_tokens",
                        "Negative, non-integer, or non-finite terminal tokens: "
                        + ", ".join(invalid_tokens),
                        source_line,
                    ))
                if (
                    usage.get("cost_usd_total") is not None
                    and nonnegative_finite_number(
                        usage.get("cost_usd_total")
                    ) is None
                ):
                    parsed.warnings.append(ParseWarning(
                        "invalid_terminal_cost",
                        "Terminal cost is negative or non-finite", source_line,
                    ))
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
