"""Cached full-report loading over the canonical game repository."""
from __future__ import annotations

import copy
import json
from typing import Optional

from werewolf.reporting.builder import REPORT_SCHEMA_VERSION, build_full_report_from_file
from werewolf.reporting.repository import GameRepository, atomic_json_write


def load_full_report(
    repository: GameRepository,
    game_id: str,
    *,
    active_game_id: Optional[str] = None,
) -> Optional[dict]:
    entry = repository.refresh_game(game_id)
    if entry is None:
        return None
    report_path = repository.report_path(game_id)
    report = None
    try:
        with open(report_path, encoding="utf-8") as handle:
            candidate = json.load(handle)
        source = candidate.get("source") or {}
        if (
            candidate.get("report_schema_version") == REPORT_SCHEMA_VERSION
            and source.get("size_bytes") == entry.get("source_size")
            and source.get("mtime_ns") == entry.get("source_mtime_ns")
        ):
            report = candidate
    except (OSError, json.JSONDecodeError, AttributeError):
        pass

    if report is None:
        report = build_full_report_from_file(
            repository.log_path(game_id), metadata=entry,
        )
        atomic_json_write(report_path, report)
    repository.update_from_report(game_id, report)

    response = copy.deepcopy(report)
    response["overview"]["display_status"] = (
        "active" if active_game_id == game_id
        else response["overview"]["completion_status"]
    )
    return response


__all__ = ["load_full_report"]
