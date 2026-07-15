"""Versioned full forensic-report construction for one parsed game log."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Optional

from werewolf.evaluation.validity import classify_game
from werewolf.reporting.analysis import (
    build_belief_analysis,
    build_decision_analysis,
    build_manipulation_signals,
)
from werewolf.reporting.parser import ParsedGameLog, parse_game_log
from werewolf.reporting.usage import compare_terminal_summary, compute_usage


REPORT_SCHEMA_VERSION = 1
ANALYSIS_ELIGIBILITY_POLICY_VERSION = 1
_AGENT_EVENT_TYPES = {
    "thought", "message", "vote", "belief_snapshot", "divine_result",
}


def _observed_models(calls: list[dict]) -> dict:
    observed = defaultdict(set)
    for call in calls:
        role = call.get("player_role") or "unknown"
        if call.get("resolved_model"):
            observed[role].add(call["resolved_model"])
    return {key: sorted(value) for key, value in observed.items()}


def _players(config: dict) -> list[dict]:
    players = []
    for raw_id, info in (config.get("role_map") or {}).items():
        if not isinstance(info, dict):
            continue
        try:
            player_id = int(raw_id)
        except (TypeError, ValueError):
            continue
        players.append({
            "id": player_id,
            "role": info.get("role"),
            "team": info.get("team"),
        })
    return sorted(players, key=lambda player: player["id"])


def _timeline(parsed: ParsedGameLog, calls_by_id: dict[str, list[dict]]) -> list[dict]:
    timeline = []
    for event in parsed.events:
        item = dict(event)
        numeric_id = item.get("id")
        event_id = item.get("event_id")
        if not event_id and isinstance(numeric_id, int):
            event_id = f"evt_{numeric_id:06d}"
            event_id_source = "derived_from_numeric_id"
        else:
            event_id_source = "persisted" if event_id else "unavailable"
        source_call_id = item.get("source_call_id")
        if source_call_id and source_call_id in calls_by_id:
            link_quality = "exact"
        else:
            link_quality = "unavailable"
        item.update({
            "event_id": event_id,
            "event_id_source": event_id_source,
            "source_call_id": source_call_id,
            "discussion_cycle": item.get("discussion_cycle"),
            "link_quality": link_quality,
        })
        timeline.append(item)
    return timeline


def build_full_report(
    parsed: ParsedGameLog,
    *,
    metadata: Optional[dict] = None,
    active_game_id: Optional[str] = None,
) -> dict:
    metadata = metadata or {}
    config = parsed.config or {}
    outcome = parsed.outcome
    game_id = config.get("game_id") or parsed.path.stem
    completion = "completed" if outcome else "incomplete"
    display_status = "active" if active_game_id == game_id else completion

    computed_usage = compute_usage(parsed.llm_calls)
    terminal_check = compare_terminal_summary(
        computed_usage, parsed.usage_summary,
    )
    warnings = [warning.to_json_dict() for warning in parsed.warnings]
    if outcome and parsed.usage_summary is None:
        warnings.append({
            "code": "missing_terminal_usage_summary",
            "message": "Completed log has no terminal usage summary",
            "source_line": None,
        })
    if terminal_check["status"] == "mismatched":
        warnings.append({
            "code": "terminal_usage_mismatch",
            "message": "Terminal usage summary differs from canonical call rows",
            "source_line": None,
        })

    calls_by_id = defaultdict(list)
    for call in parsed.llm_calls:
        if call.get("call_id"):
            calls_by_id[call["call_id"]].append(call)
    timeline = _timeline(parsed, calls_by_id)

    event_schema = config.get("event_schema_version")
    missing_strategic_evidence = []
    for event in timeline:
        if event.get("type") not in _AGENT_EVENT_TYPES:
            continue
        source_call_id = event.get("source_call_id")
        if source_call_id and source_call_id not in calls_by_id:
            missing_strategic_evidence.append(event.get("event_id"))
        elif event_schema and event_schema >= 2 and not source_call_id:
            missing_strategic_evidence.append(event.get("event_id"))

    has_core = bool(parsed.config or parsed.events or parsed.llm_calls or outcome)
    integrity = "corrupt" if not has_core else "warnings" if warnings else "clean"
    exclusion_reasons = []
    if integrity == "corrupt":
        eligibility = "ineligible"
        exclusion_reasons.append("unrecoverable_log")
    elif missing_strategic_evidence:
        eligibility = "ineligible"
        exclusion_reasons.append("missing_strategic_call_evidence")
    elif completion == "incomplete":
        eligibility = "limited"
        exclusion_reasons.append("game_incomplete")
    elif not parsed.config or not event_schema:
        eligibility = "limited"
        exclusion_reasons.append("legacy_provenance_unavailable")
    else:
        eligibility = "eligible"

    validity_rows = []
    for wrapped in parsed.rows:
        row = wrapped["record"]
        if row.get("type") == "event" and not isinstance(row.get("event"), dict):
            continue
        validity_rows.append(row)
    validity = classify_game(validity_rows) if validity_rows else {
        "clean": False, "violations": {"unrecoverable_log": 1},
        "policy_version": None,
    }
    validity["provisional"] = completion != "completed"
    if (
        not validity["clean"] and integrity != "corrupt"
        and completion == "completed"
    ):
        eligibility = "ineligible"
        exclusion_reasons.extend(
            f"validity:{name}" for name in validity["violations"]
        )

    if not parsed.llm_calls:
        usage_reliability = "unavailable"
    elif terminal_check["status"] == "mismatched":
        usage_reliability = "inconsistent"
    elif terminal_check["status"] == "missing":
        usage_reliability = "partial"
    else:
        usage_reliability = "reliable"

    players = _players(config)
    role_map = {str(player["id"]): {
        "role": player["role"], "team": player["team"],
    } for player in players}
    overview = {
        "game_id": game_id,
        "completion_status": completion,
        "display_status": display_status,
        "integrity_status": integrity,
        "analysis_eligibility": eligibility,
        "analysis_exclusion_reasons": exclusion_reasons,
        "usage_reliability": usage_reliability,
        "winner": outcome.get("winner") if outcome else None,
        "rounds": outcome.get("rounds") if outcome else None,
        "remaining": outcome.get("remaining") if outcome else None,
        "seed": config.get("seed"),
        "n_players": config.get("n_players"),
        "n_wolves": config.get("n_wolves"),
        "n_seers": config.get("n_seers"),
        "role_assignment": role_map,
        "requested_models": config.get("role_models") or {
            "all": {
                "alias": config.get("model_alias"),
                "requested_model": config.get("model"),
            }
        },
        "observed_resolved_models": _observed_models(parsed.llm_calls),
        "generation": {
            "requested": config.get("requested_generation_config"),
            "effective": config.get("generation_config"),
            "reasoning_override": config.get("requested_reasoning_override"),
        },
        "discussion_cycles": config.get("discussion_cycles"),
        "belief_snapshots": config.get("belief_snapshots"),
        "limits": config.get("limits"),
        "validity": validity,
        "usage": computed_usage,
    }
    beliefs = build_belief_analysis(config, timeline)
    decisions = build_decision_analysis(timeline, parsed.llm_calls)
    manipulation = build_manipulation_signals(config, timeline, beliefs)
    return {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "source": {
            "log_name": parsed.path.name,
            "sha256": parsed.sha256,
            "size_bytes": parsed.source_size,
            "mtime_ns": parsed.source_mtime_ns,
            "created_at": metadata.get("created_at") or config.get("created_at"),
            "created_at_source": metadata.get("created_at_source"),
            "record_counts": parsed.record_counts,
            "warnings": warnings,
        },
        "overview": overview,
        "players": players,
        "timeline": timeline,
        "beliefs": beliefs,
        "decisions": {
            **decisions,
            "missing_strategic_evidence": missing_strategic_evidence,
        },
        "manipulation_signals": manipulation,
        "usage": {
            **computed_usage,
            "terminal_consistency": terminal_check,
            "reliability": usage_reliability,
        },
        "reproducibility": {
            "code_commit": config.get("code_commit"),
            "prompt_version": config.get("prompt_version"),
            "log_schema_version": config.get("log_schema_version"),
            "event_schema_version": config.get("event_schema_version"),
            "belief_schema_version": config.get("belief_schema_version"),
            "validity_policy_version": validity.get("policy_version"),
            "runtime": config.get("runtime"),
            "report_schema_version": REPORT_SCHEMA_VERSION,
            "analysis_eligibility_policy_version": (
                ANALYSIS_ELIGIBILITY_POLICY_VERSION
            ),
        },
        "links": {
            "raw": f"/api/games/{game_id}/raw",
            "report": f"/games/{game_id}",
        },
    }


def build_full_report_from_file(
    path: str | Path, *, metadata: Optional[dict] = None,
    active_game_id: Optional[str] = None,
) -> dict:
    return build_full_report(
        parse_game_log(path), metadata=metadata, active_game_id=active_game_id,
    )


__all__ = [
    "ANALYSIS_ELIGIBILITY_POLICY_VERSION", "REPORT_SCHEMA_VERSION",
    "build_full_report", "build_full_report_from_file",
]
