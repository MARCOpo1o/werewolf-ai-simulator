"""Explicit allowlisted public projection for spoiler-safe reports.

This is spoiler protection for the trusted local app, not authorization.
"""
from __future__ import annotations


_OVERVIEW_FIELDS = (
    "game_id", "completion_status", "display_status", "integrity_status",
    "analysis_eligibility", "analysis_exclusion_reasons", "usage_reliability",
    "winner", "rounds", "remaining", "seed", "n_players", "n_wolves",
    "n_seers", "requested_models", "generation", "discussion_cycles",
    "belief_snapshots", "limits", "validity",
)
_USAGE_FIELDS = (
    "attempts", "decision_groups", "api_failures", "parse_failures",
    "validation_failures", "retries", "fallbacks", "recovered_parses",
    "errors_by_category", "tokens", "token_fields_missing", "known_cost_usd",
    "calls_with_known_cost", "calls_without_known_cost", "cost_completeness",
    "cost_sources", "by_cost_source", "by_phase", "by_required_action",
    "terminal_consistency", "reliability",
)
_EVENT_FIELDS = (
    "id", "event_id", "event_id_source", "t", "round", "phase", "type",
    "channel", "speaker_id", "discussion_cycle", "source_line", "payload",
)
_REPRO_FIELDS = (
    "code_commit", "prompt_version", "log_schema_version",
    "event_schema_version", "belief_schema_version", "validity_policy_version",
    "runtime", "report_schema_version", "report_build_version",
    "analysis_eligibility_policy_version",
)


def _allow(source: dict, fields: tuple[str, ...]) -> dict:
    return {field: source.get(field) for field in fields if field in source}


def build_public_report(full_report: dict) -> dict:
    overview = _allow(full_report.get("overview") or {}, _OVERVIEW_FIELDS)
    usage = _allow(full_report.get("usage") or {}, _USAGE_FIELDS)
    overview["usage"] = _allow(usage, tuple(
        field for field in _USAGE_FIELDS
        if field not in {"terminal_consistency", "reliability"}
    ))

    public_timeline = []
    for event in full_report.get("timeline") or []:
        if event.get("channel") != "public":
            continue
        public_timeline.append(_allow(event, _EVENT_FIELDS))
    public_decisions = [
        _allow(event, _EVENT_FIELDS)
        for event in (full_report.get("decisions") or {}).get("decision_events", [])
        if event.get("channel") == "public"
    ]
    source = full_report.get("source") or {}
    return {
        "report_schema_version": full_report.get("report_schema_version"),
        "report_build_version": full_report.get("report_build_version"),
        "privacy": {
            "include_private": False,
            "spoiler_protection_only": True,
        },
        "source": _allow(source, (
            "log_name", "sha256", "size_bytes", "mtime_ns", "created_at",
            "created_at_source", "record_counts", "warnings",
        )),
        "overview": overview,
        "players": [
            {"id": player.get("id")}
            for player in full_report.get("players") or []
        ],
        "timeline": public_timeline,
        "beliefs": {
            "available": False, "reason": "private_data_not_requested",
        },
        "decisions": {
            "available": bool(public_decisions),
            "decision_events": public_decisions,
        },
        "manipulation_signals": {
            "available": False, "reason": "private_data_not_requested",
        },
        "usage": usage,
        "reproducibility": _allow(
            full_report.get("reproducibility") or {}, _REPRO_FIELDS,
        ),
        "links": _allow(full_report.get("links") or {}, ("raw", "report")),
    }


__all__ = ["build_public_report"]
