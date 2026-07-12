"""Benchmark validity gates: classify games as clean or dirty.

A completed game whose votes came from the random fallback measures
formatting robustness, not strategic reasoning. Benchmarks therefore
report two result sets: all-games (practical robustness) and clean-games
(strategic capability). Dirty games are never silently discarded - their
counts and violation reasons are always reported.

Policy v1 violations:
- fallback_<action>: random fallback on any strategic action
  (assess_beliefs fallbacks are excluded here; they surface as snapshot
  coverage instead)
- regex_recovered_action: a strategic action salvaged from malformed
  output by regex (not the model's well-formed decision)
- context_window_exceeded / unknown_model: systematic API failures
- resolved_model_mismatch: the provider served a different model family
  than requested (e.g. silent redirect of a retired slug)
- low_snapshot_coverage: < MIN_SNAPSHOT_COVERAGE of emitted belief
  snapshots were valid (only when instrumentation was on)
"""
from __future__ import annotations

import json
from collections import Counter

VALIDITY_POLICY_VERSION = 1
MIN_SNAPSHOT_COVERAGE = 0.95


def _model_family_match(requested: str, resolved: str) -> bool:
    base = requested.split("/")[-1].lower()
    res = resolved.split("/")[-1].lower()
    return base in res or res in base


def classify_game(rows: list[dict]) -> dict:
    """Returns {"clean": bool, "violations": {name: count},
    "policy_version": int}."""
    violations: Counter = Counter()
    config = next((r for r in rows if r.get("type") == "config"), {})

    for r in rows:
        if r.get("type") != "llm_call":
            continue
        action = r.get("required_action")
        if not r.get("api_attempted"):
            if (r.get("error_category") == "fallback_used"
                    and action != "assess_beliefs"):
                violations[f"fallback_{action}"] += 1
            continue
        category = r.get("error_category")
        if category == "context_window_exceeded":
            violations["context_window_exceeded"] += 1
        if category == "unknown_model":
            violations["unknown_model"] += 1
        if (r.get("parse_method") == "regex" and r.get("validation_ok")
                and action != "assess_beliefs"):
            violations["regex_recovered_action"] += 1
        requested, resolved = r.get("requested_model"), r.get("resolved_model")
        if requested and resolved and not _model_family_match(requested, resolved):
            violations["resolved_model_mismatch"] += 1

    if config.get("belief_snapshots"):
        emitted = valid = 0
        for r in rows:
            if r.get("type") != "event":
                continue
            e = r["event"]
            if e.get("type") == "belief_snapshot":
                emitted += 1
                if (e.get("payload") or {}).get("valid"):
                    valid += 1
        if emitted and (valid / emitted) < MIN_SNAPSHOT_COVERAGE:
            violations["low_snapshot_coverage"] += 1

    return {
        "clean": not violations,
        "violations": dict(violations),
        "policy_version": VALIDITY_POLICY_VERSION,
    }


def classify_game_from_file(log_path: str) -> dict:
    with open(log_path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return classify_game(rows)


def summarize_validity(per_game: list[dict]) -> dict:
    """Rollup: clean/dirty counts and violation totals across games."""
    totals: Counter = Counter()
    dirty = 0
    for v in per_game:
        if not v["clean"]:
            dirty += 1
        totals.update(v["violations"])
    return {
        "policy_version": VALIDITY_POLICY_VERSION,
        "games": len(per_game),
        "clean_games": len(per_game) - dirty,
        "dirty_games": dirty,
        "violations_by_type": dict(totals),
    }
