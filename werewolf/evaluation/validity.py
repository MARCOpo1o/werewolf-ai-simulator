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

from werewolf.engine.beliefs import recorded_belief_payload_valid
from werewolf.json_safety import as_mapping
from werewolf.llm.registry import MODEL_REGISTRY, resolved_model_matches

VALIDITY_POLICY_VERSION = 4
MIN_SNAPSHOT_COVERAGE = 0.95


def _model_identity_match(call: dict) -> bool:
    requested = call.get("requested_model")
    resolved = call.get("resolved_model")
    if (
        requested is not None and not isinstance(requested, str)
    ) or (
        resolved is not None and not isinstance(resolved, str)
    ):
        return False
    if not requested or not resolved:
        return True
    alias = call.get("model_alias")
    if isinstance(alias, str) and alias in MODEL_REGISTRY:
        return resolved_model_matches(MODEL_REGISTRY[alias], resolved)
    matching_specs = [
        spec for spec in MODEL_REGISTRY.values() if spec.model == requested
    ]
    if matching_specs:
        return any(resolved_model_matches(spec, resolved) for spec in matching_specs)
    return requested == resolved


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
        if not _model_identity_match(r):
            violations["resolved_model_mismatch"] += 1

    if config.get("belief_snapshots"):
        emitted = valid = 0
        for r in rows:
            if r.get("type") != "event":
                continue
            e = r["event"]
            if e.get("type") == "belief_snapshot":
                emitted += 1
                if recorded_belief_payload_valid(
                    as_mapping(e.get("payload"))
                ):
                    valid += 1
        if not emitted:
            violations["missing_snapshot_instrumentation"] += 1
        elif (valid / emitted) < MIN_SNAPSHOT_COVERAGE:
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
