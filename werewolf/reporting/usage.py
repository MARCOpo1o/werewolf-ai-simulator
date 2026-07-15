"""Canonical row-based usage and mixed-cost accounting for one game."""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Callable


TOKEN_FIELDS = (
    "input_tokens", "cached_input_tokens", "output_tokens",
    "reasoning_tokens", "total_tokens",
)


def _breakdown(calls: list[dict], key_fn: Callable[[dict], object]) -> dict:
    buckets = defaultdict(lambda: {
        "attempts": 0,
        "known_cost_usd": 0.0,
        "calls_with_known_cost": 0,
        "calls_without_known_cost": 0,
        "total_tokens": 0,
    })
    for call in calls:
        key = str(key_fn(call) if key_fn(call) is not None else "unknown")
        bucket = buckets[key]
        bucket["attempts"] += 1
        cost = call.get("cost") or {}
        usd = cost.get("usd")
        if isinstance(usd, (int, float)) and not isinstance(usd, bool):
            bucket["known_cost_usd"] += float(usd)
            bucket["calls_with_known_cost"] += 1
        else:
            bucket["calls_without_known_cost"] += 1
        total = (call.get("usage") or {}).get("total_tokens")
        if isinstance(total, int) and not isinstance(total, bool):
            bucket["total_tokens"] += total
    return dict(buckets)


def compute_usage(llm_calls: list[dict]) -> dict:
    api_calls = [call for call in llm_calls if call.get("api_attempted") is True]
    known_cost = 0.0
    calls_with_cost = calls_without_cost = 0
    cost_sources: set[str] = set()
    by_cost_source = defaultdict(lambda: {
        "attempts": 0, "known_cost_usd": 0.0,
        "calls_with_known_cost": 0, "calls_without_known_cost": 0,
    })
    tokens = {field: 0 for field in TOKEN_FIELDS}
    token_fields_missing = {field: 0 for field in TOKEN_FIELDS}

    for call in api_calls:
        usage = call.get("usage") or {}
        for field in TOKEN_FIELDS:
            value = usage.get(field)
            if isinstance(value, int) and not isinstance(value, bool):
                tokens[field] += value
            else:
                token_fields_missing[field] += 1

        cost = call.get("cost") or {}
        source = cost.get("source") or "unavailable"
        cost_sources.add(source)
        bucket = by_cost_source[source]
        bucket["attempts"] += 1
        usd = cost.get("usd")
        if isinstance(usd, (int, float)) and not isinstance(usd, bool):
            known_cost += float(usd)
            calls_with_cost += 1
            bucket["known_cost_usd"] += float(usd)
            bucket["calls_with_known_cost"] += 1
        else:
            calls_without_cost += 1
            bucket["calls_without_known_cost"] += 1

    if not api_calls or calls_without_cost == 0:
        cost_completeness = "complete"
    elif calls_with_cost:
        cost_completeness = "partial"
    else:
        cost_completeness = "unavailable"

    errors = defaultdict(int)
    for call in llm_calls:
        category = call.get("error_category")
        if category:
            errors[category] += 1

    result = {
        "attempts": len(api_calls),
        "decision_groups": len({
            call.get("call_id") for call in llm_calls if call.get("call_id")
        }),
        "api_failures": sum(1 for call in api_calls if not call.get("api_ok")),
        "parse_failures": sum(call.get("parse_ok") is False for call in api_calls),
        "validation_failures": sum(
            call.get("validation_ok") is False for call in api_calls
        ),
        "retries": sum(
            isinstance(call.get("attempt"), int) and call["attempt"] > 1
            for call in api_calls
        ),
        "fallbacks": errors.get("fallback_used", 0),
        "recovered_parses": sum(
            call.get("parse_method") in ("repaired", "regex") for call in api_calls
        ),
        "errors_by_category": dict(errors),
        "tokens": tokens,
        "token_fields_missing": token_fields_missing,
        "known_cost_usd": known_cost if calls_with_cost else None,
        "calls_with_known_cost": calls_with_cost,
        "calls_without_known_cost": calls_without_cost,
        "cost_completeness": cost_completeness,
        "cost_sources": sorted(cost_sources),
        "by_cost_source": dict(by_cost_source),
        "by_player": _breakdown(api_calls, lambda call: call.get("player_id")),
        "by_role": _breakdown(api_calls, lambda call: call.get("player_role")),
        "by_requested_model": _breakdown(
            api_calls, lambda call: call.get("requested_model")
        ),
        "by_resolved_model": _breakdown(
            api_calls, lambda call: call.get("resolved_model")
        ),
        "by_phase": _breakdown(api_calls, lambda call: call.get("phase")),
        "by_required_action": _breakdown(
            api_calls, lambda call: call.get("required_action")
        ),
    }
    return result


def compare_terminal_summary(computed: dict, terminal: dict | None) -> dict:
    if terminal is None:
        return {"status": "missing", "mismatches": []}
    mismatches = []
    pairs = {
        "calls": "attempts",
        "retries": "retries",
        "fallbacks": "fallbacks",
        "api_failures": "api_failures",
        "parse_failures": "parse_failures",
        "validation_failures": "validation_failures",
        "recovered_parses": "recovered_parses",
    }
    for terminal_key, computed_key in pairs.items():
        if terminal.get(terminal_key) != computed.get(computed_key):
            mismatches.append({
                "field": terminal_key,
                "computed": computed.get(computed_key),
                "terminal": terminal.get(terminal_key),
            })
    terminal_tokens = terminal.get("tokens") or {}
    for field in TOKEN_FIELDS:
        if terminal_tokens.get(field) != computed["tokens"].get(field):
            mismatches.append({
                "field": f"tokens.{field}",
                "computed": computed["tokens"].get(field),
                "terminal": terminal_tokens.get(field),
            })
    terminal_cost = terminal.get("cost_usd_total")
    computed_cost = computed.get("known_cost_usd")
    costs_match = (
        terminal_cost is None and computed_cost is None
    ) or (
        isinstance(terminal_cost, (int, float))
        and isinstance(computed_cost, (int, float))
        and math.isclose(terminal_cost, computed_cost, rel_tol=1e-9, abs_tol=1e-12)
    )
    if not costs_match:
        mismatches.append({
            "field": "cost_usd_total",
            "computed": computed_cost,
            "terminal": terminal_cost,
        })
    return {
        "status": "matched" if not mismatches else "mismatched",
        "mismatches": mismatches,
    }


__all__ = ["compute_usage", "compare_terminal_summary"]
