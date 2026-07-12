"""Usage ledger: records every LLM attempt, streams to a sink, aggregates.

Accounting rules:
- Exact costs are summed as raw integer ticks (no float drift).
- Unavailable cost is surfaced as counts + cost_complete=False; it is
  never silently treated as zero. cost_usd_total is None when calls were
  made but no cost is known at all.
- Thread-safe: record() may be called from concurrent games later.
"""
from __future__ import annotations

import threading
from collections import defaultdict
from typing import Callable, Optional

from werewolf.llm.records import CostSource, ErrorCategory, UsageRecord

Sink = Callable[[dict], None]


class UsageLedger:
    def __init__(self, sink: Optional[Sink] = None):
        self._sink = sink
        self._records: list[UsageRecord] = []
        self._lock = threading.Lock()

    def record(self, record: UsageRecord) -> None:
        with self._lock:
            self._records.append(record)
            if self._sink is not None:
                self._sink(record.to_json_dict())

    @property
    def records(self) -> list[UsageRecord]:
        with self._lock:
            return list(self._records)

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def game_summary(self) -> dict:
        records = self.records
        api_records = [r for r in records if r.api_attempted]

        calls = len(api_records)
        api_failures = sum(1 for r in api_records if not r.api_ok)
        parse_failures = sum(1 for r in api_records if r.parse_ok is False)
        validation_failures = sum(
            1 for r in api_records if r.validation_ok is False
        )
        retries = sum(1 for r in api_records if r.attempt > 1)
        fallbacks = sum(
            1
            for r in records
            if r.error_category == ErrorCategory.FALLBACK_USED
        )
        recovered_parses = sum(
            1 for r in api_records if r.parse_method in ("repaired", "regex")
        )

        tokens = {
            "input_tokens": 0,
            "cached_input_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
            "total_tokens": 0,
        }
        calls_missing_usage = 0
        for r in api_records:
            u = r.usage
            if u.total_tokens is None and u.input_tokens is None:
                calls_missing_usage += 1
            for key in tokens:
                value = getattr(u, key)
                if value is not None:
                    tokens[key] += value

        cost_ticks_total = 0
        has_ticks = False
        cost_usd_known = 0.0
        calls_with_cost = 0
        calls_with_unavailable_cost = 0
        cost_by_source: dict[str, dict] = {}
        for r in api_records:
            c = r.cost
            src = c.source.value
            bucket = cost_by_source.setdefault(src, {"calls": 0, "usd": 0.0})
            bucket["calls"] += 1
            if c.ticks is not None:
                cost_ticks_total += c.ticks
                has_ticks = True
            if c.usd is not None:
                cost_usd_known += c.usd
                bucket["usd"] += c.usd
                calls_with_cost += 1
            if c.source == CostSource.UNAVAILABLE:
                calls_with_unavailable_cost += 1
                bucket["usd"] = None  # never report a fabricated 0 for this bucket

        if calls == 0:
            cost_usd_total = 0.0
        elif calls_with_cost == 0:
            cost_usd_total = None  # calls were made; no cost is known
        else:
            cost_usd_total = cost_usd_known

        max_round = max((r.context.round for r in api_records), default=0)
        avg_cost_per_round = (
            (cost_usd_known / max_round)
            if (max_round > 0 and calls_with_cost > 0)
            else None
        )

        return {
            "calls": calls,
            "api_failures": api_failures,
            "parse_failures": parse_failures,
            "validation_failures": validation_failures,
            "retries": retries,
            "fallbacks": fallbacks,
            "recovered_parses": recovered_parses,
            "calls_missing_usage": calls_missing_usage,
            "tokens": tokens,
            "cost_ticks_total": cost_ticks_total if has_ticks else None,
            "cost_usd_total": cost_usd_total,
            "cost_complete": calls_with_unavailable_cost == 0,
            "calls_with_unavailable_cost": calls_with_unavailable_cost,
            "cost_by_source": cost_by_source,
            "by_player": self._breakdown(api_records, lambda r: r.context.player_id),
            "by_role": self._breakdown(api_records, lambda r: r.context.player_role),
            "by_phase": self._breakdown(api_records, lambda r: r.context.phase),
            "by_required_action": self._breakdown(
                api_records, lambda r: r.context.required_action
            ),
            "avg_cost_per_round": avg_cost_per_round,
            "errors_by_category": self._error_counts(records),
        }

    @staticmethod
    def _breakdown(records: list[UsageRecord], key_fn) -> dict:
        groups: dict = defaultdict(
            lambda: {
                "calls": 0,
                "cost_ticks": 0,
                "_has_ticks": False,
                "cost_usd": 0.0,
                "_has_usd": False,
                "total_tokens": 0,
            }
        )
        for r in records:
            g = groups[key_fn(r)]
            g["calls"] += 1
            if r.cost.ticks is not None:
                g["cost_ticks"] += r.cost.ticks
                g["_has_ticks"] = True
            if r.cost.usd is not None:
                g["cost_usd"] += r.cost.usd
                g["_has_usd"] = True
            if r.usage.total_tokens is not None:
                g["total_tokens"] += r.usage.total_tokens

        out = {}
        for key, g in groups.items():
            out[str(key)] = {
                "calls": g["calls"],
                "cost_ticks": g["cost_ticks"] if g["_has_ticks"] else None,
                "cost_usd": g["cost_usd"] if g["_has_usd"] else None,
                "total_tokens": g["total_tokens"],
            }
        return out

    @staticmethod
    def _error_counts(records: list[UsageRecord]) -> dict:
        counts: dict[str, int] = defaultdict(int)
        for r in records:
            if r.error_category is not None:
                counts[r.error_category.value] += 1
        return dict(counts)


def _percentile(sorted_values: list[float], fraction: float) -> float:
    """Nearest-rank percentile on a pre-sorted list."""
    if not sorted_values:
        raise ValueError("empty")
    rank = max(1, -(-len(sorted_values) * fraction // 1))  # ceil
    return sorted_values[int(rank) - 1]


def aggregate_game_summaries(summaries: list[dict]) -> dict:
    """Batch-level rollup of per-game usage summaries (game_summary()).

    Same accounting rules as the per-game summary: exact ticks summed as
    integers, unavailable cost surfaced via counts and cost_complete,
    never fabricated zeros. Per-game cost stats (mean/median/p90/min/max)
    are computed only over games with known cost; the count of excluded
    games is reported alongside.
    """
    games = len(summaries)
    counters = (
        "calls", "api_failures", "parse_failures", "validation_failures",
        "retries", "fallbacks", "recovered_parses", "calls_missing_usage",
    )
    out: dict = {"games": games}
    for key in counters:
        out[key] = sum(s.get(key, 0) for s in summaries)

    tokens = {
        "input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0,
        "reasoning_tokens": 0, "total_tokens": 0,
    }
    for s in summaries:
        for key in tokens:
            tokens[key] += (s.get("tokens") or {}).get(key, 0) or 0
    out["tokens"] = tokens

    ticks_values = [
        s["cost_ticks_total"] for s in summaries
        if s.get("cost_ticks_total") is not None
    ]
    out["cost_ticks_total"] = sum(ticks_values) if ticks_values else None

    known_costs = [
        s["cost_usd_total"] for s in summaries
        if s.get("cost_usd_total") is not None
    ]
    games_with_calls = [s for s in summaries if s.get("calls", 0) > 0]
    if games == 0 or not games_with_calls:
        out["cost_usd_total"] = 0.0 if not games_with_calls else None
    elif not known_costs:
        out["cost_usd_total"] = None
    else:
        out["cost_usd_total"] = sum(known_costs)

    out["games_with_incomplete_cost"] = sum(
        1 for s in summaries
        if not s.get("cost_complete", True) or (
            s.get("calls", 0) > 0 and s.get("cost_usd_total") is None
        )
    )
    out["cost_complete"] = out["games_with_incomplete_cost"] == 0

    costed = sorted(
        s["cost_usd_total"] for s in summaries
        if s.get("cost_usd_total") is not None and s.get("calls", 0) > 0
    )
    if costed:
        out["cost_per_game"] = {
            "games_counted": len(costed),
            "mean": sum(costed) / len(costed),
            "median": _percentile(costed, 0.5),
            "p90": _percentile(costed, 0.9),
            "min": costed[0],
            "max": costed[-1],
        }
    else:
        out["cost_per_game"] = None

    by_source: dict[str, dict] = {}
    for s in summaries:
        for source, bucket in (s.get("cost_by_source") or {}).items():
            agg = by_source.setdefault(source, {"calls": 0, "usd": 0.0})
            agg["calls"] += bucket.get("calls", 0)
            if bucket.get("usd") is None:
                agg["usd"] = None
            elif agg["usd"] is not None:
                agg["usd"] += bucket["usd"]
    out["cost_by_source"] = by_source

    errors: dict[str, int] = defaultdict(int)
    for s in summaries:
        for category, count in (s.get("errors_by_category") or {}).items():
            errors[category] += count
    out["errors_by_category"] = dict(errors)

    return out
