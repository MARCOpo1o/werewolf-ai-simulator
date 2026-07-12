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
