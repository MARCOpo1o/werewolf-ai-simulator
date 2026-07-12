"""Normalized, versioned usage records for every LLM call attempt.

Design rules (see plan):
- Every paid attempt produces one record, including malformed/invalid ones.
- Exact provider-reported cost (xAI ticks) is stored as a raw integer.
- Estimated costs are always labeled via CostSource; unavailable cost is
  represented as None, never as a fabricated 0.
- No API keys, authorization headers, or hidden chain-of-thought content.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

# v2: added requested_generation (GenerationConfig snapshot per attempt)
SCHEMA_VERSION = 2

# xAI: 1 USD == 10^10 ticks (https://docs.x.ai/developers/cost-tracking)
TICKS_PER_USD = 10_000_000_000

# provider_metadata keys that must never be persisted.
_FORBIDDEN_METADATA_KEYS = frozenset(
    {"api_key", "apikey", "authorization", "auth_header", "bearer", "secret"}
)


class CostSource(str, Enum):
    PROVIDER_REPORTED = "provider_reported"
    PRICING_TABLE_ESTIMATE = "pricing_table_estimate"
    TOKENIZER_ESTIMATE = "tokenizer_estimate"
    UNAVAILABLE = "unavailable"


class ErrorCategory(str, Enum):
    RATE_LIMITED = "rate_limited"
    AUTHENTICATION_ERROR = "authentication_error"
    CONTEXT_WINDOW_EXCEEDED = "context_window_exceeded"
    MAX_OUTPUT_TOKENS = "max_output_tokens"
    TIMEOUT = "timeout"
    NETWORK_ERROR = "network_error"
    PROVIDER_ERROR = "provider_error"
    EMPTY_RESPONSE = "empty_response"
    MALFORMED_JSON = "malformed_json"
    INVALID_GAME_ACTION = "invalid_game_action"
    UNKNOWN_MODEL = "unknown_model"
    MISSING_API_KEY = "missing_api_key"
    FALLBACK_USED = "fallback_used"
    GAME_TURN_LIMIT = "game_turn_limit"
    COMPLETED = "completed"


@dataclass
class CostInfo:
    """Cost of a single attempt. `usd` is None when genuinely unknown."""

    source: CostSource = CostSource.UNAVAILABLE
    ticks: Optional[int] = None
    usd: Optional[float] = None

    @classmethod
    def from_ticks(cls, ticks: int) -> "CostInfo":
        """Exact provider-reported cost from raw integer ticks."""
        if not isinstance(ticks, int) or isinstance(ticks, bool) or ticks < 0:
            raise ValueError(f"ticks must be a non-negative int, got {ticks!r}")
        return cls(
            source=CostSource.PROVIDER_REPORTED,
            ticks=ticks,
            usd=ticks / TICKS_PER_USD,
        )

    @classmethod
    def estimated(cls, usd: float, source: CostSource) -> "CostInfo":
        if source in (CostSource.PROVIDER_REPORTED, CostSource.UNAVAILABLE):
            raise ValueError(f"estimated() requires an estimate source, got {source}")
        return cls(source=source, ticks=None, usd=usd)

    @classmethod
    def unavailable(cls) -> "CostInfo":
        return cls()

    def to_json_dict(self) -> dict:
        return {"source": self.source.value, "ticks": self.ticks, "usd": self.usd}


@dataclass
class TokenUsage:
    """Token counts. None means the provider did not report the field."""

    input_tokens: Optional[int] = None
    cached_input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    reasoning_tokens: Optional[int] = None
    total_tokens: Optional[int] = None

    def to_json_dict(self) -> dict:
        return {
            "input_tokens": self.input_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "output_tokens": self.output_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass
class CallContext:
    """Where in the experiment this call happened."""

    game_id: str
    round: int
    phase: str
    required_action: str
    player_id: int
    player_role: str
    player_team: str
    seed: Optional[int] = None
    batch_id: Optional[str] = None
    trial_index: Optional[int] = None
    prompt_version: Optional[str] = None
    model_alias: Optional[str] = None

    def to_json_dict(self) -> dict:
        return {
            "game_id": self.game_id,
            "batch_id": self.batch_id,
            "trial_index": self.trial_index,
            "seed": self.seed,
            "round": self.round,
            "phase": self.phase,
            "required_action": self.required_action,
            "player_id": self.player_id,
            "player_role": self.player_role,
            "player_team": self.player_team,
            "prompt_version": self.prompt_version,
            "model_alias": self.model_alias,
        }


def new_call_id() -> str:
    return uuid.uuid4().hex


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _scrub_metadata(metadata: dict) -> dict:
    """Drop any metadata keys that could carry secrets."""
    return {
        k: v
        for k, v in metadata.items()
        if k.lower() not in _FORBIDDEN_METADATA_KEYS
    }


@dataclass
class UsageRecord:
    """One record per LLM call attempt (or fallback resolution).

    `api_attempted` is False only for records that represent non-API
    resolutions (missing key, fallback after retries exhausted). Those
    records carry zero cost with source=unavailable and exist so that
    fallback use is observable and attributable.
    """

    context: CallContext
    provider: str
    requested_model: str
    call_id: str = field(default_factory=new_call_id)
    attempt: int = 1
    ts: str = field(default_factory=utc_now_iso)
    resolved_model: Optional[str] = None
    usage: TokenUsage = field(default_factory=TokenUsage)
    cost: CostInfo = field(default_factory=CostInfo.unavailable)
    latency_ms: Optional[int] = None
    provider_request_id: Optional[str] = None
    finish_reason: Optional[str] = None
    api_attempted: bool = True
    api_ok: bool = False
    parse_ok: Optional[bool] = None
    parse_method: Optional[str] = None  # "direct" | "repaired" | "regex"
    validation_ok: Optional[bool] = None
    error_category: Optional[ErrorCategory] = None
    retryable: Optional[bool] = None
    requested_generation: Optional[dict] = None
    provider_metadata: dict = field(default_factory=dict)

    def to_json_dict(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "ts": self.ts,
            "call_id": self.call_id,
            "attempt": self.attempt,
            **self.context.to_json_dict(),
            "provider": self.provider,
            "requested_model": self.requested_model,
            "resolved_model": self.resolved_model,
            "usage": self.usage.to_json_dict(),
            "cost": self.cost.to_json_dict(),
            "latency_ms": self.latency_ms,
            "provider_request_id": self.provider_request_id,
            "finish_reason": self.finish_reason,
            "api_attempted": self.api_attempted,
            "api_ok": self.api_ok,
            "parse_ok": self.parse_ok,
            "parse_method": self.parse_method,
            "validation_ok": self.validation_ok,
            "error_category": (
                self.error_category.value if self.error_category else None
            ),
            "retryable": self.retryable,
            "requested_generation": self.requested_generation,
            "provider_metadata": _scrub_metadata(self.provider_metadata),
        }
