"""Deterministic scripted provider for tests. Never touches the network."""
from __future__ import annotations

import json
from typing import Optional

from werewolf.llm.provider import ModelRequest, Provider, ProviderResult
from werewolf.llm.records import CostInfo, CostSource, ErrorCategory, TokenUsage


class FakeProviderExhausted(AssertionError):
    """Raised when a test makes more calls than it scripted."""


class FakeProvider:
    """Returns pre-scripted ProviderResults in order and captures requests."""

    name = "fake"

    def __init__(
        self,
        results: Optional[list[ProviderResult]] = None,
        default: Optional[ProviderResult] = None,
    ):
        """`default` (if given) is returned once the scripted queue is
        exhausted, enabling full-game tests of unknown call counts."""
        self._results: list[ProviderResult] = list(results or [])
        self._default = default
        self.requests: list[ModelRequest] = []

    def enqueue(self, result: ProviderResult) -> None:
        self._results.append(result)

    def complete(self, request: ModelRequest) -> ProviderResult:
        self.requests.append(request)
        if not self._results:
            if self._default is not None:
                return self._default
            raise FakeProviderExhausted(
                f"FakeProvider received call #{len(self.requests)} but only "
                f"{len(self.requests) - 1} result(s) were scripted"
            )
        return self._results.pop(0)

    @property
    def calls_made(self) -> int:
        return len(self.requests)


# ---------------------------------------------------------------------------
# Result factories for common test scenarios.
# ---------------------------------------------------------------------------

def success_result(
    payload: Optional[dict] = None,
    *,
    text: Optional[str] = None,
    input_tokens: int = 500,
    cached_input_tokens: Optional[int] = None,
    output_tokens: int = 100,
    reasoning_tokens: Optional[int] = None,
    cost_ticks: Optional[int] = 12_500_000,
    resolved_model: str = "fake-model-1",
    finish_reason: str = "stop",
    latency_ms: int = 150,
) -> ProviderResult:
    """A successful API response. `payload` is serialized as the JSON body
    unless raw `text` is given (use text for malformed-JSON scenarios)."""
    if text is None:
        text = json.dumps(payload if payload is not None else {})
    total = input_tokens + output_tokens + (reasoning_tokens or 0)
    cost = (
        CostInfo.from_ticks(cost_ticks)
        if cost_ticks is not None
        else CostInfo.unavailable()
    )
    return ProviderResult(
        ok=True,
        text=text,
        usage=TokenUsage(
            input_tokens=input_tokens,
            cached_input_tokens=cached_input_tokens,
            output_tokens=output_tokens,
            reasoning_tokens=reasoning_tokens,
            total_tokens=total,
        ),
        cost=cost,
        resolved_model=resolved_model,
        finish_reason=finish_reason,
        latency_ms=latency_ms,
    )


def estimated_cost_result(payload: dict, usd: float, **kwargs) -> ProviderResult:
    result = success_result(payload, cost_ticks=None, **kwargs)
    result.cost = CostInfo.estimated(usd, CostSource.PRICING_TABLE_ESTIMATE)
    return result


def error_result(
    category: ErrorCategory,
    *,
    message: str = "simulated provider error",
    retryable: Optional[bool] = None,
    cost_ticks: Optional[int] = None,
    usage: Optional[TokenUsage] = None,
    latency_ms: int = 50,
) -> ProviderResult:
    """A failed API attempt. May still carry usage/cost (e.g. a request that
    was billed before failing, or max_output_tokens truncation)."""
    if retryable is None:
        retryable = category in (
            ErrorCategory.RATE_LIMITED,
            ErrorCategory.TIMEOUT,
            ErrorCategory.NETWORK_ERROR,
            ErrorCategory.PROVIDER_ERROR,
        )
    return ProviderResult(
        ok=False,
        usage=usage or TokenUsage(),
        cost=(
            CostInfo.from_ticks(cost_ticks)
            if cost_ticks is not None
            else CostInfo.unavailable()
        ),
        error_category=category,
        error_message=message,
        retryable=retryable,
        latency_ms=latency_ms,
    )
