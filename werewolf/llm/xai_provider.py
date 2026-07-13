"""Direct xAI adapter. Preserves exact provider-reported billing
(usage.cost_in_usd_ticks) per https://docs.x.ai/developers/cost-tracking.

Never raises for API failures: returns ProviderResult(ok=False, ...) with a
normalized ErrorCategory. Never logs or attaches the API key to anything.
"""
from __future__ import annotations

import logging
import time
from typing import Optional

from werewolf.llm.provider import ModelRequest, ProviderResult
from werewolf.llm.records import CostInfo, ErrorCategory, TokenUsage

logger = logging.getLogger("werewolf.llm.xai")

try:
    from xai_sdk import Client
    from xai_sdk.chat import system, user
    HAS_XAI = True
except ImportError:
    HAS_XAI = False


# Exception classification: xai-sdk is gRPC-based; we match defensively on
# class names, gRPC status names, and message substrings so a pinned SDK
# upgrade degrades to provider_error instead of crashing.
_RETRYABLE = {
    ErrorCategory.RATE_LIMITED,
    ErrorCategory.TIMEOUT,
    ErrorCategory.NETWORK_ERROR,
    ErrorCategory.PROVIDER_ERROR,
}


def classify_exception(exc: Exception) -> ErrorCategory:
    name = type(exc).__name__.lower()
    text = f"{name} {exc}".lower()

    if "resource_exhausted" in text or "rate limit" in text or "429" in text:
        return ErrorCategory.RATE_LIMITED
    if (
        "unauthenticated" in text
        or "permission_denied" in text
        or "invalid api key" in text
        or "unauthorized" in text
        or "401" in text
        or "403" in text
    ):
        return ErrorCategory.AUTHENTICATION_ERROR
    if ("context" in text and ("length" in text or "window" in text)) or (
        "prompt" in text and "too long" in text
    ) or "maximum prompt length" in text:
        return ErrorCategory.CONTEXT_WINDOW_EXCEEDED
    if "deadline" in text or "timeout" in text or "timed out" in text:
        return ErrorCategory.TIMEOUT
    if (
        "unavailable" in text
        or "connection" in text
        or "network" in text
        or "dns" in text
    ):
        return ErrorCategory.NETWORK_ERROR
    if "not found" in text and "model" in text:
        return ErrorCategory.UNKNOWN_MODEL
    return ErrorCategory.PROVIDER_ERROR


def build_chat_kwargs(request: ModelRequest) -> dict:
    """Map GenerationConfig onto xai-sdk chat.create kwargs (only the
    fields that were explicitly requested)."""
    g = request.generation
    kwargs = {"model": request.model}
    if g.reasoning_effort is not None:
        kwargs["reasoning_effort"] = g.reasoning_effort
    if g.temperature is not None:
        kwargs["temperature"] = g.temperature
    if g.top_p is not None:
        kwargs["top_p"] = g.top_p
    if g.max_output_tokens is not None:
        kwargs["max_tokens"] = g.max_output_tokens
    if g.provider_seed is not None:
        kwargs["seed"] = g.provider_seed
    return kwargs


def _unexpected_kwarg_name(exc: Exception, kwargs: dict):
    """Extract the offending kwarg from a TypeError/ValueError, if
    identifiable (matches quoted names, raw names, or names with spaces:
    'Invalid reasoning effort: ...' -> reasoning_effort)."""
    text = str(exc).lower()
    for name in kwargs:
        candidates = (f"'{name}'", name, name.replace("_", " "))
        if any(c in text for c in candidates):
            return name
    return None


def _sanitize_error_message(exc: Exception, limit: int = 300) -> str:
    """Exception text with anything resembling a key redacted."""
    text = f"{type(exc).__name__}: {exc}"
    redacted = []
    for token in text.split():
        if token.lower().startswith(("xai-", "bearer")) or "api_key" in token.lower():
            redacted.append("[REDACTED]")
        else:
            redacted.append(token)
    return " ".join(redacted)[:limit]


class XAIProvider:
    name = "xai"

    def __init__(self, api_key: str, timeout: int = 120):
        if not HAS_XAI:
            raise RuntimeError(
                "xai-sdk is not installed. Run: pip install xai-sdk"
            )
        if not api_key:
            raise ValueError("XAIProvider requires a non-empty API key")
        self._client = Client(api_key=api_key, timeout=timeout)

    def complete(self, request: ModelRequest) -> ProviderResult:
        started = time.monotonic()
        try:
            chat, dropped = self._create_chat(request)
            chat.append(system(request.system_prompt))
            chat.append(user(request.user_prompt))
            response = chat.sample()
            latency_ms = int((time.monotonic() - started) * 1000)
            result = self._result_from_response(response, latency_ms)
            if dropped:
                result.provider_metadata["generation_dropped"] = dropped
            return result
        except Exception as exc:  # normalized, never propagated
            latency_ms = int((time.monotonic() - started) * 1000)
            category = classify_exception(exc)
            logger.warning("xAI call failed (%s): %s", category.value,
                           _sanitize_error_message(exc))
            return ProviderResult(
                ok=False,
                error_category=category,
                error_message=_sanitize_error_message(exc),
                retryable=category in _RETRYABLE,
                latency_ms=latency_ms,
            )

    def _create_chat(self, request: ModelRequest):
        """chat.create with the requested GenerationConfig. Params (or
        param VALUES) the installed xai-sdk rejects are dropped one at a
        time (with a warning) and reported back - never silently ignored.
        TypeError = unknown kwarg; ValueError = client-side value
        validation (e.g. sdk 1.17.0 only accepts reasoning_effort
        'low'/'high')."""
        kwargs = build_chat_kwargs(request)
        dropped: list[str] = []
        while True:
            try:
                return self._client.chat.create(**kwargs), dropped
            except (TypeError, ValueError) as exc:
                offender = _unexpected_kwarg_name(exc, kwargs)
                if offender is None or offender == "model":
                    raise
                logger.warning(
                    "xai-sdk rejected %s=%r (%s); dropping",
                    offender, kwargs[offender], exc,
                )
                kwargs.pop(offender)
                dropped.append(offender)

    @staticmethod
    def _result_from_response(response, latency_ms: int) -> ProviderResult:
        usage_obj = getattr(response, "usage", None)

        def usage_field(*names) -> Optional[int]:
            for n in names:
                value = getattr(usage_obj, n, None)
                if value is not None:
                    return int(value)
            return None

        usage = TokenUsage(
            input_tokens=usage_field("prompt_tokens", "input_tokens"),
            cached_input_tokens=usage_field(
                "cached_prompt_text_tokens", "cached_prompt_tokens",
                "cached_tokens",
            ),
            output_tokens=usage_field("completion_tokens", "output_tokens"),
            reasoning_tokens=usage_field("reasoning_tokens"),
            total_tokens=usage_field("total_tokens"),
        )

        # Exact billed cost: raw integer ticks preferred; SDK convenience
        # cost_usd used only as a cross-check source, never as a substitute.
        ticks = usage_field("cost_in_usd_ticks")
        cost = CostInfo.from_ticks(ticks) if ticks is not None else CostInfo.unavailable()

        text = getattr(response, "content", None)
        finish_reason = getattr(response, "finish_reason", None)
        if finish_reason is not None:
            finish_reason = str(finish_reason)

        result = ProviderResult(
            ok=True,
            text=text,
            usage=usage,
            cost=cost,
            resolved_model=getattr(response, "model", None),
            provider_request_id=getattr(response, "id", None),
            finish_reason=finish_reason,
            latency_ms=latency_ms,
        )
        if text is None or not str(text).strip():
            result.ok = False
            result.error_category = ErrorCategory.EMPTY_RESPONSE
            result.retryable = True
        elif finish_reason and "length" in finish_reason.lower():
            result.error_category = ErrorCategory.MAX_OUTPUT_TOKENS
        return result
