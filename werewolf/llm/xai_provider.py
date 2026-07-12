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
            if request.reasoning_effort is not None:
                try:
                    chat = self._client.chat.create(
                        model=request.model,
                        reasoning_effort=request.reasoning_effort,
                    )
                except TypeError:
                    # older xai-sdk without reasoning_effort support
                    logger.warning(
                        "xai-sdk ignored reasoning_effort=%s (unsupported)",
                        request.reasoning_effort,
                    )
                    chat = self._client.chat.create(model=request.model)
            else:
                chat = self._client.chat.create(model=request.model)
            chat.append(system(request.system_prompt))
            chat.append(user(request.user_prompt))
            response = chat.sample()
            latency_ms = int((time.monotonic() - started) * 1000)
            return self._result_from_response(response, latency_ms)
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
