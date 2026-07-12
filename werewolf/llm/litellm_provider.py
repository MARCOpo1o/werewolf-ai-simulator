"""LiteLLM adapter for non-xAI providers (Gemini, OpenAI, Anthropic, ...).

IMPORTANT accounting caveat: LiteLLM computes cost from its community
price map (model_prices_and_context_window.json), NOT from the provider's
billed amount. Records from this adapter are therefore labeled
cost_source=pricing_table_estimate (or unavailable), never
provider_reported. Exact-billing providers (xAI) use their direct adapter.
"""
from __future__ import annotations

import logging
import time
from typing import Optional

from werewolf.llm.provider import ModelRequest, ProviderResult
from werewolf.llm.records import CostInfo, CostSource, ErrorCategory, TokenUsage

logger = logging.getLogger("werewolf.llm.litellm")

try:
    import litellm
    HAS_LITELLM = True
except ImportError:
    HAS_LITELLM = False

_RETRYABLE = {
    ErrorCategory.RATE_LIMITED,
    ErrorCategory.TIMEOUT,
    ErrorCategory.NETWORK_ERROR,
    ErrorCategory.PROVIDER_ERROR,
}

# litellm exception class name -> category (litellm maps all providers to
# OpenAI-style exception classes; match by name so this works across
# versions without importing each class).
_EXCEPTION_NAME_MAP = {
    "RateLimitError": ErrorCategory.RATE_LIMITED,
    "AuthenticationError": ErrorCategory.AUTHENTICATION_ERROR,
    "PermissionDeniedError": ErrorCategory.AUTHENTICATION_ERROR,
    "ContextWindowExceededError": ErrorCategory.CONTEXT_WINDOW_EXCEEDED,
    "Timeout": ErrorCategory.TIMEOUT,
    "APITimeoutError": ErrorCategory.TIMEOUT,
    "APIConnectionError": ErrorCategory.NETWORK_ERROR,
    "ServiceUnavailableError": ErrorCategory.PROVIDER_ERROR,
    "InternalServerError": ErrorCategory.PROVIDER_ERROR,
    "NotFoundError": ErrorCategory.UNKNOWN_MODEL,
    "BadRequestError": ErrorCategory.PROVIDER_ERROR,
}


def classify_exception(exc: Exception) -> ErrorCategory:
    for klass in type(exc).__mro__:
        category = _EXCEPTION_NAME_MAP.get(klass.__name__)
        if category is not None:
            return category
    text = f"{type(exc).__name__} {exc}".lower()
    if "rate limit" in text or "429" in text or "quota" in text:
        return ErrorCategory.RATE_LIMITED
    if "context" in text and ("length" in text or "window" in text):
        return ErrorCategory.CONTEXT_WINDOW_EXCEEDED
    if "timeout" in text or "timed out" in text:
        return ErrorCategory.TIMEOUT
    if "connection" in text or "network" in text:
        return ErrorCategory.NETWORK_ERROR
    if "api key" in text or "unauthorized" in text or "401" in text:
        return ErrorCategory.AUTHENTICATION_ERROR
    return ErrorCategory.PROVIDER_ERROR


def _sanitize_error_message(exc: Exception, limit: int = 300) -> str:
    text = f"{type(exc).__name__}: {exc}"
    redacted = []
    for token in text.split():
        lowered = token.lower()
        if lowered.startswith(("aiza", "sk-", "xai-", "bearer", "key=")) or "api_key" in lowered:
            redacted.append("[REDACTED]")
        else:
            redacted.append(token)
    return " ".join(redacted)[:limit]


def build_completion_kwargs(request: ModelRequest, api_key: str, timeout: int) -> dict:
    """Map GenerationConfig onto litellm.completion kwargs (only fields
    that were explicitly requested)."""
    g = request.generation
    kwargs = dict(
        model=request.model,
        messages=[
            {"role": "system", "content": request.system_prompt},
            {"role": "user", "content": request.user_prompt},
        ],
        api_key=api_key,
        timeout=timeout,
    )
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
    if g.structured_output:
        kwargs["response_format"] = {"type": "json_object"}
    return kwargs


class LiteLLMProvider:
    name = "litellm"

    def __init__(self, api_key: str, timeout: int = 120):
        if not HAS_LITELLM:
            raise RuntimeError("litellm is not installed. Run: pip install litellm")
        if not api_key:
            raise ValueError("LiteLLMProvider requires a non-empty API key")
        self._api_key = api_key  # passed per-call; never logged or recorded
        self._timeout = timeout

    def complete(self, request: ModelRequest) -> ProviderResult:
        started = time.monotonic()
        try:
            kwargs = build_completion_kwargs(
                request, api_key=self._api_key, timeout=self._timeout,
            )
            response = litellm.completion(**kwargs)
            latency_ms = int((time.monotonic() - started) * 1000)
            return self._result_from_response(response, latency_ms)
        except Exception as exc:
            latency_ms = int((time.monotonic() - started) * 1000)
            category = classify_exception(exc)
            logger.warning("LiteLLM call failed (%s): %s", category.value,
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

        def usage_field(obj, *names) -> Optional[int]:
            for n in names:
                value = getattr(obj, n, None)
                if value is not None:
                    return int(value)
            return None

        reasoning_tokens = None
        completion_details = getattr(usage_obj, "completion_tokens_details", None)
        if completion_details is not None:
            reasoning_tokens = usage_field(completion_details, "reasoning_tokens")

        cached_tokens = None
        prompt_details = getattr(usage_obj, "prompt_tokens_details", None)
        if prompt_details is not None:
            cached_tokens = usage_field(prompt_details, "cached_tokens")

        usage = TokenUsage(
            input_tokens=usage_field(usage_obj, "prompt_tokens"),
            cached_input_tokens=cached_tokens,
            output_tokens=usage_field(usage_obj, "completion_tokens"),
            reasoning_tokens=reasoning_tokens,
            total_tokens=usage_field(usage_obj, "total_tokens"),
        )

        # Price-map estimate, clearly labeled as such.
        cost = CostInfo.unavailable()
        try:
            usd = litellm.completion_cost(completion_response=response)
            if usd is not None:
                cost = CostInfo.estimated(
                    float(usd), CostSource.PRICING_TABLE_ESTIMATE
                )
        except Exception:
            logger.debug("completion_cost unavailable for this model")

        text = None
        finish_reason = None
        choices = getattr(response, "choices", None)
        if choices:
            message = getattr(choices[0], "message", None)
            text = getattr(message, "content", None)
            finish_reason = getattr(choices[0], "finish_reason", None)
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
        elif finish_reason and finish_reason.lower() in ("length", "max_tokens"):
            result.error_category = ErrorCategory.MAX_OUTPUT_TOKENS
        return result
