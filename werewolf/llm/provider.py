"""Provider protocol and typed request/result objects.

A Provider never raises for API-level failures: it returns a structured
ProviderResult with ok=False and a normalized ErrorCategory, so the caller
can record the attempt uniformly. Only programming errors raise.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol, runtime_checkable

from werewolf.llm.records import CostInfo, ErrorCategory, TokenUsage


@dataclass(frozen=True)
class GenerationConfig:
    """Sampling parameters locked per experiment condition. None means
    'provider default' - acceptable for exploration, but benchmark runs
    should pin every field. Requested values are logged on every usage
    record; params a provider/SDK cannot apply are reported in
    provider_metadata.generation_dropped rather than silently ignored."""

    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_output_tokens: Optional[int] = None
    reasoning_effort: Optional[str] = None  # xAI/Gemini effort or budget
    provider_seed: Optional[int] = None
    structured_output: bool = False  # request JSON-mode where supported

    def to_json_dict(self) -> dict:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_output_tokens": self.max_output_tokens,
            "reasoning_effort": self.reasoning_effort,
            "provider_seed": self.provider_seed,
            "structured_output": self.structured_output,
        }


@dataclass
class ModelRequest:
    model: str
    system_prompt: str
    user_prompt: str
    generation: GenerationConfig = field(default_factory=GenerationConfig)


@dataclass
class ProviderResult:
    """Outcome of a single API attempt (success or structured failure)."""

    ok: bool
    text: Optional[str] = None
    usage: TokenUsage = field(default_factory=TokenUsage)
    cost: CostInfo = field(default_factory=CostInfo.unavailable)
    resolved_model: Optional[str] = None
    provider_request_id: Optional[str] = None
    finish_reason: Optional[str] = None
    error_category: Optional[ErrorCategory] = None
    error_message: Optional[str] = None  # sanitized; never auth material
    retryable: Optional[bool] = None
    latency_ms: Optional[int] = None
    provider_metadata: dict = field(default_factory=dict)


@runtime_checkable
class Provider(Protocol):
    """Minimal provider interface. Implementations own their API client
    and key; keys must never appear on results or in exceptions."""

    name: str

    def complete(self, request: ModelRequest) -> ProviderResult:
        ...
