"""Model registry: aliases -> provider, model ID, key env vars.

Secrets stay in environment variables; this module stores only the names
of those variables. Model IDs here are version-controlled experiment
configuration. NOTE: the resolved model reported back by the provider is
the ground truth (xAI silently redirects retired slugs), so UsageRecord
carries both requested_model and resolved_model.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Optional

from werewolf.llm.provider import GenerationConfig


@dataclass(frozen=True)
class ModelSpec:
    alias: Optional[str]           # registry alias, or None for passthrough
    provider: str                  # "xai" (more later)
    model: str                     # provider model slug
    api_key_env: tuple[str, ...] = ("GROK_API_KEY", "XAI_API_KEY")
    reasoning_effort: Optional[str] = None  # xAI: none|low|medium|high
    display_name: str = "Custom model"
    family: str = "Custom"
    description: str = "Provider model specified outside the curated catalog."
    speed_tier: str = "unknown"
    cost_tier: str = "unknown"
    tags: tuple[str, ...] = ()
    sort_order: int = 999
    selectable: bool = False
    experimental: bool = True
    acceptable_resolved_models: tuple[str, ...] = ()
    resolved_model_prefixes: tuple[str, ...] = ()


class ProviderBuildStatus(str, Enum):
    READY = "ready"
    MISSING_CREDENTIAL = "missing_credential"
    DEPENDENCY_UNAVAILABLE = "dependency_unavailable"
    INITIALIZATION_FAILED = "initialization_failed"


@dataclass(frozen=True)
class ProviderBuildResult:
    provider: Any = None
    status: ProviderBuildStatus = ProviderBuildStatus.INITIALIZATION_FAILED
    error: Optional[str] = None
    required_credentials: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return self.status == ProviderBuildStatus.READY and self.provider is not None


# NOTE (research comparability): before 2026-07 these aliases pointed at
# grok-4-1-fast / grok-4-1-fast-reasoning, which xAI retired on
# 2026-05-15 and silently redirects to grok-4.3. The aliases now target
# grok-4.3 explicitly, matching xAI's own redirect mapping (none/low
# reasoning effort), so requested_model == actual model again. Results
# logged before this change were served by whatever the redirect chose;
# compare eras via the resolved_model field, not the alias.
MODEL_REGISTRY: dict[str, ModelSpec] = {
    # NOTE: xai-sdk 1.17.0 client-side-validates reasoning_effort to
    # ('low','high') - it cannot express the API's "none" level, so every
    # call with "none" failed locally (all-fallback games). `fast` now
    # omits the param (server default); revisit if the SDK adds "none".
    "fast": ModelSpec(
        alias="fast",
        provider="xai",
        model="grok-4.3",
        reasoning_effort=None,
        display_name="Grok 4.3 Fast",
        family="Grok",
        description="Default-speed Grok configuration for interactive games.",
        speed_tier="fast",
        cost_tier="medium",
        tags=("default", "interactive"),
        sort_order=10,
        selectable=True,
        experimental=False,
        acceptable_resolved_models=("grok-4.3",),
    ),
    "reasoning": ModelSpec(
        alias="reasoning",
        provider="xai",
        model="grok-4.3",
        reasoning_effort="low",
        display_name="Grok 4.3 Reasoning",
        family="Grok",
        description="Grok 4.3 with low reasoning effort configured by default.",
        speed_tier="medium",
        cost_tier="medium",
        tags=("reasoning",),
        sort_order=20,
        selectable=True,
        experimental=False,
        acceptable_resolved_models=("grok-4.3",),
    ),
    # Google does not return billed cost in API responses, so records from
    # these models carry cost_source=pricing_table_estimate (LiteLLM price
    # map), never provider_reported. On the free tier the actual bill is
    # $0 while the estimate reflects paid-tier rates.
    "gemini_flash_lite": ModelSpec(
        alias="gemini_flash_lite",
        provider="litellm",
        model="gemini/gemini-3.1-flash-lite",  # $0.25/$1.50 per 1M (paid tier)
        api_key_env=("GEMINI_API_KEY",),
        display_name="Gemini 3.1 Flash Lite",
        family="Gemini",
        description="Low-cost baseline useful for larger exploratory batches.",
        speed_tier="fast",
        cost_tier="low",
        tags=("baseline", "batch"),
        sort_order=30,
        selectable=True,
        experimental=False,
        acceptable_resolved_models=(
            "gemini/gemini-3.1-flash-lite", "gemini-3.1-flash-lite",
        ),
    ),
    # reasoning_effort="low" caps Gemini's thinking budget (LiteLLM maps it
    # to thinkingBudget); default dynamic thinking burned ~160 reasoning
    # tokens even for trivial outputs at $9/1M output.
    "gemini_flash": ModelSpec(
        alias="gemini_flash",
        provider="litellm",
        model="gemini/gemini-3.5-flash",  # $1.50/$9.00 per 1M (paid tier)
        api_key_env=("GEMINI_API_KEY",),
        reasoning_effort="low",
        display_name="Gemini 3.5 Flash",
        family="Gemini",
        description="General-purpose Gemini model with low reasoning effort by default.",
        speed_tier="fast",
        cost_tier="medium",
        tags=("reasoning", "interactive"),
        sort_order=40,
        selectable=True,
        experimental=False,
        acceptable_resolved_models=(
            "gemini/gemini-3.5-flash", "gemini-3.5-flash",
        ),
    ),
    # Anthropic (date-pinned where the API offers it, for reproducibility)
    "claude_haiku": ModelSpec(
        alias="claude_haiku",
        provider="litellm",
        model="anthropic/claude-haiku-4-5-20251001",  # $1.00/$5.00 per 1M
        api_key_env=("ANTHROPIC_API_KEY",),
        display_name="Claude Haiku 4.5",
        family="Claude",
        description="Date-pinned Anthropic model suited to lower-cost comparisons.",
        speed_tier="fast",
        cost_tier="medium",
        tags=("date-pinned",),
        sort_order=50,
        selectable=True,
        experimental=False,
        acceptable_resolved_models=(
            "anthropic/claude-haiku-4-5-20251001", "claude-haiku-4-5-20251001",
        ),
    ),
    "claude_sonnet": ModelSpec(
        alias="claude_sonnet",
        provider="litellm",
        model="anthropic/claude-sonnet-5",  # $2.00/$10.00 per 1M
        api_key_env=("ANTHROPIC_API_KEY",),
        display_name="Claude Sonnet 5",
        family="Claude",
        description="Anthropic Sonnet configuration for higher-capability comparisons.",
        speed_tier="medium",
        cost_tier="high",
        tags=("comparison",),
        sort_order=60,
        selectable=True,
        experimental=False,
        acceptable_resolved_models=(
            "anthropic/claude-sonnet-5", "claude-sonnet-5",
        ),
    ),
    # OpenAI (per developers.openai.com/api/docs/models, 2026-07)
    "gpt_nano": ModelSpec(
        alias="gpt_nano",
        provider="litellm",
        model="openai/gpt-5.4-nano-2026-03-17",  # $0.20/$1.25 per 1M, date-pinned
        api_key_env=("OPENAI_API_KEY",),
        display_name="GPT-5.4 Nano",
        family="GPT",
        description="Date-pinned low-cost OpenAI baseline.",
        speed_tier="fast",
        cost_tier="low",
        tags=("baseline", "date-pinned"),
        sort_order=70,
        selectable=True,
        experimental=False,
        acceptable_resolved_models=(
            "openai/gpt-5.4-nano-2026-03-17", "gpt-5.4-nano-2026-03-17",
        ),
    ),
    "gpt_luna": ModelSpec(
        alias="gpt_luna",
        provider="litellm",
        model="openai/gpt-5.6-luna",  # $1.00/$6.00 per 1M, reasoning model
        api_key_env=("OPENAI_API_KEY",),
        reasoning_effort="low",  # cap thinking cost, as with gemini_flash
        display_name="GPT-5.6 Luna",
        family="GPT",
        description="OpenAI reasoning model with low effort configured by default.",
        speed_tier="medium",
        cost_tier="medium",
        tags=("reasoning",),
        sort_order=80,
        selectable=True,
        experimental=True,
        acceptable_resolved_models=("openai/gpt-5.6-luna", "gpt-5.6-luna"),
    ),
}


# LiteLLM-style provider prefixes -> key env var. A bare model name with
# no prefix is treated as xAI (historical CLI behavior).
_PREFIX_KEY_ENV = {
    "gemini": ("GEMINI_API_KEY",),
    "openai": ("OPENAI_API_KEY",),
    "anthropic": ("ANTHROPIC_API_KEY",),
    "openrouter": ("OPENROUTER_API_KEY",),
}


def resolve(name: str) -> ModelSpec:
    """Resolve an alias; pass full model IDs through. Prefixed IDs
    (e.g. 'gemini/<model>') route to the LiteLLM provider; bare IDs
    (e.g. 'grok-4.5') keep the historical direct-xAI behavior."""
    if name in MODEL_REGISTRY:
        return MODEL_REGISTRY[name]
    if "/" in name:
        prefix = name.split("/", 1)[0]
        return ModelSpec(
            alias=None,
            provider="litellm",
            model=name,
            api_key_env=_PREFIX_KEY_ENV.get(prefix, ()),
        )
    return ModelSpec(alias=None, provider="xai", model=name)


def get_api_key(spec: ModelSpec) -> str:
    """Return the first non-empty key from the spec's env vars, or ''.

    The key is returned for constructing a provider client only; it must
    never be logged, stored on records, or attached to exceptions.
    """
    for env_name in spec.api_key_env:
        value = os.environ.get(env_name, "")
        if value:
            return value
    return ""


def _safe_initialization_error(exc: Exception, secret: str = "") -> str:
    """Return a bounded message without echoing credentials."""
    text = str(exc).replace("\n", " ").strip()
    if secret:
        text = text.replace(secret, "[REDACTED]")
    return text[:300] or exc.__class__.__name__


def build_provider(spec: ModelSpec, api_key: Optional[str] = None) -> ProviderBuildResult:
    """Construct a provider and preserve why construction was unavailable."""
    key = api_key if api_key is not None else get_api_key(spec)
    if not key:
        return ProviderBuildResult(
            status=ProviderBuildStatus.MISSING_CREDENTIAL,
            error="No configured credential was found.",
            required_credentials=spec.api_key_env,
        )

    if spec.provider == "xai":
        try:
            from werewolf.llm.xai_provider import XAIProvider
            provider = XAIProvider(api_key=key)
        except (ImportError, ModuleNotFoundError) as exc:
            return ProviderBuildResult(
                status=ProviderBuildStatus.DEPENDENCY_UNAVAILABLE,
                error=_safe_initialization_error(exc, key),
                required_credentials=spec.api_key_env,
            )
        except RuntimeError as exc:
            status = (ProviderBuildStatus.DEPENDENCY_UNAVAILABLE
                      if "not installed" in str(exc).lower()
                      else ProviderBuildStatus.INITIALIZATION_FAILED)
            return ProviderBuildResult(
                status=status, error=_safe_initialization_error(exc, key),
                required_credentials=spec.api_key_env,
            )
        except Exception as exc:
            return ProviderBuildResult(
                status=ProviderBuildStatus.INITIALIZATION_FAILED,
                error=_safe_initialization_error(exc, key),
                required_credentials=spec.api_key_env,
            )
    elif spec.provider == "litellm":
        try:
            from werewolf.llm.litellm_provider import LiteLLMProvider
            provider = LiteLLMProvider(api_key=key)
        except (ImportError, ModuleNotFoundError) as exc:
            return ProviderBuildResult(
                status=ProviderBuildStatus.DEPENDENCY_UNAVAILABLE,
                error=_safe_initialization_error(exc, key),
                required_credentials=spec.api_key_env,
            )
        except RuntimeError as exc:
            status = (ProviderBuildStatus.DEPENDENCY_UNAVAILABLE
                      if "not installed" in str(exc).lower()
                      else ProviderBuildStatus.INITIALIZATION_FAILED)
            return ProviderBuildResult(
                status=status, error=_safe_initialization_error(exc, key),
                required_credentials=spec.api_key_env,
            )
        except Exception as exc:
            return ProviderBuildResult(
                status=ProviderBuildStatus.INITIALIZATION_FAILED,
                error=_safe_initialization_error(exc, key),
                required_credentials=spec.api_key_env,
            )
    else:
        return ProviderBuildResult(
            status=ProviderBuildStatus.INITIALIZATION_FAILED,
            error=f"Unknown provider: {spec.provider}",
            required_credentials=spec.api_key_env,
        )
    return ProviderBuildResult(
        provider=provider,
        status=ProviderBuildStatus.READY,
        required_credentials=spec.api_key_env,
    )


def effective_generation_config(
    requested: GenerationConfig,
    model_spec: ModelSpec,
    reasoning_override: Optional[str] = None,
) -> GenerationConfig:
    """Apply the sole reasoning precedence policy used by every surface."""
    effort = (reasoning_override
              if reasoning_override is not None
              else model_spec.reasoning_effort)
    return replace(requested, reasoning_effort=effort)


def resolved_model_matches(spec: ModelSpec, resolved: Optional[str]) -> bool:
    if not resolved:
        return False
    if resolved == spec.model or resolved in spec.acceptable_resolved_models:
        return True
    return any(resolved.startswith(prefix) for prefix in spec.resolved_model_prefixes)


def selectable_models() -> list[ModelSpec]:
    return sorted(
        (spec for spec in MODEL_REGISTRY.values() if spec.selectable),
        key=lambda spec: (spec.sort_order, spec.alias or ""),
    )


def registry_snapshot() -> dict:
    """Serializable snapshot of the registry for experiment config logs.
    Contains env-var NAMES only, never values."""
    return {
        alias: {
            "provider": spec.provider,
            "model": spec.model,
            "api_key_env": list(spec.api_key_env),
            "reasoning_effort": spec.reasoning_effort,
            "acceptable_resolved_models": list(spec.acceptable_resolved_models),
            "resolved_model_prefixes": list(spec.resolved_model_prefixes),
        }
        for alias, spec in MODEL_REGISTRY.items()
    }
