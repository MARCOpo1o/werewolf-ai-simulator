"""Model registry: aliases -> provider, model ID, key env vars.

Secrets stay in environment variables; this module stores only the names
of those variables. Model IDs here are version-controlled experiment
configuration. NOTE: the resolved model reported back by the provider is
the ground truth (xAI silently redirects retired slugs), so UsageRecord
carries both requested_model and resolved_model.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class ModelSpec:
    alias: Optional[str]           # registry alias, or None for passthrough
    provider: str                  # "xai" (more later)
    model: str                     # provider model slug
    api_key_env: tuple[str, ...] = ("GROK_API_KEY", "XAI_API_KEY")
    reasoning_effort: Optional[str] = None  # xAI: none|low|medium|high


# NOTE (research comparability): before 2026-07 these aliases pointed at
# grok-4-1-fast / grok-4-1-fast-reasoning, which xAI retired on
# 2026-05-15 and silently redirects to grok-4.3. The aliases now target
# grok-4.3 explicitly, matching xAI's own redirect mapping (none/low
# reasoning effort), so requested_model == actual model again. Results
# logged before this change were served by whatever the redirect chose;
# compare eras via the resolved_model field, not the alias.
MODEL_REGISTRY: dict[str, ModelSpec] = {
    "fast": ModelSpec(
        alias="fast",
        provider="xai",
        model="grok-4.3",
        reasoning_effort="none",
    ),
    "reasoning": ModelSpec(
        alias="reasoning",
        provider="xai",
        model="grok-4.3",
        reasoning_effort="low",
    ),
    # Google does not return billed cost in API responses, so records from
    # this model carry cost_source=pricing_table_estimate (LiteLLM price
    # map), never provider_reported.
    "gemini_flash_lite": ModelSpec(
        alias="gemini_flash_lite",
        provider="litellm",
        model="gemini/gemini-3.1-flash-lite-preview",
        api_key_env=("GEMINI_API_KEY",),
    ),
}


def resolve(name: str) -> ModelSpec:
    """Resolve an alias, or pass a full model ID through as an xAI spec
    (preserves the current CLI behavior of accepting full model names)."""
    if name in MODEL_REGISTRY:
        return MODEL_REGISTRY[name]
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


def build_provider(spec: ModelSpec, api_key: Optional[str] = None):
    """Construct a Provider for the spec, or None when no key/SDK is
    available (callers then use the existing random-fallback path).

    The key is passed straight into the provider client and is not
    retained by this module.
    """
    key = api_key if api_key is not None else get_api_key(spec)
    if not key:
        return None

    if spec.provider == "xai":
        try:
            from werewolf.llm.xai_provider import XAIProvider
            return XAIProvider(api_key=key)
        except RuntimeError:  # xai-sdk not installed
            return None
    if spec.provider == "litellm":
        try:
            from werewolf.llm.litellm_provider import LiteLLMProvider
            return LiteLLMProvider(api_key=key)
        except (ImportError, RuntimeError):
            return None
    raise ValueError(f"Unknown provider: {spec.provider}")


def registry_snapshot() -> dict:
    """Serializable snapshot of the registry for experiment config logs.
    Contains env-var NAMES only, never values."""
    return {
        alias: {
            "provider": spec.provider,
            "model": spec.model,
            "api_key_env": list(spec.api_key_env),
            "reasoning_effort": spec.reasoning_effort,
        }
        for alias, spec in MODEL_REGISTRY.items()
    }
