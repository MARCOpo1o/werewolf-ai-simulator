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


# Mirrors the historical MODEL_PRESETS in cli/run_game.py. Updating these
# slugs (e.g. to grok-4.3 after the May 15, 2026 retirement) is a separate,
# deliberate commit because it affects research comparability.
MODEL_REGISTRY: dict[str, ModelSpec] = {
    "fast": ModelSpec(
        alias="fast",
        provider="xai",
        model="grok-4-1-fast",
    ),
    "reasoning": ModelSpec(
        alias="reasoning",
        provider="xai",
        model="grok-4-1-fast-reasoning",
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
