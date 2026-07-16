"""Fail-closed health preflight for formal experiment sessions.

One probe per unique (model, effective generation) fingerprint runs at
the start of every execution session that has runnable trials. Every
probe's result and operational cost is journaled, whether or not the
session proceeds.

Formal execution accepts `ready`, and `adjusted` only when the detected
adjustment's fingerprint exactly matches an adjustment predeclared in
the execution contract AND the operator passed --allow-adjusted-health.
Unknown or changed adjustments always block: an experiment must never
silently run under provider behavior its manifest did not declare.
"""
from __future__ import annotations

import json
from dataclasses import replace
from typing import Optional

from werewolf.experiments.canonical import jcs_sha256
from werewolf.experiments.journal import sanitize_error
from werewolf.llm.provider import GenerationConfig, ModelRequest
from werewolf.llm.records import CostSource
from werewolf.llm.registry import (
    ProviderBuildStatus,
    build_provider,
    effective_generation_config,
    resolve,
    resolved_model_matches,
)

HEALTH_SYSTEM_PROMPT = "Return one JSON object only."
HEALTH_USER_PROMPT = 'Return exactly this JSON object: {"health": "ok"}'


class HealthPolicyError(RuntimeError):
    pass


def generation_from_dict(raw: Optional[dict]) -> GenerationConfig:
    raw = raw or {}
    return GenerationConfig(
        temperature=raw.get("temperature"),
        top_p=raw.get("top_p"),
        max_output_tokens=raw.get("max_output_tokens"),
        reasoning_effort=raw.get("reasoning_effort"),
        provider_seed=raw.get("provider_seed"),
        structured_output=bool(raw.get("structured_output", False)),
    )


def health_fingerprint(model_name: str, effective: GenerationConfig) -> str:
    spec = resolve(model_name)
    return jcs_sha256({
        "model_alias": spec.alias,
        "requested_model": spec.model,
        "provider": spec.provider,
        "effective_generation": effective.to_json_dict(),
    })


def adjustment_fingerprint(
    *, generation_dropped, generation_adjusted, model_identity: str
) -> str:
    """Identity of a detected provider adjustment. Predeclared
    adjustments in the manifest carry exactly these fingerprints."""
    return jcs_sha256({
        "generation_dropped": sorted(str(x) for x in generation_dropped),
        "generation_adjusted": sorted(str(x) for x in generation_adjusted),
        "model_identity": model_identity,
    })


def unique_health_targets(
    conditions: dict, generation: Optional[dict], *,
    request_timeout_seconds: int = 120,
) -> list:
    """One target per unique model/effective-generation fingerprint."""
    requested = generation_from_dict(generation)
    targets = {}
    for condition in conditions.values():
        for model_name in condition["role_models"].values():
            spec = resolve(model_name)
            effective = effective_generation_config(requested, spec)
            fingerprint = health_fingerprint(model_name, effective)
            targets.setdefault(fingerprint, {
                "health_fingerprint": fingerprint,
                "model_name": model_name,
                "model_alias": spec.alias,
                "requested_model": spec.model,
                "provider": spec.provider,
                "effective_generation": effective.to_json_dict(),
                "request_timeout_seconds": request_timeout_seconds,
            })
    return [targets[key] for key in sorted(targets)]


def _cost_completeness(cost) -> str:
    source = getattr(cost, "source", None)
    if source == CostSource.PROVIDER_REPORTED:
        return "provider_reported"
    if source in (CostSource.PRICING_TABLE_ESTIMATE,
                  CostSource.TOKENIZER_ESTIMATE):
        return "estimated"
    return "unavailable"


def probe_model(target: dict, *, provider=None) -> dict:
    """Run one health probe and return journal-ready health_check
    fields. Never raises for provider-level failures."""
    spec = resolve(target["model_name"])
    effective = generation_from_dict(target["effective_generation"])
    base = {
        "health_fingerprint": target["health_fingerprint"],
        "model_alias": spec.alias,
        "requested_model": spec.model,
        "provider": spec.provider,
        "effective_generation": effective.to_json_dict(),
        "adjustments": {"generation_dropped": [],
                        "generation_adjusted": [],
                        "model_identity": "unknown"},
        "latency_ms": None,
        "cost": None,
        "cost_completeness": "unavailable",
        "sanitized_error": None,
    }
    if provider is None:
        build = build_provider(
            spec, timeout=target.get("request_timeout_seconds", 120),
        )
        if not build.ok:
            status = (
                "missing_key"
                if build.status == ProviderBuildStatus.MISSING_CREDENTIAL
                else "provider_unavailable"
            )
            return {
                **base,
                "status": status,
                "sanitized_error": sanitize_error(
                    build.error or "provider unavailable"
                ),
            }
        provider = build.provider

    result = provider.complete(ModelRequest(
        model=spec.model,
        system_prompt=HEALTH_SYSTEM_PROMPT,
        user_prompt=HEALTH_USER_PROMPT,
        generation=effective,
    ))
    text = str(result.text or "").strip()
    json_valid = False
    if text:
        try:
            parsed = json.loads(text)
            json_valid = isinstance(parsed, dict) and parsed.get("health") == "ok"
        except ValueError:
            pass
    if not result.resolved_model:
        model_identity = "unreported"
    elif resolved_model_matches(spec, result.resolved_model):
        model_identity = "matched"
    else:
        model_identity = "mismatched"
    metadata = result.provider_metadata or {}
    dropped = [str(x) for x in (metadata.get("generation_dropped") or [])]
    adjusted = [str(x) for x in (metadata.get("generation_adjusted") or [])]
    usable = bool(result.ok) and bool(text) and json_valid \
        and model_identity != "mismatched"
    needs_adjustment = bool(dropped or adjusted) \
        or model_identity == "unreported"
    status = ("adjusted" if usable and needs_adjustment
              else "ready" if usable else "failed")
    record = {
        **base,
        "status": status,
        "resolved_model": result.resolved_model,
        "adjustments": {
            "generation_dropped": sorted(dropped),
            "generation_adjusted": sorted(adjusted),
            "model_identity": model_identity,
        },
        "latency_ms": result.latency_ms,
        "cost": result.cost.to_json_dict() if result.cost else None,
        "cost_completeness": _cost_completeness(result.cost),
        "sanitized_error": (
            sanitize_error(result.error_message or "health probe failed")
            if not result.ok else None
        ),
    }
    if status == "adjusted":
        record["adjustment_fingerprint"] = adjustment_fingerprint(
            generation_dropped=dropped,
            generation_adjusted=adjusted,
            model_identity=model_identity,
        )
    return record


def evaluate_health_record(
    record: dict,
    *,
    predeclared_fingerprints,
    allow_adjusted: bool,
) -> Optional[str]:
    """Returns None when formal execution may proceed, else the reason
    it must block."""
    status = record.get("status")
    if status == "ready":
        return None
    identity = (
        record.get("model_alias") or record.get("requested_model") or "?"
    )
    if status == "adjusted":
        fingerprint = record.get("adjustment_fingerprint")
        declared = fingerprint in set(predeclared_fingerprints or [])
        if not declared:
            return (
                f"Model {identity} reported an adjustment that is not "
                "predeclared in the execution contract; declare it in a "
                "new manifest before running."
            )
        if not allow_adjusted:
            return (
                f"Model {identity} matches a predeclared adjustment; "
                "pass --allow-adjusted-health to accept it."
            )
        return None
    return (
        f"Model {identity} health check status is {status!r}: "
        f"{record.get('sanitized_error') or 'not usable for formal runs'}"
    )
