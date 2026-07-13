"""Validated web-facing services for catalog games and provider checks."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Optional

from werewolf.engine.game import GameEngine
from werewolf.llm.provider import GenerationConfig, ModelRequest
from werewolf.llm.registry import (
    MODEL_REGISTRY,
    ModelSpec,
    ProviderBuildResult,
    ProviderBuildStatus,
    build_provider,
    effective_generation_config,
    resolved_model_matches,
)


class RequestValidationError(ValueError):
    def __init__(self, errors: dict[str, dict[str, str]]):
        super().__init__("Request validation failed")
        self.errors = errors


@dataclass(frozen=True)
class ParsedGameRequest:
    n_players: int
    n_wolves: int
    n_seers: int
    seed: int
    model: Optional[str]
    role_models: Optional[dict[str, str]]
    generation: GenerationConfig
    reasoning_override: Optional[str]
    discussion_cycles: int
    belief_snapshots: bool


_GENERATION_FIELDS = {
    "temperature", "top_p", "max_output_tokens", "provider_seed",
    "structured_output",
}
_REASONING_VALUES = {"none", "low", "medium", "high"}


def _error(code: str, message: str) -> dict[str, str]:
    return {"code": code, "message": message}


def _int_value(data: dict, key: str, default: int, errors: dict) -> int:
    value = data.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        errors[key] = _error("invalid_type", f"{key} must be an integer")
        return default
    return value


def parse_generation_settings(data: dict) -> tuple[GenerationConfig, Optional[str]]:
    errors: dict[str, dict[str, str]] = {}
    raw = data.get("generation_config", {})
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise RequestValidationError({
            "generation_config": _error("invalid_type", "generation_config must be an object")
        })
    if "reasoning_effort" in raw:
        errors["generation_config.reasoning_effort"] = _error(
            "duplicate_reasoning_input",
            "Use reasoning_override; reasoning_effort is not accepted here.",
        )
    for key in sorted(set(raw) - _GENERATION_FIELDS - {"reasoning_effort"}):
        errors[f"generation_config.{key}"] = _error(
            "unknown_field", f"Unknown generation setting: {key}",
        )

    def optional_number(key: str, *, minimum=None, maximum=None, integer=False):
        value = raw.get(key)
        if value is None:
            return None
        expected = int if integer else (int, float)
        if isinstance(value, bool) or not isinstance(value, expected):
            errors[f"generation_config.{key}"] = _error(
                "invalid_type", f"{key} must be {'an integer' if integer else 'a number'} or null",
            )
            return None
        if isinstance(value, float) and not math.isfinite(value):
            errors[f"generation_config.{key}"] = _error(
                "invalid_value", f"{key} must be finite",
            )
            return None
        if minimum is not None and value < minimum or maximum is not None and value > maximum:
            errors[f"generation_config.{key}"] = _error(
                "out_of_range", f"{key} must be between {minimum} and {maximum}",
            )
            return None
        return value

    structured = raw.get("structured_output", False)
    if not isinstance(structured, bool):
        errors["generation_config.structured_output"] = _error(
            "invalid_type", "structured_output must be a boolean",
        )
        structured = False

    override = data.get("reasoning_override")
    if override is not None and (
        not isinstance(override, str) or override not in _REASONING_VALUES
    ):
        errors["reasoning_override"] = _error(
            "invalid_value", "reasoning_override must be null, none, low, medium, or high",
        )
        override = None

    config = GenerationConfig(
        temperature=optional_number("temperature", minimum=0, maximum=2),
        top_p=optional_number("top_p", minimum=0, maximum=1),
        max_output_tokens=optional_number(
            "max_output_tokens", minimum=1, maximum=65536, integer=True,
        ),
        provider_seed=optional_number(
            "provider_seed", minimum=-(2**31), maximum=2**31 - 1, integer=True,
        ),
        structured_output=structured,
    )
    if errors:
        raise RequestValidationError(errors)
    return config, override


def _selectable_spec(alias: Any, field: str, errors: dict) -> Optional[ModelSpec]:
    if not isinstance(alias, str) or alias not in MODEL_REGISTRY:
        errors[field] = _error("invalid_model", "Choose a configured model alias")
        return None
    spec = MODEL_REGISTRY[alias]
    if not spec.selectable:
        errors[field] = _error("model_not_selectable", "This model is not selectable")
        return None
    return spec


def parse_game_request(data: Any) -> ParsedGameRequest:
    if not isinstance(data, dict):
        raise RequestValidationError({"request": _error("invalid_type", "JSON object required")})
    errors: dict[str, dict[str, str]] = {}
    n_players = _int_value(data, "n_players", 7, errors)
    n_wolves = _int_value(data, "n_wolves", 2, errors)
    n_seers = _int_value(data, "n_seers", 1, errors)
    seed = _int_value(data, "seed", 42, errors)
    discussion_cycles = _int_value(data, "discussion_cycles", 2, errors)
    snapshots = data.get("belief_snapshots", True)
    if not isinstance(snapshots, bool):
        errors["belief_snapshots"] = _error("invalid_type", "belief_snapshots must be a boolean")
        snapshots = True

    if not 4 <= n_players <= 15:
        errors["n_players"] = _error("out_of_range", "n_players must be between 4 and 15")
    if not 1 <= n_wolves <= 5:
        errors["n_wolves"] = _error("out_of_range", "n_wolves must be between 1 and 5")
    if n_seers not in (0, 1):
        errors["n_seers"] = _error("out_of_range", "n_seers must be 0 or 1")
    if n_wolves >= n_players or n_wolves + n_seers >= n_players:
        errors["roles"] = _error("invalid_role_counts", "At least one ordinary villager is required")
    if not 1 <= discussion_cycles <= 5:
        errors["discussion_cycles"] = _error("out_of_range", "discussion_cycles must be between 1 and 5")

    has_model = "model" in data
    has_roles = "role_models" in data
    model: Optional[str] = data.get("model", "fast") if not has_roles else None
    role_models: Optional[dict[str, str]] = None
    if has_model and has_roles:
        errors["model"] = _error("conflicting_models", "Provide model or role_models, not both")
    if has_roles:
        raw_roles = data.get("role_models")
        if not isinstance(raw_roles, dict):
            errors["role_models"] = _error("invalid_type", "role_models must be an object")
        else:
            required = {"werewolf", "villager", "seer"}
            if set(raw_roles) != required:
                errors["role_models"] = _error(
                    "incomplete_roles", "role_models must contain werewolf, villager, and seer",
                )
            else:
                role_models = dict(raw_roles)
                if n_seers == 0:
                    role_models["seer"] = role_models["villager"]
                for role, alias in role_models.items():
                    _selectable_spec(alias, f"role_models.{role}", errors)
    else:
        _selectable_spec(model, "model", errors)

    try:
        generation, override = parse_generation_settings(data)
    except RequestValidationError as exc:
        errors.update(exc.errors)
        generation, override = GenerationConfig(), None
    if errors:
        raise RequestValidationError(errors)
    return ParsedGameRequest(
        n_players=n_players, n_wolves=n_wolves, n_seers=n_seers, seed=seed,
        model=model, role_models=role_models, generation=generation,
        reasoning_override=override, discussion_cycles=discussion_cycles,
        belief_snapshots=snapshots,
    )


def _provider_error(result: ProviderBuildResult, spec: ModelSpec) -> dict[str, str]:
    if result.status == ProviderBuildStatus.MISSING_CREDENTIAL:
        names = " or ".join(spec.api_key_env) or "provider credential"
        return _error("missing_key", f"{names} is not configured")
    return _error("provider_unavailable", result.error or "Provider could not be initialized")


def create_engine_from_payload(data: Any) -> GameEngine:
    parsed = parse_game_request(data)
    aliases = ([parsed.model] if parsed.model else list(dict.fromkeys(parsed.role_models.values())))
    builds: dict[str, ProviderBuildResult] = {}
    errors: dict[str, dict[str, str]] = {}
    for alias in aliases:
        spec = MODEL_REGISTRY[alias]
        result = build_provider(spec)
        builds[alias] = result
        if not result.ok:
            if parsed.role_models:
                for role, role_alias in parsed.role_models.items():
                    if role_alias == alias:
                        errors[f"role_models.{role}"] = _provider_error(result, spec)
            else:
                errors["model"] = _provider_error(result, spec)
    if errors:
        raise RequestValidationError(errors)

    common = dict(
        n_players=parsed.n_players, n_wolves=parsed.n_wolves,
        n_seers=parsed.n_seers, seed=parsed.seed,
        transcript_enabled=False, show_all_channels=True, show_prompts=False,
        belief_snapshots=parsed.belief_snapshots,
        generation_config=parsed.generation,
        reasoning_override=parsed.reasoning_override,
        discussion_cycles=parsed.discussion_cycles,
    )
    if parsed.role_models:
        role_providers = {
            role: builds[alias].provider for role, alias in parsed.role_models.items()
        }
        return GameEngine(
            **common, api_key="", model=parsed.role_models["villager"],
            role_models=parsed.role_models, role_providers=role_providers,
        )
    spec = MODEL_REGISTRY[parsed.model]
    return GameEngine(
        **common, api_key="", model=spec.model, provider=builds[parsed.model].provider,
        model_alias=spec.alias,
    )


def health_check(alias: str, data: Any) -> tuple[dict, int]:
    errors: dict[str, dict[str, str]] = {}
    spec = _selectable_spec(alias, "model", errors)
    if errors:
        return {"status": "failed", "errors": errors}, 404
    try:
        generation, override = parse_generation_settings(data or {})
    except RequestValidationError as exc:
        return {"status": "failed", "errors": exc.errors}, 400
    build = build_provider(spec)
    if not build.ok:
        status = ("missing_key" if build.status == ProviderBuildStatus.MISSING_CREDENTIAL
                  else "provider_unavailable")
        return {
            "status": status,
            "checks": None,
            "error": _provider_error(build, spec),
            "required_credentials": list(spec.api_key_env),
        }, 200

    effective = effective_generation_config(generation, spec, override)
    result = build.provider.complete(ModelRequest(
        model=spec.model,
        system_prompt="Return one JSON object only.",
        user_prompt='Return exactly this JSON object: {"health": "ok"}',
        generation=effective,
    ))
    text = str(result.text or "").strip()
    json_valid = False
    if text:
        try:
            parsed = json.loads(text)
            json_valid = (
                isinstance(parsed, dict) and parsed.get("health") == "ok"
            )
        except (TypeError, ValueError):
            pass
    if not result.resolved_model:
        model_identity = "unreported"
    elif resolved_model_matches(spec, result.resolved_model):
        model_identity = "matched"
    else:
        model_identity = "mismatched"
    metadata = result.provider_metadata or {}
    dropped = list(metadata.get("generation_dropped") or [])
    adjusted = list(metadata.get("generation_adjusted") or [])
    adjustments_detected = bool(dropped or adjusted)
    checks = {
        "api_ok": bool(result.ok),
        "nonempty_output": bool(text),
        "json_valid": json_valid,
        "model_match": (
            True if model_identity == "matched"
            else False if model_identity == "mismatched"
            else None
        ),
        "model_identity": model_identity,
        "generation_adjustments_detected": adjustments_detected,
    }
    usable = all((
        checks["api_ok"], checks["nonempty_output"], json_valid,
        model_identity != "mismatched",
    ))
    needs_adjustment = adjustments_detected or model_identity == "unreported"
    status = "adjusted" if usable and needs_adjustment else "ready" if usable else "failed"
    warnings = []
    if model_identity == "unreported":
        warnings.append("Provider did not report the resolved model identity.")
    return {
        "status": status,
        "checks": checks,
        "alias": alias,
        "requested_model": spec.model,
        "resolved_model": result.resolved_model,
        "effective_generation": effective.to_json_dict(),
        "generation_dropped": dropped,
        "generation_adjusted": adjusted,
        "warnings": warnings,
        "latency_ms": result.latency_ms,
        "finish_reason": result.finish_reason,
        "usage": result.usage.to_json_dict(),
        "cost": result.cost.to_json_dict(),
        "error": ({
            "code": result.error_category.value if result.error_category else "health_check_failed",
            "message": result.error_message or "Health check did not pass",
        } if status == "failed" else None),
    }, 200
