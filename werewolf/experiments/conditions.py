"""Experiment conditions: explicit role-model assignments per condition.

Arbitrary explicit conditions are the primitive; the crossed A/B helper
is sugar producing the standard producer-target matrix. Every condition
fully materializes its role -> model assignment, so the manifest never
depends on runtime defaults.
"""
from __future__ import annotations

import re

from werewolf.llm.registry import resolve

ROLE_NAMES = ("werewolf", "villager", "seer")

_CONDITION_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_\-]{0,63}$")


class ConditionError(ValueError):
    pass


def normalize_conditions(conditions) -> dict:
    """Validate and normalize {condition_id: {"role_models": {...}}}."""
    if not isinstance(conditions, dict) or not conditions:
        raise ConditionError("conditions must be a non-empty object")
    normalized = {}
    for condition_id, condition in conditions.items():
        if not isinstance(condition_id, str) or not _CONDITION_ID_RE.match(
            condition_id
        ):
            raise ConditionError(
                f"Invalid condition_id {condition_id!r}: use "
                "lowercase [a-z0-9_-], max 64 chars"
            )
        if not isinstance(condition, dict):
            raise ConditionError(f"Condition {condition_id} must be an object")
        role_models = condition.get("role_models")
        if not isinstance(role_models, dict):
            raise ConditionError(
                f"Condition {condition_id} requires a role_models object"
            )
        if set(role_models) != set(ROLE_NAMES):
            raise ConditionError(
                f"Condition {condition_id} role_models must assign exactly "
                f"{list(ROLE_NAMES)}"
            )
        for role, model in role_models.items():
            if not isinstance(model, str) or not model:
                raise ConditionError(
                    f"Condition {condition_id} role {role} needs a model "
                    "alias or ID string"
                )
        unknown = set(condition) - {"role_models", "description"}
        if unknown:
            raise ConditionError(
                f"Condition {condition_id} has unknown keys: {sorted(unknown)}"
            )
        normalized[condition_id] = {
            "role_models": {role: role_models[role] for role in ROLE_NAMES},
        }
        if isinstance(condition.get("description"), str):
            normalized[condition_id]["description"] = condition["description"]
    return normalized


def build_crossed_conditions(model_a: str, model_b: str) -> dict:
    """The 2x2 producer-target matrix over two models. The seer detects
    for the village, so it always plays the village-side model."""
    if model_a == model_b:
        raise ConditionError(
            "Crossed experiments need two distinct models"
        )

    def assignment(wolves: str, village: str) -> dict:
        return {"role_models": {
            "werewolf": wolves, "villager": village, "seer": village,
        }}

    return normalize_conditions({
        "a_homogeneous": assignment(model_a, model_a),
        "b_homogeneous": assignment(model_b, model_b),
        "a_wolves_b_village": assignment(model_a, model_b),
        "b_wolves_a_village": assignment(model_b, model_a),
    })


def condition_models(conditions: dict) -> list:
    """Unique model names referenced by any condition, sorted."""
    models = set()
    for condition in conditions.values():
        models.update(condition["role_models"].values())
    return sorted(models)


def model_catalog(conditions: dict) -> dict:
    """Manifest `models` block: requested identity and provider mapping
    for every referenced model (env-var NAMES only, never values)."""
    catalog = {}
    for name in condition_models(conditions):
        spec = resolve(name)
        catalog[name] = {
            "alias": spec.alias,
            "requested_model": spec.model,
            "provider": spec.provider,
            "registry_reasoning_default": spec.reasoning_effort,
            "api_key_env": list(spec.api_key_env),
            "acceptable_resolved_models": list(spec.acceptable_resolved_models),
        }
    return catalog
