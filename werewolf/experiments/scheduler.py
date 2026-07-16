"""Deterministic trial scheduling.

The complete schedule is materialized before execution and stored in
the execution contract: every scheduled trial's identity and position
is fixed before the first paid request. Within each (seed, repetition)
block, condition order is deterministically randomized by hashing

    SHA-256(JCS({"scheduler_seed", "seed", "repetition", "condition_id"}))

so no condition systematically runs first (time-of-day and provider
warm-cache effects average out) while remaining reproducible.

Trial IDs come from an independently domain-separated canonical-object
digest, so they can never collide with schedule ordering keys derived
from the same fields.

"Deterministic" here applies to configuration and scheduling only;
LLM outputs remain nondeterministic.
"""
from __future__ import annotations

from werewolf.experiments.canonical import canonical_object_digest, jcs_sha256

SCHEDULER_VERSION = 1

_TRIAL_ID_DOMAIN = "werewolf-experiment-trial-id-v1"


class ScheduleError(ValueError):
    pass


def schedule_order_key(
    scheduler_seed: int, seed: int, repetition: int, condition_id: str
) -> str:
    return jcs_sha256({
        "scheduler_seed": scheduler_seed,
        "seed": seed,
        "repetition": repetition,
        "condition_id": condition_id,
    })


def trial_id(
    experiment_id: str, condition_id: str, seed: int, repetition: int
) -> str:
    digest = canonical_object_digest(_TRIAL_ID_DOMAIN, {
        "experiment_id": experiment_id,
        "condition_id": condition_id,
        "seed": seed,
        "repetition": repetition,
    })
    return f"trial_{digest[:32]}"


def materialize_schedule(
    *,
    experiment_id: str,
    conditions: dict,
    seeds: list,
    repetitions: int,
    scheduler_seed: int,
) -> list:
    """Every (seed, repetition, condition) exactly once, blocks in
    manifest seed order, conditions hash-ordered within each block."""
    if repetitions < 1:
        raise ScheduleError("repetitions must be >= 1")
    if len(set(seeds)) != len(seeds):
        raise ScheduleError("seeds must be unique")
    schedule = []
    trial_index = 0
    for seed in seeds:
        for repetition in range(repetitions):
            ordered = sorted(
                conditions,
                key=lambda condition_id: schedule_order_key(
                    scheduler_seed, seed, repetition, condition_id
                ),
            )
            for scheduler_position, condition_id in enumerate(ordered):
                schedule.append({
                    "trial_index": trial_index,
                    "scheduler_position": scheduler_position,
                    "trial_id": trial_id(
                        experiment_id, condition_id, seed, repetition
                    ),
                    "condition_id": condition_id,
                    "seed": seed,
                    "repetition": repetition,
                })
                trial_index += 1
    return schedule


def verify_schedule(manifest: dict) -> list:
    """Recompute the schedule from the manifest's own inputs and diff it
    against the stored materialization. Returns error strings."""
    execution = manifest.get("execution_contract") or {}
    scheduler = execution.get("scheduler") or {}
    if scheduler.get("version") != SCHEDULER_VERSION:
        return [
            f"scheduler version {scheduler.get('version')!r} is not "
            f"supported (current: {SCHEDULER_VERSION})"
        ]
    try:
        expected = materialize_schedule(
            experiment_id=manifest.get("experiment_id"),
            conditions=execution.get("conditions") or {},
            seeds=execution.get("seeds") or [],
            repetitions=execution.get("repetitions") or 0,
            scheduler_seed=scheduler.get("scheduler_seed"),
        )
    except (ScheduleError, TypeError) as exc:
        return [f"schedule could not be recomputed: {exc}"]
    stored = execution.get("schedule")
    if stored != expected:
        return [
            "stored schedule does not match deterministic materialization "
            f"({len(stored or [])} stored vs {len(expected)} recomputed "
            "entries)"
        ]
    return []
