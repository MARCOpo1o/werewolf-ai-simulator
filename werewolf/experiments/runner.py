"""Experiment manifest construction and the formal execution runner.

Execution is CLI-only, sequential, and fail-closed:

- The manifest is validated (hash, schedule, prompt-profile drift,
  pinned policy values) before anything runs.
- The current execution runtime hash must equal the pinned one; an
  execution-relevant code change blocks resume, while analysis-only
  changes never do.
- One health probe per model/generation fingerprint gates every
  session that has runnable trials; results and costs are journaled
  regardless of outcome.
- The game ID is journaled in trial_started before the first provider
  request; crash reconciliation uses the execution-side verifier only.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from werewolf.agents.ai_agent import MAX_RETRIES
from werewolf.engine.limits import limits_dict
from werewolf.experiments import manifest as manifest_store
from werewolf.experiments.aggregate import (
    AGGREGATE_ANALYSIS_VERSION,
    COMPARISON_METHOD_VERSION,
    DEFAULT_BOOTSTRAP,
    METRIC_WEIGHTING,
)
from werewolf.experiments.conditions import model_catalog, normalize_conditions
from werewolf.experiments.health import (
    evaluate_health_record,
    probe_model,
    unique_health_targets,
)
from werewolf.experiments.journal import (
    JournalWriter,
    SOURCE_MISSING,
    SOURCE_RECORDED,
    TRIAL_COMPLETED,
    TRIAL_FAILED,
    TRIAL_INTERRUPTED,
    TRIAL_STARTED,
    sanitize_error,
)
from werewolf.experiments.locks import execution_lock
from werewolf.experiments.manifest import (
    MANIFEST_SCHEMA_VERSION,
    ManifestError,
    execution_contract_sha256,
    finalize_manifest,
    load_verified_manifest,
)
from werewolf.experiments.profiles import (
    resolve_prompt_profile,
    verify_prompt_profile,
)
from werewolf.experiments.runtime_hash import (
    analysis_runtime_hash,
    execution_runtime_hash,
    repository_commit,
)
from werewolf.experiments.scheduler import (
    SCHEDULER_VERSION,
    materialize_schedule,
    verify_schedule,
)
from werewolf.experiments.verifier import reconcile_attempt_source

# Formal execution defaults, pinned verbatim into every manifest's
# policies block. Values that mirror code constants are re-checked at
# run time so a drifted constant can never silently ship.
FORMAL_EXECUTION_DEFAULTS = {
    "allow_provider_fallback": False,
    "action_failure_policy": "abort_game",
    "agent_action_max_attempts": 3,
    "request_timeout_seconds": 120,
    "retryable_errors": [
        "rate_limited", "timeout", "network_error", "provider_error",
    ],
    "retry_backoff": "none",
    "max_trial_attempts": 2,
    "max_rounds": 20,
    "intertrial_delay_seconds": 0,
    "public_message_limit": 800,
    "wolf_message_limit": 800,
    "memory_limit": 2500,
}

DEFAULT_GAME = {
    "n_players": 7,
    "n_wolves": 2,
    "n_seers": 1,
    "discussion_cycles": 2,
    "belief_snapshots": True,
}

DEFAULT_GENERATION = {
    "temperature": None,
    "top_p": None,
    "max_output_tokens": 4096,
    "reasoning_effort": None,
    "provider_seed": None,
    "structured_output": False,
}

_COMPARISON_KEYS = frozenset({
    "comparison_id", "condition_a", "condition_b", "metric_id",
    "analysis_view", "design", "effect", "direction",
})


class ExperimentRunError(RuntimeError):
    pass


class ExecutionRuntimeChanged(ExperimentRunError):
    """Execution-relevant code changed since the manifest was pinned."""


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_comparisons(comparisons, condition_ids) -> list:
    errors = []
    if not isinstance(comparisons, list):
        return ["comparisons must be a list"]
    seen = set()
    for i, comparison in enumerate(comparisons):
        if not isinstance(comparison, dict):
            errors.append(f"comparison #{i} must be an object")
            continue
        missing = _COMPARISON_KEYS - set(comparison)
        if missing:
            errors.append(f"comparison #{i} missing keys: {sorted(missing)}")
            continue
        unknown = set(comparison) - _COMPARISON_KEYS
        if unknown:
            errors.append(f"comparison #{i} unknown keys: {sorted(unknown)}")
        cid = comparison["comparison_id"]
        if cid in seen:
            errors.append(f"duplicate comparison_id {cid!r}")
        seen.add(cid)
        for side in ("condition_a", "condition_b"):
            if comparison[side] not in condition_ids:
                errors.append(
                    f"comparison {cid!r} references unknown condition "
                    f"{comparison[side]!r}"
                )
        if comparison["design"] not in ("paired", "independent"):
            errors.append(
                f"comparison {cid!r} design must be paired or independent"
            )
        if comparison["effect"] != "difference":
            errors.append(
                f"comparison {cid!r} effect must be 'difference' (v1)"
            )
        if comparison["direction"] not in ("a_minus_b", "b_minus_a"):
            errors.append(
                f"comparison {cid!r} direction must be a_minus_b or "
                "b_minus_a"
            )
    return errors


def default_crossed_comparisons() -> list:
    comparisons = []
    for metric_id in ("village_win_rate", "wolf_win_rate"):
        comparisons.append({
            "comparison_id": f"a_vs_b_clean_{metric_id}",
            "condition_a": "a_homogeneous",
            "condition_b": "b_homogeneous",
            "metric_id": metric_id,
            "analysis_view": "clean_eligible",
            "design": "paired",
            "effect": "difference",
            "direction": "a_minus_b",
        })
    comparisons.append({
        "comparison_id": "a_wolves_vs_b_wolves_crossed_clean_wolf_win_rate",
        "condition_a": "a_wolves_b_village",
        "condition_b": "b_wolves_a_village",
        "metric_id": "wolf_win_rate",
        "analysis_view": "clean_eligible",
        "design": "paired",
        "effect": "difference",
        "direction": "a_minus_b",
    })
    return comparisons


def build_experiment_manifest(
    *,
    experiment_id: str,
    conditions: dict,
    seeds: list,
    repetitions: int,
    description: str = "",
    game: Optional[dict] = None,
    generation: Optional[dict] = None,
    policies: Optional[dict] = None,
    predeclared_adjustments: Optional[list] = None,
    scheduler_seed: int = 0,
    comparisons: Optional[list] = None,
    bootstrap: Optional[dict] = None,
    prompt_profile: str = "baseline_v1",
) -> dict:
    """Assemble and finalize a canonical manifest from configuration."""
    from werewolf.evaluation.belief_metrics import METRICS_VERSION
    from werewolf.evaluation.validity import VALIDITY_POLICY_VERSION
    from werewolf.reporting.builder import REPORT_BUILD_VERSION

    conditions = normalize_conditions(conditions)
    game = {**DEFAULT_GAME, **(game or {})}
    generation = {**DEFAULT_GENERATION, **(generation or {})}
    policies = {**FORMAL_EXECUTION_DEFAULTS, **(policies or {})}
    unknown_policies = set(policies) - set(FORMAL_EXECUTION_DEFAULTS)
    if unknown_policies:
        raise ManifestError(
            f"Unknown policy keys: {sorted(unknown_policies)}"
        )
    predeclared = list(predeclared_adjustments or [])
    for entry in predeclared:
        if not isinstance(entry, dict) or not isinstance(
            entry.get("fingerprint"), str
        ) or not isinstance(entry.get("description"), str):
            raise ManifestError(
                "predeclared_adjustments entries need "
                "{description, fingerprint}"
            )
    comparisons = comparisons if comparisons is not None else []
    comparison_errors = validate_comparisons(comparisons, set(conditions))
    if comparison_errors:
        raise ManifestError(
            "Invalid comparisons: " + "; ".join(comparison_errors)
        )

    schedule = materialize_schedule(
        experiment_id=experiment_id,
        conditions=conditions,
        seeds=list(seeds),
        repetitions=repetitions,
        scheduler_seed=scheduler_seed,
    )
    manifest = {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "created_at": utc_now_iso(),
        "description": description,
        "execution_contract": {
            "conditions": conditions,
            "game": game,
            "seeds": list(seeds),
            "repetitions": repetitions,
            "prompt_profile": resolve_prompt_profile(prompt_profile),
            "models": model_catalog(conditions),
            "generation": generation,
            "predeclared_adjustments": predeclared,
            "policies": policies,
            "scheduler": {
                "version": SCHEDULER_VERSION,
                "scheduler_seed": scheduler_seed,
            },
            "schedule": schedule,
            "execution_runtime_hash": execution_runtime_hash(),
        },
        "analysis_contract": {
            "report_build_version": REPORT_BUILD_VERSION,
            "validity_policy_version": VALIDITY_POLICY_VERSION,
            "belief_metrics_version": METRICS_VERSION,
            "aggregate_analysis_version": AGGREGATE_ANALYSIS_VERSION,
            "comparison_method_version": COMPARISON_METHOD_VERSION,
            "bootstrap": {**DEFAULT_BOOTSTRAP, **(bootstrap or {})},
            "metric_weighting": METRIC_WEIGHTING,
            "analysis_runtime_hash": analysis_runtime_hash(),
        },
        "comparisons": comparisons,
        "metadata": {
            **repository_commit(),
            "analysis_code_commit": repository_commit().get(
                "repository_commit"
            ),
        },
    }
    return finalize_manifest(manifest)


def validate_manifest_for_execution(manifest: dict) -> list:
    """Deep validation beyond structure: schedule, comparisons, prompt
    drift, and pinned values that mirror code constants."""
    errors = list(manifest_store.validate_manifest(manifest))
    if errors:
        return errors
    execution = manifest["execution_contract"]
    errors.extend(verify_schedule(manifest))
    errors.extend(validate_comparisons(
        manifest["comparisons"], set(execution["conditions"]),
    ))
    errors.extend(verify_prompt_profile(execution["prompt_profile"]))
    policies = execution["policies"]
    limits = limits_dict()
    pins = (
        ("public_message_limit", limits["public_message_max_chars"]),
        ("wolf_message_limit", limits["wolf_message_max_chars"]),
        ("memory_limit", limits["memory_max_chars"]),
        ("agent_action_max_attempts", MAX_RETRIES),
    )
    for key, current in pins:
        if policies.get(key) != current:
            errors.append(
                f"pinned policy {key}={policies.get(key)!r} no longer "
                f"matches the code constant ({current!r}); create a new "
                "manifest"
            )
    return errors


# --------------------------------------------------------------------------
# Execution
# --------------------------------------------------------------------------

def _attempt_fields(started: dict) -> dict:
    return {key: started[key] for key in (
        "trial_id", "attempt_id", "attempt_number", "trial_index",
        "scheduler_position", "condition_id", "seed", "repetition",
        "game_id",
    )}


def _abort_engine(engine, reason: str) -> None:
    """Ask real engines to persist incomplete-run evidence before close.

    Test doubles and third-party factories are permitted to omit ``abort``;
    their terminal journal row still records the failure.
    """
    abort = getattr(engine, "abort", None)
    if callable(abort):
        try:
            abort(reason)
        except Exception:
            # The original execution exception is the authoritative failure.
            # Do not hide it behind best-effort forensic logging.
            pass


def _default_engine_factory(entry: dict, manifest: dict, games_dir: Path):
    from werewolf.engine.game import GameEngine
    from werewolf.experiments.health import generation_from_dict

    execution = manifest["execution_contract"]
    game = execution["game"]
    policies = execution["policies"]
    condition = execution["conditions"][entry["condition_id"]]
    return GameEngine(
        n_players=game["n_players"],
        n_wolves=game["n_wolves"],
        n_seers=game["n_seers"],
        seed=entry["seed"],
        output_dir=str(games_dir),
        api_key="",
        model=condition["role_models"]["villager"],
        show_all_channels=False,
        show_prompts=False,
        transcript_enabled=False,
        batch_id=f"{manifest['experiment_id']}/{entry['condition_id']}",
        trial_index=entry["trial_index"],
        belief_snapshots=game["belief_snapshots"],
        generation_config=generation_from_dict(execution["generation"]),
        discussion_cycles=game["discussion_cycles"],
        role_models=condition["role_models"],
        allow_provider_fallback=policies["allow_provider_fallback"],
        action_failure_policy=policies["action_failure_policy"],
        max_rounds=policies["max_rounds"],
        agent_action_max_attempts=policies["agent_action_max_attempts"],
        retryable_error_categories=policies["retryable_errors"],
        retry_backoff=policies["retry_backoff"],
        request_timeout_seconds=policies["request_timeout_seconds"],
    )


def _reconcile_open_attempts(
    writer: JournalWriter, games_dir: Path, game_rules: dict,
) -> dict:
    """Crash recovery: resolve every open attempt using the minimal
    execution-side verifier, never the PR 2 report builder."""
    counts = {"recovered": 0, "interrupted": 0}
    for trial in list(writer.state.trials.values()):
        open_attempt = trial.open_attempt
        if open_attempt is None:
            continue
        started = open_attempt["started"]
        source = reconcile_attempt_source(
            games_dir / f"{started['game_id']}.jsonl",
            expected_game_id=started["game_id"],
            game_rules=game_rules,
        )
        fields = _attempt_fields(started)
        verification = source["verification"]
        if source["source_status"] == SOURCE_MISSING:
            writer.append(TRIAL_INTERRUPTED, {
                **fields,
                "recorded_game_sha256": None,
                "source_status": SOURCE_MISSING,
                "reason": "no game log was found during crash recovery",
            })
            counts["interrupted"] += 1
        elif verification["complete"]:
            outcome = verification["outcome"]
            writer.append(TRIAL_COMPLETED, {
                **fields,
                "recorded_game_sha256": source["recorded_game_sha256"],
                "source_status": SOURCE_RECORDED,
                "winner": outcome["winner"],
                "rounds": outcome["rounds"],
                "recovered": True,
                "verifier": {
                    "verifier_version": verification["verifier_version"],
                    "checks": verification["checks"],
                },
            })
            counts["recovered"] += 1
        else:
            writer.append(TRIAL_INTERRUPTED, {
                **fields,
                "recorded_game_sha256": source["recorded_game_sha256"],
                "source_status": SOURCE_RECORDED,
                "reason": "; ".join(verification["reasons"])[:300],
                "verifier": {
                    "verifier_version": verification["verifier_version"],
                    "checks": verification["checks"],
                },
            })
            counts["interrupted"] += 1
    return counts


def _runnable_entries(
    manifest: dict, writer: JournalWriter, retry_failed: bool,
) -> tuple:
    """Schedule entries still needing an attempt, plus exhausted ones."""
    max_attempts = (
        manifest["execution_contract"]["policies"]["max_trial_attempts"]
    )
    runnable, exhausted = [], []
    for entry in manifest["execution_contract"]["schedule"]:
        trial = writer.state.trials.get(entry["trial_id"])
        if trial is not None and trial.completed:
            continue
        if trial is None:
            runnable.append(entry)
            continue

        attempts = trial.attempt_count
        terminal_type = trial.last_terminal_type
        if terminal_type == TRIAL_INTERRUPTED and attempts < max_attempts:
            # Interruption is an unknown execution state. Retry it within
            # the pinned attempt budget after reconciliation.
            runnable.append(entry)
        elif terminal_type == TRIAL_FAILED and retry_failed:
            # Explicit execution failures are never retried by --resume.
            # Each --retry-failed invocation grants one deliberate retry.
            runnable.append(entry)
        else:
            exhausted.append(entry)
    return runnable, exhausted


def run_experiment(
    root,
    experiment_id: str,
    *,
    resume: bool = False,
    retry_failed: bool = False,
    allow_adjusted_health: bool = False,
    engine_factory: Optional[Callable] = None,
    health_prober: Optional[Callable] = None,
    progress: Callable = print,
    clock: Callable = time.sleep,
) -> dict:
    manifest = load_verified_manifest(root, experiment_id)
    errors = validate_manifest_for_execution(manifest)
    if errors:
        raise ExperimentRunError(
            "Manifest failed execution validation: " + "; ".join(errors)
        )
    execution = manifest["execution_contract"]
    policies = execution["policies"]

    current_hash = execution_runtime_hash()
    if current_hash != execution["execution_runtime_hash"]:
        raise ExecutionRuntimeChanged(
            "Execution-relevant code changed since this manifest was "
            f"created (pinned {execution['execution_runtime_hash'][:12]}…, "
            f"current {current_hash[:12]}…). Create a new experiment for "
            "the new runtime; analysis-only changes never trigger this."
        )

    experiment_dir = manifest_store.experiment_dir(root, experiment_id)
    games_dir = manifest_store.games_dir(root, experiment_id)
    games_dir.mkdir(parents=True, exist_ok=True)

    with execution_lock(experiment_dir, experiment_id):
        writer = JournalWriter(
            manifest_store.journal_path(root, experiment_id),
            manifest_content_sha256=manifest["manifest_content_sha256"],
            execution_contract_sha256=execution_contract_sha256(manifest),
        )
        has_prior_records = bool(
            writer.state.trials or writer.state.sessions
        )
        if has_prior_records and not (resume or retry_failed):
            raise ExperimentRunError(
                f"Experiment {experiment_id} already has lifecycle "
                "records; pass --resume (or --retry-failed) to continue."
            )

        recovery = _reconcile_open_attempts(
            writer, games_dir, execution["game"],
        )
        runnable, exhausted = _runnable_entries(
            manifest, writer, retry_failed,
        )
        counts = {
            "recovered": recovery["recovered"],
            "reconciled_interrupted": recovery["interrupted"],
            "completed": 0, "failed": 0, "interrupted": 0,
            "skipped_exhausted": len(exhausted),
        }
        if not runnable:
            progress(
                f"No runnable trials for {experiment_id} "
                f"({counts['skipped_exhausted']} exhausted)."
            )
            return counts

        writer.session_started(
            execution_runtime_hash=current_hash,
            resume=has_prior_records,
            runnable_trials=len(runnable),
            retry_failed=retry_failed,
            allow_adjusted_health=allow_adjusted_health,
        )

        # Health preflight: one probe per unique fingerprint, always
        # journaled (results and costs survive even when blocked).
        prober = health_prober or probe_model
        blockers = []
        declared = {
            entry["fingerprint"]
            for entry in execution["predeclared_adjustments"]
        }
        for target in unique_health_targets(
            execution["conditions"], execution["generation"],
            request_timeout_seconds=policies["request_timeout_seconds"],
        ):
            record = prober(target)
            writer.append("health_check", record)
            reason = evaluate_health_record(
                record,
                predeclared_fingerprints=declared,
                allow_adjusted=allow_adjusted_health,
            )
            if reason:
                blockers.append(reason)
        if blockers:
            writer.session_aborted(
                "health preflight blocked execution: "
                + " | ".join(blockers)
            )
            raise ExperimentRunError(
                "Health preflight blocked execution:\n- "
                + "\n- ".join(blockers)
            )

        factory = engine_factory or _default_engine_factory
        delay = policies["intertrial_delay_seconds"]
        total = len(runnable)
        try:
            for i, entry in enumerate(runnable):
                trial = writer.state.trials.get(entry["trial_id"])
                attempt_number = (trial.attempt_count if trial else 0) + 1
                attempt_id = f"{entry['trial_id']}_a{attempt_number}"
                try:
                    engine = factory(entry, manifest, games_dir)
                except Exception as exc:
                    # No trial_started yet and no provider request made:
                    # this is a session-level failure, not a paid attempt.
                    writer.session_aborted(
                        "engine construction failed for trial "
                        f"{entry['trial_id']}: {sanitize_error(exc)}"
                    )
                    raise ExperimentRunError(
                        f"Engine construction failed for trial "
                        f"{entry['trial_id']}: {sanitize_error(exc)}"
                    ) from exc
                started_fields = {
                    "trial_id": entry["trial_id"],
                    "attempt_id": attempt_id,
                    "attempt_number": attempt_number,
                    "trial_index": entry["trial_index"],
                    "scheduler_position": entry["scheduler_position"],
                    "condition_id": entry["condition_id"],
                    "seed": entry["seed"],
                    "repetition": entry["repetition"],
                    "game_id": engine.state.game_id,
                }
                # Journal the game identity BEFORE the first provider
                # request so a crash can always be reconciled.
                writer.append(TRIAL_STARTED, started_fields)
                try:
                    engine.run()
                except KeyboardInterrupt:
                    _abort_engine(engine, "operator_interrupt")
                    engine.close()
                    source = reconcile_attempt_source(
                        games_dir / f"{engine.state.game_id}.jsonl",
                        expected_game_id=engine.state.game_id,
                        game_rules=execution["game"],
                    )
                    writer.append(TRIAL_INTERRUPTED, {
                        **started_fields,
                        "recorded_game_sha256":
                            source["recorded_game_sha256"],
                        "source_status": source["source_status"],
                        "reason": "operator interrupt",
                    })
                    counts["interrupted"] += 1
                    writer.session_aborted("operator interrupt")
                    raise
                except Exception as exc:
                    _abort_engine(engine, exc.__class__.__name__)
                    engine.close()
                    source = reconcile_attempt_source(
                        games_dir / f"{engine.state.game_id}.jsonl",
                        expected_game_id=engine.state.game_id,
                        game_rules=execution["game"],
                    )
                    writer.append(TRIAL_FAILED, {
                        **started_fields,
                        "recorded_game_sha256":
                            source["recorded_game_sha256"],
                        "source_status": source["source_status"],
                        "sanitized_error": sanitize_error(exc),
                        "error_category": exc.__class__.__name__,
                    })
                    counts["failed"] += 1
                    progress(
                        f"  [{i + 1}/{total}] {entry['condition_id']} "
                        f"seed={entry['seed']} rep={entry['repetition']} "
                        f"FAILED: {sanitize_error(exc)[:80]}"
                    )
                else:
                    engine.close()
                    source = reconcile_attempt_source(
                        games_dir / f"{engine.state.game_id}.jsonl",
                        expected_game_id=engine.state.game_id,
                        game_rules=execution["game"],
                    )
                    verification = source["verification"]
                    if source["source_status"] == SOURCE_RECORDED \
                            and verification["complete"]:
                        outcome = verification["outcome"]
                        writer.append(TRIAL_COMPLETED, {
                            **started_fields,
                            "recorded_game_sha256":
                                source["recorded_game_sha256"],
                            "source_status": SOURCE_RECORDED,
                            "winner": outcome["winner"],
                            "rounds": outcome["rounds"],
                            "verifier": {
                                "verifier_version":
                                    verification["verifier_version"],
                                "checks": verification["checks"],
                            },
                        })
                        counts["completed"] += 1
                        progress(
                            f"  [{i + 1}/{total}] {entry['condition_id']} "
                            f"seed={entry['seed']} "
                            f"rep={entry['repetition']} "
                            f"winner={outcome['winner']}"
                        )
                    else:
                        reasons = (
                            "; ".join(verification["reasons"])
                            if verification else "game log missing"
                        )
                        writer.append(TRIAL_FAILED, {
                            **started_fields,
                            "recorded_game_sha256":
                                source["recorded_game_sha256"],
                            "source_status": source["source_status"],
                            "sanitized_error": sanitize_error(
                                "engine finished but the completion "
                                f"verifier rejected the log: {reasons}"
                            ),
                            "error_category": "verification_failed",
                        })
                        counts["failed"] += 1
                if delay and i + 1 < total:
                    clock(delay)
        except KeyboardInterrupt:
            raise
        else:
            writer.session_finished(
                completed_trials=counts["completed"],
                failed_trials=counts["failed"],
                interrupted_trials=counts["interrupted"],
            )
    return counts
