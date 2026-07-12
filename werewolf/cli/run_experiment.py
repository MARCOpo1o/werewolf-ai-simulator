"""Crossed two-model experiment runner (producer-target matrix).

Homogeneous self-play cannot distinguish "model A produces better
manipulation" from "model A's villagers are weak detectors". For a model
pair (A, B) this runner executes four conditions on the SAME seeds (same
seed => identical role assignment, verified by test):

    a_homogeneous       A wolves vs A village
    b_homogeneous       B wolves vs B village
    a_wolves_b_village  A wolves vs B village
    b_wolves_a_village  B wolves vs A village

Each (condition, seed) is repeated --repetitions times because API models
remain nondeterministic even with pinned generation parameters.

Usage:
    python -m werewolf.cli.run_experiment \
        --experiment-id pilot_2026_07 \
        --model-a gemini_flash_lite --model-b fast \
        --num-seeds 10 --seed-start 5000 --repetitions 2 \
        --n 7 --wolves 2 --seers 1 --quiet

Outputs (in --output-dir):
    experiment_<id>.jsonl          one record per game (crash-safe, appended)
    experiment_<id>_summary.json   full spec + per-condition summaries
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone

from werewolf.agents.prompts import get_prompt_version
from werewolf.cli.run_game import get_api_key, load_env_file, setup_logging
from werewolf.cli.run_trials import build_batch_summary, run_one_trial
from werewolf.engine.beliefs import BELIEF_SCHEMA_VERSION
from werewolf.engine.game import get_code_commit
from werewolf.engine.limits import limits_dict
from werewolf.evaluation.belief_metrics import METRICS_VERSION
from werewolf.llm.provider import GenerationConfig
from werewolf.llm.records import SCHEMA_VERSION
from werewolf.llm.registry import registry_snapshot, resolve


def build_conditions(model_a: str, model_b: str) -> dict[str, dict]:
    """condition_id -> role_models mapping for the 2x2 producer-target
    matrix. Seer detects for the village, so it plays the village model."""
    def assignment(wolves: str, village: str) -> dict:
        return {"werewolf": wolves, "villager": village, "seer": village}

    return {
        "a_homogeneous": assignment(model_a, model_a),
        "b_homogeneous": assignment(model_b, model_b),
        "a_wolves_b_village": assignment(model_a, model_b),
        "b_wolves_a_village": assignment(model_b, model_a),
    }


def run_crossed_experiment(
    *,
    experiment_id: str,
    model_a: str,
    model_b: str,
    seeds: list[int],
    repetitions: int,
    n_players: int,
    n_wolves: int,
    n_seers: int,
    output_dir: str,
    quiet: bool = True,
    generation_config: GenerationConfig = None,
    discussion_cycles: int = 2,
    belief_snapshots: bool = True,
    progress=print,
) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    conditions = build_conditions(model_a, model_b)
    generation_config = generation_config or GenerationConfig()

    manifest_path = os.path.join(output_dir, f"experiment_{experiment_id}.jsonl")
    started_at = datetime.now(timezone.utc).isoformat()

    records_by_condition: dict[str, list] = {c: [] for c in conditions}
    total = len(conditions) * len(seeds) * repetitions
    done = 0
    trial_index = 0
    with open(manifest_path, "w", encoding="utf-8") as manifest:
        for condition_id, role_models in conditions.items():
            for seed in seeds:
                for repetition in range(repetitions):
                    record = run_one_trial(
                        trial_index=trial_index,
                        seed=seed,
                        n_players=n_players,
                        n_wolves=n_wolves,
                        n_seers=n_seers,
                        output_dir=output_dir,
                        api_key="",  # keys resolved per role via registry
                        model=role_models["villager"],
                        quiet=quiet,
                        batch_id=f"{experiment_id}/{condition_id}",
                        belief_snapshots=belief_snapshots,
                        generation_config=generation_config,
                        discussion_cycles=discussion_cycles,
                        role_models=role_models,
                    )
                    record["condition_id"] = condition_id
                    record["repetition"] = repetition
                    record["experiment_id"] = experiment_id
                    records_by_condition[condition_id].append(record)
                    manifest.write(json.dumps(record) + "\n")
                    manifest.flush()
                    trial_index += 1
                    done += 1
                    progress(f"  [{done}/{total}] {condition_id} "
                             f"seed={seed} rep={repetition} "
                             f"winner={record['winner']}")

    completed_at = datetime.now(timezone.utc).isoformat()

    condition_summaries = {}
    for condition_id, records in records_by_condition.items():
        condition_summaries[condition_id] = build_batch_summary(
            records,
            run_id=f"{experiment_id}/{condition_id}",
            started_at=started_at,
            completed_at=completed_at,
            trials_requested=len(seeds) * repetitions,
            failed_trials=0,
            config={"role_models": conditions[condition_id]},
            manifest_path=manifest_path,
        )

    summary = {
        # full condition specification (benchmark reproducibility)
        "experiment_id": experiment_id,
        "started_at": started_at,
        "completed_at": completed_at,
        "code_commit": get_code_commit(),
        "python_version": sys.version,
        "seeds": seeds,
        "repetitions_per_seed": repetitions,
        "game": {
            "players": n_players,
            "wolves": n_wolves,
            "seers": n_seers,
            "discussion_cycles": discussion_cycles,
        },
        "instrumentation": {
            "belief_snapshots": belief_snapshots,
            "belief_schema": BELIEF_SCHEMA_VERSION,
            "metrics_version": METRICS_VERSION,
            "usage_schema": SCHEMA_VERSION,
            "limits": limits_dict(),
            "prompt_version": get_prompt_version(),
        },
        "models": {
            "a": {"requested": model_a, **_spec_dict(model_a)},
            "b": {"requested": model_b, **_spec_dict(model_b)},
            "generation_config": generation_config.to_json_dict(),
        },
        "model_registry_snapshot": registry_snapshot(),
        "conditions": condition_summaries,
        "manifest_path": manifest_path,
    }

    summary_path = os.path.join(
        output_dir, f"experiment_{experiment_id}_summary.json"
    )
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    summary["summary_path"] = summary_path
    return summary


def _spec_dict(name: str) -> dict:
    spec = resolve(name)
    return {
        "model": spec.model,
        "alias": spec.alias,
        "provider": spec.provider,
        "reasoning_effort": spec.reasoning_effort,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run the crossed (producer-target) two-model experiment"
    )
    parser.add_argument("--experiment-id", type=str, required=True)
    parser.add_argument("--model-a", type=str, required=True)
    parser.add_argument("--model-b", type=str, required=True)
    parser.add_argument("--seed-start", type=int, default=5000)
    parser.add_argument("--num-seeds", type=int, default=10)
    parser.add_argument("--seed-file", type=str, default=None,
                        help="JSON file with an explicit list of seeds "
                             "(overrides --seed-start/--num-seeds)")
    parser.add_argument("--repetitions", type=int, default=2,
                        help="Repetitions per (condition, seed) (default: 2)")
    parser.add_argument("--n", type=int, default=7)
    parser.add_argument("--wolves", type=int, default=2)
    parser.add_argument("--seers", type=int, default=1)
    parser.add_argument("--output-dir", type=str, default="outputs/experiments")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--no-belief-snapshots", action="store_true")
    parser.add_argument("--discussion-cycles", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--max-output-tokens", type=int, default=None)
    parser.add_argument("--provider-seed", type=int, default=None)
    args = parser.parse_args()

    load_env_file()
    setup_logging(args.debug)

    # fail fast on missing keys for either model
    for name in (args.model_a, args.model_b):
        if not get_api_key(name):
            spec = resolve(name)
            env_names = " or ".join(spec.api_key_env) or "an API key"
            raise SystemExit(f"Error: {env_names} not set (model {name})")

    if args.seed_file:
        with open(args.seed_file, encoding="utf-8") as f:
            seeds = list(json.load(f))
    else:
        seeds = list(range(args.seed_start, args.seed_start + args.num_seeds))

    summary = run_crossed_experiment(
        experiment_id=args.experiment_id,
        model_a=args.model_a,
        model_b=args.model_b,
        seeds=seeds,
        repetitions=args.repetitions,
        n_players=args.n,
        n_wolves=args.wolves,
        n_seers=args.seers,
        output_dir=args.output_dir,
        quiet=args.quiet,
        generation_config=GenerationConfig(
            temperature=args.temperature,
            top_p=args.top_p,
            max_output_tokens=args.max_output_tokens,
            provider_seed=args.provider_seed,
        ),
        discussion_cycles=args.discussion_cycles,
        belief_snapshots=not args.no_belief_snapshots,
    )

    print(f"\nExperiment complete: {summary['experiment_id']}")
    for condition_id, cond in summary["conditions"].items():
        usage = cond["usage"]
        cost = usage["cost_usd_total"]
        cost_str = f"${cost:.4f}" if cost is not None else "$?"
        print(f"  {condition_id}: wolf_win_rate={cond['wolf_win_rate']:.2f} "
              f"({cond['trials_completed']} games, {cost_str})")
    print(f"Manifest: {summary['manifest_path']}")
    print(f"Summary:  {summary['summary_path']}")


if __name__ == "__main__":
    main()
