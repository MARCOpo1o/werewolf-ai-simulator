import argparse
import csv
import json
import os
from datetime import datetime, timezone
from statistics import mean

from werewolf.cli.run_game import MODEL_PRESETS, get_api_key, load_env_file, setup_logging
from werewolf.engine.game import GameEngine
from werewolf.evaluation.belief_metrics import (
    aggregate_belief_metrics,
    compute_game_metrics_from_file,
)
from werewolf.llm.ledger import aggregate_game_summaries
from werewolf.llm.provider import GenerationConfig
from werewolf.llm.registry import build_provider, registry_snapshot, resolve


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_config(n_players: int, n_wolves: int, n_seers: int):
    if n_players < 3:
        raise ValueError("Need at least 3 players")
    if n_wolves < 1:
        raise ValueError("Need at least 1 wolf")
    if n_wolves >= n_players:
        raise ValueError("Number of wolves must be less than total players")
    if n_seers not in (0, 1):
        raise ValueError("Number of seers must be 0 or 1")
    if n_wolves + n_seers >= n_players:
        raise ValueError("Need at least 1 villager")


def run_one_trial(
    trial_index: int,
    seed: int,
    n_players: int,
    n_wolves: int,
    n_seers: int,
    output_dir: str,
    api_key: str,
    model: str,
    quiet: bool,
    provider=None,
    model_alias: str = None,
    reasoning_effort: str = None,
    batch_id: str = None,
    belief_snapshots: bool = True,
    generation_config=None,
    discussion_cycles: int = 2,
    role_models: dict = None,
    role_providers: dict = None,
) -> dict:
    engine = GameEngine(
        n_players=n_players,
        n_wolves=n_wolves,
        n_seers=n_seers,
        seed=seed,
        output_dir=output_dir,
        api_key=api_key,
        model=model,
        show_all_channels=not quiet,
        show_prompts=False,
        transcript_enabled=not quiet,
        provider=provider,
        model_alias=model_alias,
        reasoning_effort=reasoning_effort,
        batch_id=batch_id,
        trial_index=trial_index,
        belief_snapshots=belief_snapshots,
        generation_config=generation_config,
        discussion_cycles=discussion_cycles,
        role_models=role_models,
        role_providers=role_providers,
    )
    winner = engine.run()
    remaining = [p.id for p in engine.state.get_alive_players()]
    return {
        "trial_index": trial_index,
        "seed": seed,
        "batch_id": batch_id,
        "game_id": engine.state.game_id,
        "winner": winner,
        "rounds": engine.state.round,
        "remaining": remaining,
        "log_path": engine.logger.filepath,
        "usage": engine.ledger.game_summary(),
        "config": {
            "n_players": n_players,
            "n_wolves": n_wolves,
            "n_seers": n_seers,
            "model": model,
            "model_alias": model_alias,
            "reasoning_effort": reasoning_effort,
            "role_models": engine.role_models_resolved,
        },
    }


def write_manifest(path: str, records: list[dict]):
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def write_summary_json(path: str, summary: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def write_summary_csv(path: str, summary: dict):
    usage = summary.get("usage") or {}
    per_game = usage.get("cost_per_game") or {}
    fields = [
        "trials_requested",
        "trials_completed",
        "wolf_wins",
        "village_wins",
        "wolf_win_rate",
        "village_win_rate",
        "avg_rounds",
        "total_cost_usd",
        "cost_complete",
        "games_with_incomplete_cost",
        "mean_cost_per_game_usd",
        "median_cost_per_game_usd",
        "p90_cost_per_game_usd",
        "min_cost_per_game_usd",
        "max_cost_per_game_usd",
        "total_tokens",
        "reasoning_tokens",
        "cached_input_tokens",
        "llm_calls",
        "retries",
        "fallbacks",
        "harmful_revision_rate",
        "beneficial_revision_rate",
        "vote_belief_alignment_rate",
        "brier_post",
        "wolf_awareness_mae",
        "started_at",
        "completed_at",
    ]
    beliefs = summary.get("belief_metrics") or {}

    def metric(key, field):
        return (beliefs.get(key) or {}).get(field)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerow({
            "trials_requested": summary["trials_requested"],
            "trials_completed": summary["trials_completed"],
            "wolf_wins": summary["outcome_counts"]["wolf"],
            "village_wins": summary["outcome_counts"]["village"],
            "wolf_win_rate": summary["wolf_win_rate"],
            "village_win_rate": summary["village_win_rate"],
            "avg_rounds": summary["avg_rounds"],
            # Empty cells (not 0) when cost is unknown.
            "total_cost_usd": usage.get("cost_usd_total"),
            "cost_complete": usage.get("cost_complete"),
            "games_with_incomplete_cost": usage.get("games_with_incomplete_cost"),
            "mean_cost_per_game_usd": per_game.get("mean"),
            "median_cost_per_game_usd": per_game.get("median"),
            "p90_cost_per_game_usd": per_game.get("p90"),
            "min_cost_per_game_usd": per_game.get("min"),
            "max_cost_per_game_usd": per_game.get("max"),
            "total_tokens": (usage.get("tokens") or {}).get("total_tokens"),
            "reasoning_tokens": (usage.get("tokens") or {}).get("reasoning_tokens"),
            "cached_input_tokens": (usage.get("tokens") or {}).get("cached_input_tokens"),
            "llm_calls": usage.get("calls"),
            "retries": usage.get("retries"),
            "fallbacks": usage.get("fallbacks"),
            "harmful_revision_rate": metric("harmful_revision", "rate"),
            "beneficial_revision_rate": metric("beneficial_revision", "rate"),
            "vote_belief_alignment_rate": metric("vote_belief_alignment", "rate"),
            "brier_post": (beliefs.get("calibration_brier") or {}).get("post"),
            "wolf_awareness_mae": metric("wolf_suspicion_awareness", "mae"),
            "started_at": summary["started_at"],
            "completed_at": summary["completed_at"],
        })


def build_batch_summary(
    records: list[dict],
    *,
    run_id: str,
    started_at: str,
    completed_at: str,
    trials_requested: int,
    failed_trials: int,
    config: dict,
    manifest_path: str,
    health_check_records: list[dict] = None,
) -> dict:
    wolf_wins = sum(1 for r in records if r["winner"] == "wolf")
    village_wins = sum(1 for r in records if r["winner"] == "village")
    rounds = [r["rounds"] for r in records]
    total = len(records)

    summary = {
        "run_id": run_id,
        "started_at": started_at,
        "completed_at": completed_at,
        "trials_requested": trials_requested,
        "trials_completed": total,
        "failed_trials": failed_trials,
        "outcome_counts": {"wolf": wolf_wins, "village": village_wins},
        "wolf_win_rate": (wolf_wins / total) if total else 0.0,
        "village_win_rate": (village_wins / total) if total else 0.0,
        "avg_rounds": mean(rounds) if rounds else 0.0,
        "usage": aggregate_game_summaries(
            [r["usage"] for r in records if r.get("usage")]
        ),
        "belief_metrics": aggregate_belief_metrics([
            compute_game_metrics_from_file(r["log_path"])
            for r in records if r.get("log_path")
        ]),
        "config": config,
        "model_registry_snapshot": registry_snapshot(),
        "manifest_path": manifest_path,
    }
    if health_check_records is not None:
        summary["health_check"] = {
            "games": len(health_check_records),
            "usage": aggregate_game_summaries(
                [r["usage"] for r in health_check_records if r.get("usage")]
            ),
        }
    return summary


def run_health_check(
    checks: int,
    seed_start: int,
    n_players: int,
    n_wolves: int,
    n_seers: int,
    output_dir: str,
    api_key: str,
    model: str,
    provider=None,
    model_alias: str = None,
    reasoning_effort: str = None,
    batch_id: str = None,
    belief_snapshots: bool = True,
    generation_config=None,
    discussion_cycles: int = 2,
) -> list[dict]:
    health_output_dir = os.path.join(output_dir, "healthcheck")
    records = []
    for i in range(checks):
        records.append(run_one_trial(
            trial_index=i,
            seed=seed_start + i,
            n_players=n_players,
            n_wolves=n_wolves,
            n_seers=n_seers,
            output_dir=health_output_dir,
            api_key=api_key,
            model=model,
            quiet=True,
            provider=provider,
            model_alias=model_alias,
            reasoning_effort=reasoning_effort,
            batch_id=batch_id,
            belief_snapshots=belief_snapshots,
            generation_config=generation_config,
            discussion_cycles=discussion_cycles,
        ))
    return records


def _fmt_cost(usd) -> str:
    return f"${usd:.4f}" if usd is not None else "$?"


def main():
    parser = argparse.ArgumentParser(description="Run many Werewolf trials and summarize results")
    parser.add_argument("--trials", type=int, default=200, help="Number of trials to run (default: 200)")
    parser.add_argument("--seed-start", type=int, default=1000, help="First seed value (default: 1000)")
    parser.add_argument("--n", type=int, default=7, help="Number of players (default: 7)")
    parser.add_argument("--wolves", type=int, default=2, help="Number of wolves (default: 2)")
    parser.add_argument("--seers", type=int, default=1, help="Number of seers, 0 or 1 (default: 1)")
    parser.add_argument("--output-dir", type=str, default="outputs/games", help="Directory for game logs/results")
    parser.add_argument(
        "--model",
        type=str,
        default="fast",
        help=f"Model alias or full ID. Aliases: {list(MODEL_PRESETS.keys())} (default: fast)",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress transcript output for faster execution")
    parser.add_argument("--debug", action="store_true", help="Enable debug logs")
    parser.add_argument(
        "--health-check",
        type=int,
        default=5,
        help="Run this many preflight games before trials (0 disables, default: 5)",
    )
    parser.add_argument(
        "--health-check-only",
        action="store_true",
        help="Run only health check and exit",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue remaining trials when a trial fails",
    )
    parser.add_argument(
        "--no-belief-snapshots",
        action="store_true",
        help="Disable structured belief/suspicion snapshots (cheaper, but "
             "games cannot be analyzed for manipulation metrics)",
    )
    parser.add_argument("--discussion-cycles", type=int, default=2,
                        help="Discussion cycles per day (default: 2)")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--max-output-tokens", type=int, default=None)
    parser.add_argument("--provider-seed", type=int, default=None)

    args = parser.parse_args()
    load_env_file()
    setup_logging(args.debug)

    spec = resolve(args.model)
    model_name = spec.model

    api_key = get_api_key(args.model)
    if not api_key:
        env_names = " or ".join(spec.api_key_env) or "an API key"
        raise SystemExit(f"Error: {env_names} environment variable is not set.")
    provider = build_provider(spec, api_key=api_key)

    validate_config(args.n, args.wolves, args.seers)
    if args.trials < 1 and not args.health_check_only:
        raise SystemExit("Error: --trials must be >= 1")
    if args.health_check < 0:
        raise SystemExit("Error: --health-check must be >= 0")

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # run_id doubles as batch_id and is generated up front so every game
    # log and usage record can be attributed to this batch.
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    manifest_path = os.path.join(output_dir, f"trials_manifest_{run_id}.jsonl")

    generation_config = GenerationConfig(
        temperature=args.temperature,
        top_p=args.top_p,
        max_output_tokens=args.max_output_tokens,
        reasoning_effort=spec.reasoning_effort,
        provider_seed=args.provider_seed,
    )
    trial_kwargs = dict(
        provider=provider,
        model_alias=spec.alias,
        reasoning_effort=spec.reasoning_effort,
        batch_id=run_id,
        belief_snapshots=not args.no_belief_snapshots,
        generation_config=generation_config,
        discussion_cycles=args.discussion_cycles,
    )

    health_records = None
    if args.health_check > 0:
        health_records = run_health_check(
            checks=args.health_check,
            seed_start=args.seed_start,
            n_players=args.n,
            n_wolves=args.wolves,
            n_seers=args.seers,
            output_dir=output_dir,
            api_key=api_key,
            model=model_name,
            **trial_kwargs,
        )
        health_cost = aggregate_game_summaries(
            [r["usage"] for r in health_records]
        )["cost_usd_total"]
        print(f"Health check passed ({args.health_check} games, "
              f"cost {_fmt_cost(health_cost)})")
    if args.health_check_only:
        return

    started_at = _now_utc()
    records = []
    errors = 0
    total = args.trials
    # Manifest is appended and flushed per trial so a crash mid-batch
    # loses nothing.
    manifest_file = open(manifest_path, "w", encoding="utf-8")
    try:
        for i in range(total):
            seed = args.seed_start + i
            try:
                record = run_one_trial(
                    trial_index=i,
                    seed=seed,
                    n_players=args.n,
                    n_wolves=args.wolves,
                    n_seers=args.seers,
                    output_dir=output_dir,
                    api_key=api_key,
                    model=model_name,
                    quiet=args.quiet,
                    **trial_kwargs,
                )
                records.append(record)
                manifest_file.write(json.dumps(record) + "\n")
                manifest_file.flush()
            except Exception as exc:
                errors += 1
                if args.continue_on_error:
                    print(f"[trial {i}] failed: {exc}")
                    continue
                raise

            done = i + 1
            w = sum(1 for r in records if r["winner"] == "wolf")
            v = sum(1 for r in records if r["winner"] == "village")
            costs = [
                r["usage"]["cost_usd_total"] for r in records
                if r["usage"].get("cost_usd_total") is not None
            ]
            cost_str = _fmt_cost(sum(costs) if costs else None)
            bar_len = 30
            filled = int(bar_len * done / total)
            bar = "█" * filled + "░" * (bar_len - filled)
            err_str = f" err={errors}" if errors else ""
            print(
                f"\r  [{bar}] {done}/{total}  W:{w} V:{v}  {cost_str}{err_str}",
                end="",
                flush=True,
            )
        print()
    finally:
        manifest_file.close()

    completed_at = _now_utc()

    summary = build_batch_summary(
        records,
        run_id=run_id,
        started_at=started_at,
        completed_at=completed_at,
        trials_requested=args.trials,
        failed_trials=errors,
        config={
            "n_players": args.n,
            "n_wolves": args.wolves,
            "n_seers": args.seers,
            "seed_start": args.seed_start,
            "model": model_name,
            "model_alias": spec.alias,
            "reasoning_effort": spec.reasoning_effort,
            "provider": spec.provider,
            "generation_config": generation_config.to_json_dict(),
            "discussion_cycles": args.discussion_cycles,
            "quiet": args.quiet,
            "health_check": args.health_check,
        },
        manifest_path=manifest_path,
        health_check_records=health_records,
    )

    summary_json_path = os.path.join(output_dir, f"trials_summary_{run_id}.json")
    summary_csv_path = os.path.join(output_dir, f"trials_summary_{run_id}.csv")
    write_summary_json(summary_json_path, summary)
    write_summary_csv(summary_csv_path, summary)

    usage = summary["usage"]
    per_game = usage.get("cost_per_game") or {}
    print(f"Trials complete: {len(records)}/{args.trials}")
    print(
        f"Batch cost: {_fmt_cost(usage['cost_usd_total'])}"
        + ("" if usage["cost_complete"] else
           f" (incomplete: {usage['games_with_incomplete_cost']} games)")
    )
    if per_game:
        print(
            f"Per game: mean {_fmt_cost(per_game['mean'])} | "
            f"median {_fmt_cost(per_game['median'])} | "
            f"p90 {_fmt_cost(per_game['p90'])}"
        )
    print(
        f"Calls: {usage['calls']} (retries: {usage['retries']}, "
        f"fallbacks: {usage['fallbacks']})"
    )
    print(f"Manifest: {manifest_path}")
    print(f"Summary JSON: {summary_json_path}")
    print(f"Summary CSV: {summary_csv_path}")


if __name__ == "__main__":
    main()
