import argparse
import csv
import json
import os
from datetime import datetime, timezone
from statistics import mean

from werewolf.cli.run_game import MODEL_PRESETS, get_api_key, load_env_file, setup_logging
from werewolf.engine.game import GameEngine


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
    )
    winner = engine.run()
    remaining = [p.id for p in engine.state.get_alive_players()]
    return {
        "trial_index": trial_index,
        "seed": seed,
        "game_id": engine.state.game_id,
        "winner": winner,
        "rounds": engine.state.round,
        "remaining": remaining,
        "log_path": engine.logger.filepath,
        "config": {
            "n_players": n_players,
            "n_wolves": n_wolves,
            "n_seers": n_seers,
            "model": model,
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
    fields = [
        "trials_requested",
        "trials_completed",
        "wolf_wins",
        "village_wins",
        "wolf_win_rate",
        "village_win_rate",
        "avg_rounds",
        "started_at",
        "completed_at",
    ]
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
            "started_at": summary["started_at"],
            "completed_at": summary["completed_at"],
        })


def run_health_check(
    checks: int,
    seed_start: int,
    n_players: int,
    n_wolves: int,
    n_seers: int,
    output_dir: str,
    api_key: str,
    model: str,
):
    health_output_dir = os.path.join(output_dir, "healthcheck")
    for i in range(checks):
        run_one_trial(
            trial_index=i,
            seed=seed_start + i,
            n_players=n_players,
            n_wolves=n_wolves,
            n_seers=n_seers,
            output_dir=health_output_dir,
            api_key=api_key,
            model=model,
            quiet=True,
        )


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
        help=f"Model preset or full name. Presets: {list(MODEL_PRESETS.keys())} (default: fast)",
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

    args = parser.parse_args()
    load_env_file()
    setup_logging(args.debug)
    model_name = MODEL_PRESETS.get(args.model, args.model)

    api_key = get_api_key()
    if not api_key:
        raise SystemExit("Error: GROK_API_KEY or XAI_API_KEY environment variable is not set.")

    validate_config(args.n, args.wolves, args.seers)
    if args.trials < 1 and not args.health_check_only:
        raise SystemExit("Error: --trials must be >= 1")
    if args.health_check < 0:
        raise SystemExit("Error: --health-check must be >= 0")

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    if args.health_check > 0:
        run_health_check(
            checks=args.health_check,
            seed_start=args.seed_start,
            n_players=args.n,
            n_wolves=args.wolves,
            n_seers=args.seers,
            output_dir=output_dir,
            api_key=api_key,
            model=model_name,
        )
        print(f"Health check passed ({args.health_check} games)")
    if args.health_check_only:
        return

    started_at = _now_utc()
    records = []
    for i in range(args.trials):
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
            )
            records.append(record)
        except Exception as exc:
            if args.continue_on_error:
                print(f"[trial {i}] failed: {exc}")
                continue
            raise

    completed_at = _now_utc()

    wolf_wins = sum(1 for r in records if r["winner"] == "wolf")
    village_wins = sum(1 for r in records if r["winner"] == "village")
    rounds = [r["rounds"] for r in records]
    total = len(records)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    manifest_path = os.path.join(output_dir, f"trials_manifest_{run_id}.jsonl")
    summary_json_path = os.path.join(output_dir, f"trials_summary_{run_id}.json")
    summary_csv_path = os.path.join(output_dir, f"trials_summary_{run_id}.csv")

    summary = {
        "run_id": run_id,
        "started_at": started_at,
        "completed_at": completed_at,
        "trials_requested": args.trials,
        "trials_completed": total,
        "outcome_counts": {
            "wolf": wolf_wins,
            "village": village_wins,
        },
        "wolf_win_rate": (wolf_wins / total) if total else 0.0,
        "village_win_rate": (village_wins / total) if total else 0.0,
        "avg_rounds": mean(rounds) if rounds else 0.0,
        "config": {
            "n_players": args.n,
            "n_wolves": args.wolves,
            "n_seers": args.seers,
            "seed_start": args.seed_start,
            "model": model_name,
            "quiet": args.quiet,
            "health_check": args.health_check,
        },
        "manifest_path": manifest_path,
    }

    write_manifest(manifest_path, records)
    write_summary_json(summary_json_path, summary)
    write_summary_csv(summary_csv_path, summary)

    print(f"Trials complete: {total}/{args.trials}")
    print(f"Manifest: {manifest_path}")
    print(f"Summary JSON: {summary_json_path}")
    print(f"Summary CSV: {summary_csv_path}")


if __name__ == "__main__":
    main()
