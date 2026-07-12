"""Post-hoc belief/manipulation analysis of game logs.

Usage:
    python -m werewolf.cli.analyze --game outputs/games/game_x.jsonl [...]
    python -m werewolf.cli.analyze --manifest outputs/games/trials_manifest_X.jsonl
    ... [--output metrics.json]

Pure computation over existing logs - never makes API calls, so it is
free and can be re-run whenever metric definitions change.
"""
import argparse
import json

from werewolf.evaluation.belief_metrics import (
    aggregate_belief_metrics,
    compute_game_metrics_from_file,
)


def _fmt(value, digits=3):
    return "n/a" if value is None else f"{value:.{digits}f}"


def print_metrics(metrics: dict, label: str):
    print(f"\n=== {label} ===")
    if not metrics.get("available", True):
        print(f"  not analyzable: {metrics.get('reason')}")
        return
    coverage = metrics.get("coverage") or {}
    for checkpoint, c in coverage.items():
        print(f"  snapshots {checkpoint}: {c['valid']}/{c['emitted']} valid")
    shift = metrics["belief_shift_toward_wolves"]
    print(f"  belief shift toward wolves: {_fmt(shift['mean'])} (n={shift['n']})")
    print(f"  initial correctness:        "
          f"{_fmt(metrics['initial_correctness']['rate'])} "
          f"(n={metrics['initial_correctness']['n']})")
    print(f"  harmful revision rate:      "
          f"{_fmt(metrics['harmful_revision']['rate'])} "
          f"(n initially correct={metrics['harmful_revision']['n']})")
    print(f"  beneficial revision rate:   "
          f"{_fmt(metrics['beneficial_revision']['rate'])} "
          f"(n initially wrong={metrics['beneficial_revision']['n']})")
    alignment = metrics["vote_belief_alignment"]
    consistency = metrics["response_internal_consistency"]
    print(f"  vote-belief alignment:      {_fmt(alignment['rate'])} "
          f"(n={alignment['n']}); response internal consistency: "
          f"{_fmt(consistency['rate'])} (n={consistency['n']})")
    brier = metrics["calibration_brier"]
    print(f"  Brier calibration:          pre {_fmt(brier['pre'])} "
          f"(n={brier['n_pre']}) | post {_fmt(brier['post'])} "
          f"(n={brier['n_post']})")
    awareness = metrics["wolf_suspicion_awareness"]
    print(f"  wolf suspicion-awareness MAE: {_fmt(awareness['mae'])} "
          f"(n={awareness['n']})")


def main():
    parser = argparse.ArgumentParser(
        description="Compute belief/manipulation metrics from game logs"
    )
    parser.add_argument("--game", nargs="*", default=[],
                        help="Per-game JSONL log path(s)")
    parser.add_argument("--manifest", type=str, default=None,
                        help="Trial manifest JSONL; analyzes every game in it")
    parser.add_argument("--output", type=str, default=None,
                        help="Write full metrics JSON to this path")
    args = parser.parse_args()

    game_paths = list(args.game)
    if args.manifest:
        with open(args.manifest, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    if record.get("log_path"):
                        game_paths.append(record["log_path"])
    if not game_paths:
        raise SystemExit("Nothing to analyze: pass --game and/or --manifest")

    per_game = {}
    for path in game_paths:
        per_game[path] = compute_game_metrics_from_file(path)
        print_metrics(per_game[path], path)

    result = {"per_game": per_game}
    if len(game_paths) > 1:
        aggregate = aggregate_belief_metrics(list(per_game.values()))
        result["aggregate"] = aggregate
        print(f"\n=== AGGREGATE ({aggregate['games_with_metrics']}/"
              f"{aggregate['games']} games analyzable) ===")
        print(json.dumps(aggregate, indent=2))

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"\nWritten: {args.output}")


if __name__ == "__main__":
    main()
