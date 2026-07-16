"""CLI for the multi-game benchmark experiment system.

Usage:
    python -m werewolf.cli.experiment validate <manifest.json>
    python -m werewolf.cli.experiment create-crossed \
        --experiment-id pilot --model-a fast --model-b gemini_flash_lite \
        --num-seeds 10 --seed-start 5000 --repetitions 2
    python -m werewolf.cli.experiment run <experiment_id> \
        [--resume] [--retry-failed] [--allow-adjusted-health]
    python -m werewolf.cli.experiment summarize <experiment_id> \
        [--analysis-policy current]
    python -m werewolf.cli.experiment rebuild-index

Paid execution is CLI-only, sequential, and fail-closed. The browser
surfaces are read-only and can never launch execution.
"""
from __future__ import annotations

import argparse
import json
import sys

from werewolf.cli.run_game import load_env_file, setup_logging
from werewolf.experiments.conditions import build_crossed_conditions
from werewolf.experiments.exports import write_summary_exports
from werewolf.experiments.index import rebuild_experiment_index
from werewolf.experiments.manifest import (
    DEFAULT_EXPERIMENTS_ROOT,
    ManifestError,
    validate_manifest,
    write_manifest,
)
from werewolf.experiments.runner import (
    ExperimentRunError,
    build_experiment_manifest,
    default_crossed_comparisons,
    run_experiment,
    validate_manifest_for_execution,
)
from werewolf.experiments.summaries import summarize_experiment


def _add_root(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--root", default=str(DEFAULT_EXPERIMENTS_ROOT),
        help="Experiments storage root (default: outputs/experiments)",
    )


def cmd_validate(args) -> int:
    try:
        with open(args.manifest, encoding="utf-8") as f:
            manifest = json.load(f)
    except (OSError, ValueError) as exc:
        print(f"Cannot read manifest: {exc}", file=sys.stderr)
        return 2
    errors = validate_manifest(manifest)
    if not errors:
        errors = validate_manifest_for_execution(manifest)
    if errors:
        print("Manifest is INVALID:")
        for error in errors:
            print(f"  - {error}")
        return 1
    print("Manifest is valid.")
    print(f"  experiment_id:          {manifest['experiment_id']}")
    print(f"  manifest_content_sha256: "
          f"{manifest['manifest_content_sha256']}")
    print(f"  scheduled trials:       "
          f"{len(manifest['execution_contract']['schedule'])}")
    return 0


def cmd_create_crossed(args) -> int:
    if args.seed_file:
        with open(args.seed_file, encoding="utf-8") as f:
            seeds = list(json.load(f))
    else:
        seeds = list(range(args.seed_start,
                           args.seed_start + args.num_seeds))
    try:
        manifest = build_experiment_manifest(
            experiment_id=args.experiment_id,
            conditions=build_crossed_conditions(args.model_a, args.model_b),
            seeds=seeds,
            repetitions=args.repetitions,
            description=(
                args.description
                or f"Crossed {args.model_a} vs {args.model_b}"
            ),
            game={
                "n_players": args.n,
                "n_wolves": args.wolves,
                "n_seers": args.seers,
                "discussion_cycles": args.discussion_cycles,
                "belief_snapshots": not args.no_belief_snapshots,
            },
            generation=(
                {"max_output_tokens": args.max_output_tokens}
                if args.max_output_tokens else None
            ),
            scheduler_seed=args.scheduler_seed,
            comparisons=default_crossed_comparisons(),
        )
        path = write_manifest(args.root, manifest)
    except ManifestError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    rebuild_experiment_index(args.root)
    print(f"Created experiment {args.experiment_id}")
    print(f"  manifest: {path}")
    print(f"  manifest_content_sha256: "
          f"{manifest['manifest_content_sha256']}")
    print(f"  scheduled trials: "
          f"{len(manifest['execution_contract']['schedule'])}")
    print(f"Run it with: python -m werewolf.cli.experiment run "
          f"{args.experiment_id}")
    return 0


def cmd_run(args) -> int:
    load_env_file()
    setup_logging(args.debug)
    try:
        counts = run_experiment(
            args.root, args.experiment_id,
            resume=args.resume,
            retry_failed=args.retry_failed,
            allow_adjusted_health=args.allow_adjusted_health,
        )
    except (ExperimentRunError, ManifestError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    finally:
        try:
            rebuild_experiment_index(args.root)
        except Exception:
            pass  # the index is derived; a failed refresh loses nothing
    print(
        f"Run finished: {counts['completed']} completed, "
        f"{counts['failed']} failed, {counts['interrupted']} interrupted, "
        f"{counts['recovered']} recovered, "
        f"{counts['skipped_exhausted']} exhausted."
    )
    return 0


def cmd_summarize(args) -> int:
    try:
        result = summarize_experiment(
            args.root, args.experiment_id,
            analysis_policy=args.analysis_policy,
            exporter=lambda root, experiment_id, payload:
                write_summary_exports(root, experiment_id, payload),
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    rebuild_experiment_index(args.root)
    if result["created"]:
        print(f"Created summary revision {result['revision']}")
        print(f"  path: {result['path']}")
    else:
        print(
            f"No-op: revision {result['revision']} already covers this "
            "exact input"
        )
    print(f"  summary_input_sha256: {result['summary_input_sha256']}")
    return 0


def cmd_rebuild_index(args) -> int:
    index = rebuild_experiment_index(args.root)
    print(f"Rebuilt index with {len(index['experiments'])} experiment(s).")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m werewolf.cli.experiment",
        description="Reproducible multi-game benchmark experiments",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("validate", help="Validate a manifest file")
    p.add_argument("manifest", help="Path to a manifest.json")
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("create-crossed",
                       help="Create a crossed A/B experiment manifest")
    p.add_argument("--experiment-id", required=True)
    p.add_argument("--model-a", required=True)
    p.add_argument("--model-b", required=True)
    p.add_argument("--seed-start", type=int, default=5000)
    p.add_argument("--num-seeds", type=int, default=10)
    p.add_argument("--seed-file", default=None,
                   help="JSON file with an explicit seed list")
    p.add_argument("--repetitions", type=int, default=2)
    p.add_argument("--n", type=int, default=7)
    p.add_argument("--wolves", type=int, default=2)
    p.add_argument("--seers", type=int, default=1)
    p.add_argument("--discussion-cycles", type=int, default=2)
    p.add_argument("--no-belief-snapshots", action="store_true")
    p.add_argument("--max-output-tokens", type=int, default=None)
    p.add_argument("--scheduler-seed", type=int, default=0)
    p.add_argument("--description", default=None)
    _add_root(p)
    p.set_defaults(func=cmd_create_crossed)

    p = sub.add_parser("run", help="Execute a created experiment")
    p.add_argument("experiment_id")
    p.add_argument("--resume", action="store_true",
                   help="Continue an experiment with existing records")
    p.add_argument("--retry-failed", action="store_true",
                   help="Grant one extra attempt to exhausted trials")
    p.add_argument("--allow-adjusted-health", action="store_true",
                   help="Accept predeclared health adjustments")
    p.add_argument("--debug", action="store_true")
    _add_root(p)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("summarize",
                       help="Build an immutable summary revision")
    p.add_argument("experiment_id")
    p.add_argument("--analysis-policy", choices=("pinned", "current"),
                   default="pinned")
    _add_root(p)
    p.set_defaults(func=cmd_summarize)

    p = sub.add_parser("rebuild-index",
                       help="Rebuild the derived experiment index")
    _add_root(p)
    p.set_defaults(func=cmd_rebuild_index)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
