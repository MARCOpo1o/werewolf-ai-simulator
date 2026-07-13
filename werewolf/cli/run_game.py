import argparse
import logging
import os
import sys
from pathlib import Path

from werewolf.llm.provider import GenerationConfig
from werewolf.llm.registry import (
    MODEL_REGISTRY,
    build_provider,
    get_api_key as _registry_key,
    resolve,
)

# Backward-compatible alias map, derived from the registry (single source
# of truth in werewolf/llm/registry.py).
MODEL_PRESETS = {alias: spec.model for alias, spec in MODEL_REGISTRY.items()}


def load_env_file():
    env_path = Path(__file__).parent.parent.parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    if key not in os.environ:
                        os.environ[key] = value


def setup_logging(debug: bool):
    level = logging.DEBUG if debug else logging.WARNING
    
    logging.basicConfig(
        level=level,
        format="[%(levelname)s] %(name)s: %(message)s"
    )
    
    if debug:
        print("[DEBUG MODE ENABLED - verbose logging active]")


def get_api_key(model: str = "fast") -> str:
    """Key for the given model alias/ID, from the env vars its registry
    spec names (default mirrors the legacy GROK_API_KEY/XAI_API_KEY lookup)."""
    return _registry_key(resolve(model))


def main():
    parser = argparse.ArgumentParser(
        description="Run a Werewolf game with AI agents powered by Grok"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--n",
        type=int,
        default=7,
        help="Number of players (default: 7)"
    )
    parser.add_argument(
        "--wolves",
        type=int,
        default=2,
        help="Number of werewolves (default: 2)"
    )
    parser.add_argument(
        "--seers",
        type=int,
        default=1,
        help="Number of seers (0 or 1, default: 1)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/games",
        help="Output directory for game logs (default: outputs/games)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="fast",
        help=f"Model preset or full name. Presets: {list(MODEL_PRESETS.keys())} (default: fast)"
    )
    parser.add_argument(
        "--hide-thoughts",
        action="store_true",
        help="Hide moderator-only channel (agent thoughts) from console output"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging for troubleshooting"
    )
    parser.add_argument(
        "--show-prompts",
        action="store_true",
        help="Show full prompts sent to each agent (requires --debug)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress transcript output for faster batch-style runs"
    )
    parser.add_argument(
        "--no-belief-snapshots",
        action="store_true",
        help="Disable structured belief/suspicion snapshots (cheaper, but "
             "games cannot be analyzed for manipulation metrics)"
    )
    parser.add_argument("--discussion-cycles", type=int, default=2,
                        help="Discussion cycles per day (default: 2; "
                             "order reverses between cycles)")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Sampling temperature (default: provider default)")
    parser.add_argument("--top-p", type=float, default=None,
                        help="Nucleus sampling top_p (default: provider default)")
    parser.add_argument("--max-output-tokens", type=int, default=None,
                        help="Max output tokens per call (default: provider default)")
    parser.add_argument("--provider-seed", type=int, default=None,
                        help="Provider-side sampling seed, where supported")

    args = parser.parse_args()

    load_env_file()
    setup_logging(args.debug)

    spec = resolve(args.model)
    model_name = spec.model
    model_alias = spec.alias

    api_key = get_api_key(args.model)
    if not api_key:
        env_names = " or ".join(spec.api_key_env)
        print(f"Error: {env_names} environment variable is not set.")
        print(f"Please set it, e.g. export {spec.api_key_env[0]}=your_api_key")
        sys.exit(1)

    if args.wolves >= args.n:
        print(f"Error: Number of wolves ({args.wolves}) must be less than total players ({args.n})")
        sys.exit(1)

    if args.n < 3:
        print("Error: Need at least 3 players")
        sys.exit(1)

    if args.seers not in (0, 1):
        print("Error: Number of seers must be 0 or 1")
        sys.exit(1)

    if args.wolves + args.seers >= args.n:
        print(f"Error: Need at least 1 villager. Got wolves={args.wolves}, seers={args.seers}, players={args.n}")
        sys.exit(1)

    from werewolf.engine.game import GameEngine

    if not args.quiet:
        print(f"Starting Werewolf game with {args.n} players ({args.wolves} wolves, {args.seers} seers)")
        print(f"Seed: {args.seed}")
        print(f"Model: {model_name}")
        print(f"Output: {args.output_dir}")
        print()

    provider_result = build_provider(spec, api_key=api_key)
    if not provider_result.ok:
        raise SystemExit(
            f"Error: provider unavailable ({provider_result.status.value}): "
            f"{provider_result.error or 'unknown initialization error'}"
        )

    engine = GameEngine(
        n_players=args.n,
        n_wolves=args.wolves,
        n_seers=args.seers,
        seed=args.seed,
        output_dir=args.output_dir,
        api_key=api_key,
        model=model_name,
        show_all_channels=not args.hide_thoughts,
        show_prompts=args.show_prompts and args.debug,
        transcript_enabled=not args.quiet,
        # Build the provider from the resolved spec so aliases route to
        # the right provider (gemini_* -> LiteLLM, fast/reasoning -> xAI).
        provider=provider_result.provider,
        model_alias=model_alias,
        belief_snapshots=not args.no_belief_snapshots,
        generation_config=GenerationConfig(
            temperature=args.temperature,
            top_p=args.top_p,
            max_output_tokens=args.max_output_tokens,
            provider_seed=args.provider_seed,
        ),
        discussion_cycles=args.discussion_cycles,
    )

    winner = engine.run()

    print(f"\nGame complete! Winner: {winner}")
    print(f"Log saved to: {engine.logger.filepath}")

    summary = engine.ledger.game_summary()
    if summary["calls"]:
        print(
            f"LLM calls: {summary['calls']} "
            f"(retries: {summary['retries']}, fallbacks: {summary['fallbacks']})"
        )
        cost = summary["cost_usd_total"]
        if cost is None:
            print("Cost: unavailable (provider reported no cost data)")
        else:
            exact = "" if summary["cost_complete"] else " (incomplete)"
            print(f"Cost: ${cost:.6f}{exact} "
                  f"[sources: {', '.join(summary['cost_by_source'])}]")


if __name__ == "__main__":
    main()
