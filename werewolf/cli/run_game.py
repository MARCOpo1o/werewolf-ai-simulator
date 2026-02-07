import argparse
import logging
import os
import sys
from pathlib import Path

MODEL_PRESETS = {
    "fast": "grok-4-1-fast",
    "reasoning": "grok-4-1-fast-reasoning",
}


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

    args = parser.parse_args()

    load_env_file()
    setup_logging(args.debug)

    model_name = MODEL_PRESETS.get(args.model, args.model)

    api_key = os.environ.get("GROK_API_KEY")
    if not api_key:
        print("Error: GROK_API_KEY environment variable is not set.")
        print("Please set it with: export GROK_API_KEY=your_api_key")
        sys.exit(1)

    if args.wolves >= args.n:
        print(f"Error: Number of wolves ({args.wolves}) must be less than total players ({args.n})")
        sys.exit(1)

    if args.n < 3:
        print("Error: Need at least 3 players")
        sys.exit(1)

    from werewolf.engine.game import GameEngine

    print(f"Starting Werewolf game with {args.n} players ({args.wolves} wolves)")
    print(f"Seed: {args.seed}")
    print(f"Model: {model_name}")
    print(f"Output: {args.output_dir}")
    print()

    engine = GameEngine(
        n_players=args.n,
        n_wolves=args.wolves,
        seed=args.seed,
        output_dir=args.output_dir,
        api_key=api_key,
        model=model_name,
        show_all_channels=not args.hide_thoughts,
        show_prompts=args.show_prompts and args.debug
    )

    winner = engine.run()

    print(f"\nGame complete! Winner: {winner}")
    print(f"Log saved to: {engine.logger.filepath}")


if __name__ == "__main__":
    main()
