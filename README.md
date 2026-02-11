# Werewolf Multi-Agent Simulator

A multi-agent Werewolf (Mafia) game where all players are AI agents. The goal is to **observe how AI agents manipulate each other** and **how well they can lie** to other agents—who to trust, when to bluff, and how persuasion and deception play out when every player is an LLM.

Powered by the Grok API; run locally with a simple web UI or (later) at scale for experiments.

## Requirements

- Python 3.10+
- Grok API key (from x.ai)

## Setup

1. Create a `.env` file in the project root with your API key (e.g. copy `.env.example` to `.env` and fill it in):

```bash
GROK_API_KEY=your_api_key_here
```

The app also accepts `XAI_API_KEY`. The `.env` file is gitignored so the key is never committed.

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Running tests

From the project root:

```bash
PYTHONPATH=. python3 -m unittest tests.test_validate_action tests.test_roles_and_ids -v
```

Integration tests (`test_integration_trials.py`) require `GROK_API_KEY` or `XAI_API_KEY` in `.env` and will run real games.

## Running the game (Web UI)

From the project root:

```bash
python -m werewolf.web.app
```

Then open [http://localhost:5000](http://localhost:5000) in your browser. Use the UI to start a new game (choose players, wolves, seed), then advance through night and day phases step by step.

## Running the game (CLI)

```bash
python -m werewolf --n 7 --wolves 2 --seers 1 --seed 42
```

Useful flags:
- `--seers 0` disables the seer role.
- `--quiet` suppresses transcript output for faster runs.
- `--model fast|reasoning|<full-model-name>`

## Running batch trials (CLI)

```bash
python -m werewolf.cli.run_trials --trials 200 --seed-start 1000 --n 7 --wolves 2 --seers 0 --quiet
```

This command writes:
- Per-game JSONL logs in `outputs/games/`
- Trial manifest: `trials_manifest_<run_id>.jsonl`
- Aggregate summaries: `trials_summary_<run_id>.json` and `.csv`

By default, a preflight health check runs 5 games before the batch.

### Screenshots

| Game setup | In-game (phase / transcript) |
|------------|------------------------------|
| ![Setup](screenshots/setup.png) | ![Game](screenshots/gameplay.png) |

## Game rules (summary)

- **Roles**: Werewolves (know each other, kill at night), Seer (divine one player per night), Villagers (deduce wolves).
- **Phases**: Night (wolf chat → wolf kill → seer divine), Day (announce victim → discussion → vote).
- **Win**: Village wins when all wolves are eliminated; wolves win when they outnumber or equal villagers.

## Outputs and repo hygiene

- Game logs are written to `outputs/games/` (JSONL per game).
- The repo already ignores `outputs/` in `.gitignore`. Keeping it ignored is recommended: logs are regeneratable, can be large, and are often environment-specific when running at scale.

## Next steps

Planned follow-up: **run many games at scale** (batch or headless) and **analyze results**—e.g. how often wolves win, how persuasion and lying correlate with outcomes, and how different setups affect manipulation. The web UI is for single-game inspection and debugging before scaling.

## Project structure

```
werewolf/
  __main__.py           # CLI entry (python -m werewolf)
  web/
    app.py              # Web UI server (python -m werewolf.web.app)
  cli/
    run_trials.py       # Batch trial runner + aggregate summaries
  engine/
    game.py             # Game loop
    state.py            # GameState, PlayerState
    visibility.py       # Observation building
  agents/
    ai_agent.py         # Grok API + memory
    prompts.py          # Role prompts
outputs/
  games/                # JSONL logs (gitignored)
```
