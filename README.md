# Werewolf Multi-Agent Simulator

A minimal, locally runnable Python simulation of the Werewolf (Mafia) party game where all players are AI agents powered by the Grok API.

## Features

- All players are AI agents with role-specific objectives
- Each agent has persistent memory across turns
- Strict moderator orchestration ensures fair play
- Web UI to run and step through a single game
- JSONL logging for analysis and replay

## Requirements

- Python 3.10+
- Grok API key (from x.ai)

## Setup

1. Create a `.env` file in the project root with your Grok API key (e.g. copy `.env.example` to `.env` and fill it in):

```bash
GROK_API_KEY=your_api_key_here
```

The app also accepts `XAI_API_KEY`. The `.env` file is gitignored so the key is never committed.

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the game (Web UI)

From the project root:

```bash
python -m werewolf.web.app
```

Then open [http://localhost:5000](http://localhost:5000) in your browser. Use the UI to start a new game (choose players, wolves, seed), then advance through night and day phases step by step.

### Screenshots

| Game setup | In-game (phase / transcript) |
|------------|------------------------------|
| ![Setup](screenshots/Screenshot%202026-02-06%20at%207.48.48%20PM.png) | ![Game](screenshots/Screenshot%202026-02-06%20at%207.59.01%20PM.png) |

## Game rules (summary)

- **Roles**: Werewolves (know each other, kill at night), Seer (divine one player per night), Villagers (deduce wolves).
- **Phases**: Night (wolf chat → wolf kill → seer divine), Day (announce victim → discussion → vote).
- **Win**: Village wins when all wolves are eliminated; wolves win when they outnumber or equal villagers.

## Outputs and repo hygiene

- Game logs are written to `outputs/games/` (JSONL per game).
- The repo already ignores `outputs/` in `.gitignore`. Keeping it ignored is recommended: logs are regeneratable, can be large, and are often environment-specific when running at scale.

## Next steps

Planned follow-up: support **running many games at scale** (e.g. batch or headless runs) and **analyzing results** (aggregate win rates, role performance, etc.). The web UI is for single-game inspection and debugging before scaling.

## Project structure

```
werewolf/
  __main__.py           # CLI entry (python -m werewolf)
  web/
    app.py              # Web UI server (python -m werewolf.web.app)
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
