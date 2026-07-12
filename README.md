# Werewolf AI Simulator — an LLM Deception Benchmark

A multi-agent Werewolf (Mafia) game where all players are LLM agents. The research goal is to **measure how AI agents deceive, manipulate, and resist manipulation**: who to trust, when to bluff, and how persuasion plays out when every player is an LLM.

Supports multiple model providers with **per-call cost accounting**: xAI Grok models via the native SDK (exact provider-reported billing), and Gemini/OpenAI/Anthropic/OpenRouter models via LiteLLM (clearly-labeled cost estimates).

## Requirements

- Python 3.10+
- An API key for at least one provider (xAI or Google Gemini)

## Setup

1. Create a `.env` file in the project root (copy `.env.example`) with the keys for the models you'll use:

```bash
GROK_API_KEY=...      # or XAI_API_KEY, for grok models
GEMINI_API_KEY=...    # for gemini models
```

`.env` is gitignored so keys are never committed.

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Verify a model works end to end (makes exactly one tiny API request):

```bash
python3 scripts/smoke_test_model.py gemini_flash_lite
```

## Models

Model aliases live in `werewolf/llm/registry.py`:

| Alias | Model | Provider | Notes |
|---|---|---|---|
| `fast` | grok-4.3, reasoning effort `none` | xAI (exact billing) | default |
| `reasoning` | grok-4.3, reasoning effort `low` | xAI (exact billing) | |
| `gemini_flash_lite` | gemini-3.1-flash-lite | LiteLLM (estimate) | cheapest, ~$0.01/game |
| `gemini_flash` | gemini-3.5-flash, thinking capped `low` | LiteLLM (estimate) | |

Full model IDs also work: bare IDs (`--model grok-4.5`) go to xAI; prefixed IDs (`--model gemini/<model>`) go through LiteLLM.

## Running the game (CLI)

```bash
python -m werewolf --n 7 --wolves 2 --seers 1 --seed 42 --model gemini_flash_lite
```

Useful flags: `--seers 0`, `--quiet`, `--model <alias-or-id>`. Every game ends with a cost report:

```
LLM calls: 23 (retries: 1, fallbacks: 0)
Cost: $0.011240 [sources: pricing_table_estimate]
```

## Running the game (Web UI)

```bash
python -m werewolf.web.app
```

Open [http://localhost:5000](http://localhost:5000). The JSON API (`/api/new`, `/api/advance`, `/api/state`, `/api/usage`) accepts any model alias and exposes the live usage/cost summary, so games can also be driven programmatically.

## Running batch trials (CLI)

```bash
python -m werewolf.cli.run_trials --trials 200 --seed-start 1000 --n 7 --wolves 2 --seers 0 --quiet
```

Writes per-game JSONL logs to `outputs/games/`, a trial manifest, and JSON/CSV summaries. A preflight health check runs 5 games by default.

## Usage & cost accounting

Every LLM call attempt — including malformed responses, invalid actions, retries, and provider failures — produces an `llm_call` record in the per-game JSONL log (schema in `werewolf/llm/records.py`), with:

- token counts (input / cached / output / reasoning)
- cost with an explicit source: `provider_reported` (exact, xAI ticks), `pricing_table_estimate` (LiteLLM price map), or `unavailable` — estimates are never presented as exact, and unavailable cost is never silently treated as zero
- requested vs. resolved model (providers can silently redirect retired slugs), prompt version hash, error category, parse method, latency

Each game log ends with a `usage_summary` record: totals plus cost by player, role, phase, and action.

## Running tests

The full suite runs free — no API key, no network (LLM calls are simulated by a fake provider):

```bash
PYTHONPATH=. python3 -m unittest discover -s tests -v
```

Live paid tests are opt-in only via `scripts/smoke_test_model.py`.

## Game rules (summary)

- **Roles**: Werewolves (know each other, kill at night), Seer (divine one player per night), Villagers (deduce wolves).
- **Phases**: Night (wolf chat → wolf kill → seer divine), Day (announce victim → discussion → vote).
- **Win**: Village wins when all wolves are eliminated; wolves win when they outnumber or equal villagers.

### Screenshots

| Game setup | In-game (phase / transcript) |
|------------|------------------------------|
| ![Setup](screenshots/setup.png) | ![Game](screenshots/gameplay.png) |

## Outputs and repo hygiene

Game logs are written to `outputs/games/` (JSONL per game, gitignored): regeneratable, potentially large, environment-specific.

## Next steps

- Batch-level cost aggregation in trial manifests and summaries
- Heterogeneous games (different models/prompts per player or role)
- Deception metrics computed offline from game logs (lie rate, suspicion accuracy, persuasion success)
- Pre-run cost estimation from historical game records

## Project structure

```
werewolf/
  __main__.py           # CLI entry (python -m werewolf)
  web/
    app.py              # Web UI server + JSON API (incl. /api/usage)
  cli/
    run_game.py         # Single-game CLI
    run_trials.py       # Batch trial runner + aggregate summaries
  engine/
    game.py             # Game loop
    state.py            # GameState, PlayerState
    visibility.py       # Observation building
    logging.py          # Per-game JSONL logs (events, llm_call, usage_summary)
  agents/
    ai_agent.py         # Prompting, parsing, retries (provider-agnostic)
    prompts.py          # Role prompts (content-hashed for reproducibility)
  llm/
    registry.py         # Model aliases -> provider, model ID, key env vars
    provider.py         # Provider protocol (typed request/result)
    xai_provider.py     # Direct xAI adapter (exact cost_in_usd_ticks)
    litellm_provider.py # Gemini/OpenAI/Anthropic/... adapter (estimates)
    records.py          # UsageRecord schema (one per call attempt)
    ledger.py           # Thread-safe ledger + per-game aggregation
scripts/
  smoke_test_model.py   # One-request live check for any model alias
outputs/
  games/                # JSONL logs (gitignored)
```
