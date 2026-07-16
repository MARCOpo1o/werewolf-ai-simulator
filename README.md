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
| `fast` | grok-4.3, provider-default reasoning | xAI (exact billing) | default |
| `reasoning` | grok-4.3, reasoning effort `low` | xAI (exact billing) | |
| `gemini_flash_lite` | gemini-3.1-flash-lite | LiteLLM (estimate) | cheapest, ~$0.01/game |
| `gemini_flash` | gemini-3.5-flash, thinking capped `low` | LiteLLM (estimate) | |
| `claude_haiku` | claude-haiku-4-5-20251001 | LiteLLM (estimate) | date-pinned |
| `claude_sonnet` | claude-sonnet-5 | LiteLLM (estimate) | |
| `gpt_nano` | gpt-5.4-nano-2026-03-17 | LiteLLM (estimate) | date-pinned |
| `gpt_luna` | gpt-5.6-luna, reasoning effort `low` | LiteLLM (estimate) | experimental |

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

Open [http://localhost:5000](http://localhost:5000). The setup supports a Quick homogeneous game or a Model Matchup with independent Werewolf, Villager, and Seer models. Custom settings expose generation controls, discussion cycles, and belief instrumentation. Optional health checks make exactly one provider request and report model identity, JSON validity, detected parameter adjustments, usage, and cost before a game starts.

The JSON API exposes `/api/models`, `/api/models/<alias>/health-check`, `/api/new`, `/api/advance`, `/api/state`, and `/api/usage`. Web game creation accepts curated aliases only; CLI tools continue to support full provider model IDs.

Completed and interrupted games remain available at `/games`. Each `/games/<game_id>` page is a persisted, single-game forensic report with a filterable timeline, belief changes, decision attempts, reliability diagnostics, cost accounting, and reproducibility metadata. Reports default to a server-generated spoiler-safe projection. Revealing private data refetches the report from the server; it is spoiler protection for this trusted local app, not an authorization boundary.

The related APIs are:

- `GET /api/games` — stable cursor-paginated game history
- `GET /api/games/<game_id>/report` — spoiler-safe report by default
- `GET /api/games/<game_id>/report?include_private=true` — complete forensic report, with `Cache-Control: no-store`
- `GET /api/games/<game_id>/raw` — canonical JSONL download, with `Cache-Control: no-store`

JSONL remains canonical. Rebuildable metadata, report sidecars, and `outputs/games/index.json` only accelerate history and report reads. Reconciliation runs once on first history access, incrementally after a game completes or a report is requested, and explicitly through `GameRepository.rebuild()` for tests and maintenance. Normal history requests read the derived index instead of rescanning every log. See [the single-game report contract](docs/single-game-report.md) for status, privacy, storage, and compatibility details.

## Running formal model-assignment benchmark experiments (CLI)

The PR 3A experiment runner is the reproducible, CLI-only path for multi-game
model-assignment comparisons. It materializes a versioned manifest, runs
conditions sequentially with deterministic seed-block ordering, journals every
attempt, and writes immutable aggregate summary revisions:

```bash
python3 -m werewolf.cli.experiment create-crossed \
  --experiment-id pilot_a_vs_b \
  --model-a gemini_flash_lite --model-b fast \
  --num-seeds 10 --seed-start 42001 --repetitions 2
python3 -m werewolf.cli.experiment run pilot_a_vs_b
python3 -m werewolf.cli.experiment summarize pilot_a_vs_b
```

Use `--resume` after an interruption and `--analysis-policy current` only when
you intentionally want a new summary revision under the current analysis
implementation. The web UI at `/experiments` is read-only: it shows persisted
experiment history, summary revisions, metrics, declared comparisons, exports,
and links to individual forensic reports. It cannot start paid work.

See [the multi-game benchmark contract](docs/multi-game-benchmark.md) for
storage, crash recovery, source integrity, uncertainty, and interpretation
limits.

## Running batch trials (CLI)

```bash
python -m werewolf.cli.run_trials --trials 200 --seed-start 1000 --n 7 --wolves 2 --seers 0 --quiet
```

Writes per-game JSONL logs to `outputs/games/`, a trial manifest (appended per trial, so a crash loses nothing), and JSON/CSV summaries. A preflight health check runs 5 games by default; its cost is reported separately. The progress bar shows live cumulative cost, and the batch summary includes total cost, mean/median/P90/min/max cost per game, token totals, retry/fallback counts, cost-source breakdown, and a model-registry snapshot for reproducibility.

## Usage & cost accounting

Every LLM call attempt — including malformed responses, invalid actions, retries, and provider failures — produces an `llm_call` record in the per-game JSONL log (schema in `werewolf/llm/records.py`), with:

- token counts (input / cached / output / reasoning)
- cost with an explicit source: `provider_reported` (exact, xAI ticks), `pricing_table_estimate` (LiteLLM price map), or `unavailable` — estimates are never presented as exact, and unavailable cost is never silently treated as zero
- requested vs. resolved model (providers can silently redirect retired slugs), prompt version hash, error category, parse method, latency

Each game log ends with a `usage_summary` record: totals plus cost by player, role, phase, and action.

## Belief snapshots & manipulation metrics

Every game (unless run with `--no-belief-snapshots`) privately asks each player twice per day — once before discussion (`assess_beliefs` action) and once inside the vote response — for a structured assessment: per-player wolf probabilities, intended vote, vote confidence, most influential recent speaker, and (wolves only) second-order estimates of how suspicious each player is of them. Snapshots are logged as moderator-only `belief_snapshot` events (schema in `werewolf/engine/beliefs.py`); they are never shown to other players and never affect the game — a valid vote with malformed beliefs still counts, and missing snapshots are recorded as missing, never imputed.

Six metrics are computed from the logs (`werewolf/evaluation/belief_metrics.py`): belief shift toward wolves, harmful revision rate, beneficial revision rate (always reported together), vote-belief alignment plus intention-action gap, Brier calibration per checkpoint, and wolf suspicion-awareness error (numerical theory-of-mind). Batch summaries include the aggregates automatically; re-analyze any logs with:

```bash
python -m werewolf.cli.analyze --manifest outputs/games/trials_manifest_<run_id>.jsonl
```

Known limitation: eliciting probabilities is itself an intervention and may influence play. Instrumentation is identical across all models and roles, so cross-model comparisons remain valid; `--no-belief-snapshots` preserves the uninstrumented baseline.

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

- Replayable checkpoints and counterfactual branches (inject different deceptive arguments into one exact state, measure causal belief shifts)
- Multilingual and hidden-adversary replications after the baseline protocol is stable

## Project structure

```
werewolf/
  __main__.py           # CLI entry (python -m werewolf)
  web/
    app.py              # Live game, history, report UI, and JSON APIs
  reporting/
    repository.py       # Rebuildable game index and atomic derived storage
    builder.py          # Versioned single-game forensic report
    privacy.py          # Allowlisted spoiler-safe report projection
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
