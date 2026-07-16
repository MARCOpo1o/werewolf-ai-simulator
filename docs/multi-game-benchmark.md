# Multi-game model-assignment benchmark experiments

PR 3A compares controlled collections of games whose conditions vary model
assignments by role. It does not assign a universal model score, launch paid
work from the browser, or turn observed dialogue associations into causal
persuasion claims.

## Implemented scope

An explicit condition currently contains a complete Werewolf, Villager, and
Seer model assignment plus an optional description. Game rules, prompt profile,
and generation settings are shared by every condition in one experiment. The
only supported prompt profile is `baseline_v1`, which preserves the existing
game prompts. Per-condition prompt, reasoning, and generation overrides are a
later extension and are not accepted by the manifest validator.

Experiment game logs are canonical under
`outputs/experiments/<experiment_id>/games/`. The experiment report links these
logs to the single-game forensic report API, but they are not added to the
global `outputs/games/index.json` history in PR 3A. Unifying those stores is
deferred rather than silently copying canonical evidence.

## Create, validate, run, and summarize

Create a crossed A/B manifest:

```bash
python3 -m werewolf.cli.experiment create-crossed \
  --experiment-id pilot_a_vs_b \
  --model-a gemini_flash_lite --model-b fast \
  --num-seeds 10 --seed-start 42001 --repetitions 2
```

Then run the formal, sequential executor and derive one immutable summary:

```bash
python3 -m werewolf.cli.experiment validate outputs/experiments/pilot_a_vs_b/manifest.json
python3 -m werewolf.cli.experiment run pilot_a_vs_b
python3 -m werewolf.cli.experiment summarize pilot_a_vs_b
```

Use `--resume` after interruption. The runner reconciles an open attempt from
its canonical game log before scheduling more paid work. It does not depend on
the PR 2 report builder to decide whether an already-paid game completed.

`--retry-failed` grants one additional attempt to an exhausted trial. It does
not erase failed or interrupted attempts; their evidence and known costs remain
part of operational accounting.

## Canonical records

```text
outputs/experiments/<experiment_id>/
  manifest.json       canonical configuration
  trials.jsonl        canonical lifecycle journal
  games/*.jsonl       canonical game and provider evidence
  summaries/*.json    immutable derived analysis revisions
  summary.json        rebuildable revision catalog
  exports/.../*.csv   rebuildable tabular exports
```

The manifest and every lifecycle record carry hashes. Execution and analysis
contracts are deliberately separate:

- Execution changes—engine behavior, prompts, provider routing, generation,
  retry/fallback policy, schedule, or relevant dependencies—block resume.
- Analysis changes create a new summary revision; they do not block execution.

A completed, failed, or interrupted attempt records the final game-log SHA-256
when a log exists. Summarization reads each source once, hashes those exact
bytes, and excludes missing or modified sources from authoritative totals.

## Formal-run policy

Formal experiments default to no provider fallback, abort a game on a strategic
action failure, cap a game at 20 rounds, and pin provider timeout, retry,
generation, context, output, and message-memory limits in the manifest.

Each execution session runs one provider health probe for every unique
model/effective-generation fingerprint. A detected provider adjustment is
accepted only when its exact fingerprint was predeclared in the manifest and
the runner receives `--allow-adjusted-health`.

The browser is read-only. `/experiments` displays persisted results and links
each verified trial to its PR 2 forensic report; it cannot create, resume, or
run an experiment.

## Analysis views and statistics

Completed games appear in overlapping views:

- `all_completed`
- `clean_eligible`
- `completed_not_clean_eligible`

Operational attempts are separate from scheduled trials. A trial that fails
twice then completes is one completed trial and three attempts. Failed work and
health-check cost remain visible in operational accounting.

Game outcomes use game-weighted point estimates. Revision, alignment, retry,
repair, and decision-group metrics use pooled eligible observations. Brier and
ECE are prediction-weighted. All uncertainty intervals resample seed clusters,
retaining all repetitions within each sampled seed.

The version-1 bootstrap uses 2,000 deterministic resamples and 95% percentile
intervals. Fewer than five seed clusters produce an estimate labeled
`insufficient_clusters`, not a misleading interval. Paired comparisons require
shared eligible seed clusters with matching role-map and game-rule hashes; they
are never silently downgraded to independent comparisons. Intervals are
descriptive, no multiplicity correction is applied, and they are not
significance declarations.

## Limits of interpretation

Team outcomes by assigned model and role are descriptive in mixed teams, not
causal attribution. The benchmark reports belief movement, harmful revision,
retention, alignment, suspicion awareness, reliability, latency, and cost, but
does not claim that a particular message caused an observed belief change.

Controlled forked interventions remain a future causal benchmark layer.

PR 3A is therefore a model-assignment benchmark layer, not yet the complete
arbitrary-condition benchmark. Per-condition prompts and generation policies,
global game-history integration, and causal intervention analysis remain
explicitly outside this implementation.
