# Persisted single-game forensic reports

PR 2 explains one recorded game. It does not rank models, build leaderboards, estimate population-level calibration, or support cross-game benchmark claims.

## Canonical data and derived storage

The JSONL file at `outputs/games/<game_id>.jsonl` is canonical. Everything else is disposable and rebuildable:

```text
outputs/games/index.json
outputs/games/<game_id>.meta.json
outputs/games/<game_id>.report.json
```

Deleting the index or sidecars loses no game information. Reconciliation scans canonical JSONL files, rebuilds missing or stale sidecars, and ignores or removes orphaned derived files.

Reconciliation runs:

- once on application startup or first history access;
- incrementally for one game after completion or report access;
- through `GameRepository.rebuild()` for an explicit full rebuild in tests and maintenance.

Normal `/api/games` requests read the in-memory derived index and do not rescan every JSONL file. Index mutations are protected by an in-process re-entrant lock. Multi-process writers and distributed deployments are outside this storage contract.

Derived JSON uses a same-directory temporary file, `flush()`, `fsync()`, and `os.replace()`. Directory syncing is best-effort where supported. Sidecar freshness uses source size and `mtime_ns`; built reports also retain the canonical JSONL SHA-256.

History sorts descending by immutable `(created_at, game_id)`. Timestamp precedence is:

1. the canonical config/start record;
2. the earliest valid JSONL timestamp;
3. the filesystem timestamp.

Sidecars record `created_at_source`. A recovered canonical timestamp is never downgraded to a filesystem fallback. Pagination cursors are opaque, versioned encodings of the sort tuple rather than array offsets.

## Independent status dimensions

Reports keep separate status fields:

```text
completion_status: active | incomplete | completed
integrity_status: clean | warnings | corrupt
analysis_eligibility: eligible | limited | ineligible
usage_reliability: reliable | partial | inconsistent | unavailable
```

Only `incomplete` or `completed` is persisted. `active` is a runtime display overlay when the current in-memory engine owns that game ID, so a process crash cannot leave a stale active status in a sidecar.

A completed game may have integrity warnings or be unsuitable for strategic analysis. Conversely, a terminal usage-summary mismatch changes accounting reliability but does not by itself invalidate the transcript, beliefs, or decisions. Strategic eligibility changes only when canonical evidence is absent or unreliable, such as action events without required call records, concealed fallbacks, or untrusted model identity.

## Usage and cost precedence

`llm_call` records are authoritative for report totals. A terminal `usage_summary` is only a consistency check and never replaces recomputed values.

Cost reporting keeps exact, estimated, and unavailable calls distinct. Reports expose known cost, counts with and without known cost, completeness, and source categories. A partial total is never presented as the complete price of the game.

## Provenance and legacy logs

New events use a deterministic ID derived from their numeric sequence:

```text
event_id = evt_<zero-padded numeric id>
```

All events produced by one agent response share its `source_call_id`; each event still has its own `event_id`. Discussion events record `discussion_cycle` directly.

The parser preserves `source_line` and distinguishes exact, inferred, ambiguous, and unavailable links. Legacy decision links are inferred only when player, action, phase, round, ordering, and uniqueness support one candidate. Missing legacy fields remain unavailable rather than being fabricated.

## Privacy contract

`GET /api/games/<game_id>/report` returns an explicit allowlisted public projection. It excludes private thoughts, wolf chat, Seer results, role/team truth, private beliefs, player-model mappings, private call metadata, and ground-truth manipulation signals.

The report page reveals private information by refetching:

```text
GET /api/games/<game_id>/report?include_private=true
```

Private and raw responses use `Cache-Control: no-store`. The raw-log endpoint remains intentionally complete.

This is spoiler protection, not authentication. Anyone who can request `include_private=true` or the raw endpoint can access private game data. Remote or multi-user deployment requires authorization outside PR 2.

## Analysis boundaries

The report derives only signals present in structured records. It does not mine free-form thoughts with keyword rules or ask another model to judge them. Self-reported influential speakers appear only when explicitly present in the recorded belief schema.

Belief views may show predicted probabilities, binary outcomes, movement toward or away from truth, and per-prediction squared error. A value is labeled a Brier-score contribution only when the versioned belief schema defines it as a valid probability for that binary outcome. A single game does not support a calibration curve or a claim that one model is better calibrated.

Manipulation and resistance panels are descriptive, non-causal signals. Causal susceptibility and cross-model conclusions require controlled multi-game experiments and belong in later benchmark work.
