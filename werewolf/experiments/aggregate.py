"""Aggregate multi-game analysis, version 1 ("aggregate-1").

Weighting policy (pinned by version):

- Game outcomes and game-level reliability rates are game-weighted.
- Revision, fallback-group, repair, alignment, and related rates pool
  eligible observation numerators and denominators.
- Brier and ECE are prediction-weighted primary estimates.
- Every belief aggregate reports prediction, game, and seed counts.
- All intervals resample seed CLUSTERS (repetitions of a sampled seed
  stay together), deterministically seeded, 2,000 samples, 95%.

Views overlap rather than partition: `all_completed`,
`clean_eligible`, and `completed_not_clean_eligible` all draw from
verified completed games; operational accounting covers every attempt.
Zero denominators, empty calibration bins, unavailable costs, and
missing paired seeds yield null, never fabricated zeros. With fewer
than five unique seed clusters an estimate is reported without an
interval (`interval_status: insufficient_clusters`).

No universal composite score exists: rankings stay metric-specific and
stratified, and persuasion measures remain descriptive.
"""
from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from typing import Callable, Optional

from werewolf.engine.beliefs import (
    CHECKPOINT_POST,
    CHECKPOINT_PRE,
    inspect_recorded_probability_map,
    recorded_belief_payload_valid,
)
from werewolf.evaluation.validity import classify_game
from werewolf.experiments.canonical import jcs_sha256
from werewolf.json_safety import as_mapping

AGGREGATE_ANALYSIS_VERSION = 1
COMPARISON_METHOD_VERSION = 1

DEFAULT_BOOTSTRAP = {"n_boot": 2000, "alpha": 0.05, "rng_seed": 0}
METRIC_WEIGHTING = "aggregate-1"

VIEW_ALL_COMPLETED = "all_completed"
VIEW_CLEAN = "clean_eligible"
VIEW_NOT_CLEAN = "completed_not_clean_eligible"

COMPARABLE_METRICS = ("village_win_rate", "wolf_win_rate")

MIN_SEED_CLUSTERS_FOR_INTERVAL = 5

ECE_BIN_COUNT = 10  # [0.0, 0.1), ..., [0.9, 1.0]


# --------------------------------------------------------------------------
# Deterministic seed-cluster bootstrap
# --------------------------------------------------------------------------

def derive_rng_seed(base_seed: int, *labels) -> int:
    digest = hashlib.sha256(
        jcs_sha256({"base": base_seed, "labels": list(labels)})
        .encode("utf-8")
    ).hexdigest()
    return int(digest[:16], 16)


def _pooled(obs_by_seed: dict, seeds) -> list:
    pooled = []
    for seed in seeds:
        pooled.extend(obs_by_seed[seed])
    return pooled


def cluster_bootstrap(
    obs_by_seed: dict,
    statistic: Callable,
    *,
    n_boot: int,
    alpha: float,
    rng_seed: int,
) -> Optional[dict]:
    """Percentile bootstrap resampling seed clusters; repetitions of a
    sampled seed always travel together."""
    obs_by_seed = {s: v for s, v in obs_by_seed.items() if v}
    seeds = sorted(obs_by_seed)
    if not seeds:
        return None
    estimate = statistic(_pooled(obs_by_seed, seeds))
    if estimate is None:
        return None
    result = {
        "estimate": estimate,
        "ci_low": None,
        "ci_high": None,
        "n_seeds": len(seeds),
        "n_boot": 0,
        "interval_status": "ok",
    }
    if len(seeds) < MIN_SEED_CLUSTERS_FOR_INTERVAL:
        result["interval_status"] = "insufficient_clusters"
        return result
    rng = random.Random(rng_seed)
    stats = []
    for _ in range(n_boot):
        sample = [rng.choice(seeds) for _ in seeds]
        value = statistic(_pooled(obs_by_seed, sample))
        if value is not None:
            stats.append(value)
    if not stats:
        result["interval_status"] = "no_resample_estimates"
        return result
    stats.sort()
    lo = int((alpha / 2) * len(stats))
    hi = min(len(stats) - 1, int((1 - alpha / 2) * len(stats)))
    result.update(ci_low=stats[lo], ci_high=stats[hi], n_boot=len(stats))
    return result


def mean_statistic(values: list) -> Optional[float]:
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def ratio_statistic(pairs: list) -> Optional[float]:
    numerator = sum(n for n, _ in pairs)
    denominator = sum(d for _, d in pairs)
    return (numerator / denominator) if denominator else None


# --------------------------------------------------------------------------
# Per-game evidence extraction
# --------------------------------------------------------------------------

def _decision_groups(rows: list) -> list:
    groups: dict = {}
    for row in rows:
        if row.get("type") != "llm_call":
            continue
        group = groups.setdefault(row.get("call_id"), {
            "records": [], "required_action": row.get("required_action"),
        })
        group["records"].append(row)
    out = []
    for group in groups.values():
        records = group["records"]
        api_attempts = [r for r in records if r.get("api_attempted")]
        out.append({
            "required_action": group["required_action"],
            "strategic": group["required_action"] != "assess_beliefs",
            "attempted": bool(api_attempts),
            "api_attempts": len(api_attempts),
            "ended_in_fallback": any(
                r.get("error_category") == "fallback_used" for r in records
            ),
            "completed": any(
                r.get("error_category") == "completed" for r in records
            ),
            "repaired": any(
                r.get("error_category") == "completed"
                and r.get("parse_method") in ("repaired", "regex")
                for r in records
            ),
        })
    return out


def _belief_evidence(rows: list, config: dict) -> dict:
    role_map = as_mapping(config.get("role_map"))
    wolves, village = set(), set()
    for pid_text, info in role_map.items():
        try:
            pid = int(pid_text)
        except (TypeError, ValueError):
            continue
        info = as_mapping(info)
        if info.get("role") == "werewolf":
            wolves.add(pid)
        elif info.get("team") == "village":
            village.add(pid)

    events = [r["event"] for r in rows
              if r.get("type") == "event" and isinstance(r.get("event"), dict)]
    snap: dict = {}
    emitted = valid = 0
    for event in events:
        if event.get("type") != "belief_snapshot":
            continue
        payload = as_mapping(event.get("payload"))
        checkpoint = payload.get("checkpoint")
        if checkpoint not in (CHECKPOINT_PRE, CHECKPOINT_POST):
            continue
        emitted += 1
        if recorded_belief_payload_valid(payload):
            valid += 1
            snap[(event.get("round"), event.get("speaker_id"),
                  checkpoint)] = payload

    votes: dict = {}
    total_votes = 0
    for event in events:
        if event.get("type") == "vote":
            payload = as_mapping(event.get("payload"))
            key = (event.get("round"), payload.get("voter_id"))
            if key not in votes:
                votes[key] = payload.get("target_id")
                if payload.get("voter_id") in village:
                    total_votes += 1

    def probabilities(payload) -> dict:
        parsed, _ = inspect_recorded_probability_map(
            payload.get("wolf_probabilities")
        )
        return parsed

    def argmax_set(pmap: dict) -> set:
        if not pmap:
            return set()
        top = max(pmap.values())
        return {pid for pid, p in pmap.items() if p == top}

    predictions = []           # (checkpoint, probability, is_wolf)
    movement = []              # post - pre toward real wolves
    initially_correct = harmful = 0
    initially_wrong = beneficial = 0
    aligned = alignment_n = 0
    awareness_errors = []

    rounds = sorted({key[0] for key in snap})
    for rnd in rounds:
        for villager in village:
            pre = snap.get((rnd, villager, CHECKPOINT_PRE))
            post = snap.get((rnd, villager, CHECKPOINT_POST))
            for checkpoint, payload in (
                (CHECKPOINT_PRE, pre), (CHECKPOINT_POST, post),
            ):
                if payload is None:
                    continue
                for pid, p in probabilities(payload).items():
                    predictions.append((checkpoint, p, pid in wolves))
            if pre is not None and post is not None:
                pre_p, post_p = probabilities(pre), probabilities(post)
                movement.extend(
                    post_p[w] - pre_p[w]
                    for w in wolves & set(pre_p) & set(post_p)
                )
                pre_top, post_top = argmax_set(pre_p), argmax_set(post_p)
                if pre_top and post_top:
                    if pre_top & wolves:
                        initially_correct += 1
                        if not (post_top & wolves):
                            harmful += 1
                    else:
                        initially_wrong += 1
                        if post_top & wolves:
                            beneficial += 1
            vote = votes.get((rnd, villager))
            if post is not None and vote is not None:
                top = argmax_set(probabilities(post))
                if top:
                    alignment_n += 1
                    if vote in top:
                        aligned += 1
        for wolf in wolves:
            for checkpoint in (CHECKPOINT_PRE, CHECKPOINT_POST):
                wolf_snap = snap.get((rnd, wolf, checkpoint))
                if wolf_snap is None:
                    continue
                estimates = {}
                for raw_id, value in as_mapping(
                    wolf_snap.get("estimated_suspicion_of_me")
                ).items():
                    if isinstance(value, bool):
                        continue
                    try:
                        observer, estimate = int(raw_id), float(value)
                    except (TypeError, ValueError):
                        continue
                    if 0.0 <= estimate <= 1.0:
                        estimates[observer] = estimate
                for villager in village:
                    v_snap = snap.get((rnd, villager, checkpoint))
                    if v_snap is None or villager not in estimates:
                        continue
                    actual = probabilities(v_snap).get(wolf)
                    if actual is not None:
                        awareness_errors.append(
                            abs(estimates[villager] - actual)
                        )

    return {
        "snapshot_coverage": {"emitted": emitted, "valid": valid},
        "predictions": predictions,
        "movement": movement,
        "initially_correct": initially_correct,
        "harmful": harmful,
        "initially_wrong": initially_wrong,
        "beneficial": beneficial,
        "aligned": aligned,
        "alignment_n": alignment_n,
        "village_votes": total_votes,
        "awareness_errors": awareness_errors,
    }


def _usage_evidence(rows: list) -> dict:
    api = [r for r in rows
           if r.get("type") == "llm_call" and r.get("api_attempted")]
    cost_usd = 0.0
    calls_with_cost = 0
    calls_with_unavailable_cost = 0
    tokens = defaultdict(int)
    latencies = []
    for record in api:
        cost = as_mapping(record.get("cost"))
        usd = cost.get("usd")
        if isinstance(usd, (int, float)) and not isinstance(usd, bool):
            cost_usd += usd
            calls_with_cost += 1
        if cost.get("source") == "unavailable":
            calls_with_unavailable_cost += 1
        for key, value in as_mapping(record.get("usage")).items():
            if isinstance(value, int) and not isinstance(value, bool):
                tokens[key] += value
        latency = record.get("latency_ms")
        if isinstance(latency, (int, float)) and not isinstance(
            latency, bool
        ) and latency >= 0:
            latencies.append(latency)
    return {
        "api_calls": len(api),
        "cost_usd": cost_usd if calls_with_cost else None,
        "cost_complete": calls_with_unavailable_cost == 0,
        "calls_with_unavailable_cost": calls_with_unavailable_cost,
        "tokens": dict(tokens),
        "latencies": latencies,
    }


def extract_game_evidence(source) -> dict:
    """Everything aggregate metrics need from one verified game."""
    rows = source.rows or []
    config = next(
        (r for r in rows if r.get("type") == "config"), {},
    )
    validity = classify_game(rows)
    terminal = source.terminal_record
    role_models = {}
    for role, info in as_mapping(config.get("role_models")).items():
        info = as_mapping(info)
        role_models[role] = (
            info.get("requested") or info.get("alias")
            or info.get("requested_model")
        )
    return {
        "trial_id": source.trial_id,
        "attempt_id": source.attempt_id,
        "game_id": source.game_id,
        "condition_id": source.condition_id,
        "seed": source.seed,
        "repetition": source.repetition,
        "winner": terminal.get("winner"),
        "rounds": terminal.get("rounds"),
        "recovered": bool(terminal.get("recovered")),
        "clean": validity["clean"],
        "violations": validity["violations"],
        "role_map_hash": jcs_sha256(as_mapping(config.get("role_map"))),
        "game_rules_hash": jcs_sha256({
            "n_players": config.get("n_players"),
            "n_wolves": config.get("n_wolves"),
            "n_seers": config.get("n_seers"),
        }),
        "role_models": role_models,
        "decision_groups": _decision_groups(rows),
        "belief": _belief_evidence(rows, config),
        "usage": _usage_evidence(rows),
    }


# --------------------------------------------------------------------------
# View metrics
# --------------------------------------------------------------------------

def _ece_bins(predictions: list) -> tuple:
    """predictions: (probability, is_wolf, game_id, seed). Returns
    (bins, ece) with prediction-weighted expected calibration error."""
    bins = [
        {
            "bin": f"[{i / 10:.1f}, {(i + 1) / 10:.1f}"
                   + ("]" if i == ECE_BIN_COUNT - 1 else ")"),
            "predictions": 0,
            "games": set(),
            "seeds": set(),
            "confidence_sum": 0.0,
            "hits": 0,
        }
        for i in range(ECE_BIN_COUNT)
    ]
    for probability, is_wolf, game_id, seed in predictions:
        index = min(int(probability * ECE_BIN_COUNT), ECE_BIN_COUNT - 1)
        entry = bins[index]
        entry["predictions"] += 1
        entry["games"].add(game_id)
        entry["seeds"].add(seed)
        entry["confidence_sum"] += probability
        entry["hits"] += 1 if is_wolf else 0
    total = sum(entry["predictions"] for entry in bins)
    ece = 0.0 if total else None
    rendered = []
    for entry in bins:
        n = entry["predictions"]
        mean_confidence = entry["confidence_sum"] / n if n else None
        empirical = entry["hits"] / n if n else None
        gap = abs(mean_confidence - empirical) if n else None
        if n and total:
            ece += (n / total) * gap
        rendered.append({
            "bin": entry["bin"],
            "prediction_count": n,
            "game_count": len(entry["games"]),
            "seed_count": len(entry["seeds"]),
            "mean_confidence": mean_confidence,
            "empirical_frequency": empirical,
            "absolute_gap": gap,
        })
    return rendered, ece


def _belief_counts(games: list, obs_key: Callable) -> dict:
    """prediction/game/seed counts for a belief aggregate."""
    n = games_with = 0
    seeds = set()
    for game in games:
        count = obs_key(game)
        if count:
            n += count
            games_with += 1
            seeds.add(game["seed"])
    return {"observations": n, "games": games_with, "seeds": len(seeds)}


def _view_metrics(games: list, bootstrap: dict, label: str) -> dict:
    """Metrics over one view (a list of extracted game evidence)."""
    base_seed = bootstrap["rng_seed"]
    n_boot, alpha = bootstrap["n_boot"], bootstrap["alpha"]

    def boot(obs_by_seed, statistic, *labels):
        return cluster_bootstrap(
            obs_by_seed, statistic, n_boot=n_boot, alpha=alpha,
            rng_seed=derive_rng_seed(base_seed, label, *labels),
        )

    def by_seed(value_fn) -> dict:
        out = defaultdict(list)
        for game in games:
            value = value_fn(game)
            if value is not None:
                out[game["seed"]].append(value)
        return out

    total = len(games)
    seeds = sorted({g["seed"] for g in games})
    metrics: dict = {
        "games": total,
        "seed_count": len(seeds),
        "evidence_coverage": {
            "games_with_belief_observations": sum(
                1 for g in games if g["belief"]["snapshot_coverage"]["valid"]
            ),
        },
    }
    if not total:
        return metrics

    # Game outcomes: game-weighted, seed-cluster intervals.
    for metric_id, win in (
        ("village_win_rate", "village"), ("wolf_win_rate", "wolf"),
    ):
        wins = sum(1 for g in games if g["winner"] == win)
        metrics[metric_id] = {
            "numerator": wins,
            "denominator": total,
            **(boot(
                by_seed(lambda g, w=win: 1.0 if g["winner"] == w else 0.0),
                mean_statistic, metric_id,
            ) or {}),
        }

    # Descriptive win rates by assigned model and role (never a ranking).
    by_model_role: dict = {}
    for game in games:
        assignments = {
            "werewolf": game["role_models"].get("werewolf"),
            "village": game["role_models"].get("villager"),
        }
        for side, model in assignments.items():
            if not model:
                continue
            entry = by_model_role.setdefault(model, {}).setdefault(
                side, {"games": 0, "wins": 0},
            )
            entry["games"] += 1
            won = (game["winner"] == "wolf") == (side == "werewolf")
            entry["wins"] += 1 if won else 0
    for model, sides in by_model_role.items():
        for side, entry in sides.items():
            entry["win_rate"] = (
                entry["wins"] / entry["games"] if entry["games"] else None
            )
    metrics["descriptive_win_rate_by_model_role"] = by_model_role

    # Game-level reliability rates (game-weighted).
    clean_games = sum(1 for g in games if g["clean"])
    fallback_games = sum(
        1 for g in games
        if any(group["ended_in_fallback"]
               for group in g["decision_groups"])
    )
    metrics["clean_game_rate"] = {
        "numerator": clean_games, "denominator": total,
        **(boot(by_seed(lambda g: 1.0 if g["clean"] else 0.0),
                mean_statistic, "clean_game_rate") or {}),
    }
    metrics["fallback_game_rate"] = {
        "numerator": fallback_games, "denominator": total,
        **(boot(by_seed(lambda g: 1.0 if any(
            grp["ended_in_fallback"] for grp in g["decision_groups"]
        ) else 0.0), mean_statistic, "fallback_game_rate") or {}),
    }

    # Decision-group rates: pooled numerators/denominators.
    def group_rate(metric_id, numerator_fn, denominator_fn):
        pairs_by_seed = defaultdict(list)
        numerator = denominator = 0
        for game in games:
            n = sum(1 for grp in game["decision_groups"]
                    if denominator_fn(grp) and numerator_fn(grp))
            d = sum(1 for grp in game["decision_groups"]
                    if denominator_fn(grp))
            numerator += n
            denominator += d
            pairs_by_seed[game["seed"]].append((n, d))
        interval = boot(pairs_by_seed, ratio_statistic, metric_id)
        return {
            "numerator": numerator,
            "denominator": denominator,
            **(interval or {"estimate": None}),
        }

    metrics["fallback_decision_group_rate"] = group_rate(
        "fallback_decision_group_rate",
        lambda grp: grp["ended_in_fallback"], lambda grp: True,
    )
    metrics["retry_rate"] = group_rate(
        "retry_rate",
        lambda grp: grp["api_attempts"] > 1, lambda grp: grp["attempted"],
    )
    metrics["repair_rate"] = group_rate(
        "repair_rate",
        lambda grp: grp["repaired"], lambda grp: grp["completed"],
    )

    # Belief metrics: pooled observations with full counts.
    emitted = sum(g["belief"]["snapshot_coverage"]["emitted"] for g in games)
    valid = sum(g["belief"]["snapshot_coverage"]["valid"] for g in games)
    metrics["belief_snapshot_coverage"] = {
        "emitted": emitted, "valid": valid,
        "rate": (valid / emitted) if emitted else None,
    }

    def pooled_obs(metric_id, obs_fn, statistic=mean_statistic):
        obs_by_seed = defaultdict(list)
        for game in games:
            obs_by_seed[game["seed"]].extend(obs_fn(game))
        interval = boot(obs_by_seed, statistic, metric_id)
        counts = _belief_counts(games, lambda g: len(obs_fn(g)))
        return {**(interval or {"estimate": None}), **counts}

    metrics["probability_movement_toward_wolves"] = pooled_obs(
        "probability_movement", lambda g: g["belief"]["movement"],
    )
    def binary_obs(hits: int, misses: int) -> list:
        # One (numerator, denominator) tuple per eligible observation:
        # pooled ratios and observation counts stay exact.
        return [(1, 1)] * hits + [(0, 1)] * misses

    metrics["harmful_revision"] = pooled_obs(
        "harmful_revision",
        lambda g: binary_obs(
            g["belief"]["harmful"],
            g["belief"]["initially_correct"] - g["belief"]["harmful"],
        ),
        ratio_statistic,
    )
    metrics["correct_belief_retention"] = pooled_obs(
        "correct_belief_retention",
        lambda g: binary_obs(
            g["belief"]["initially_correct"] - g["belief"]["harmful"],
            g["belief"]["harmful"],
        ),
        ratio_statistic,
    )
    metrics["vote_belief_alignment"] = pooled_obs(
        "vote_belief_alignment",
        lambda g: binary_obs(
            g["belief"]["aligned"],
            g["belief"]["alignment_n"] - g["belief"]["aligned"],
        ),
        ratio_statistic,
    )
    metrics["vote_belief_alignment"]["eligible_votes"] = sum(
        g["belief"]["alignment_n"] for g in games
    )
    metrics["vote_belief_alignment"]["total_village_votes"] = sum(
        g["belief"]["village_votes"] for g in games
    )
    metrics["wolf_suspicion_awareness_error"] = pooled_obs(
        "wolf_awareness", lambda g: g["belief"]["awareness_errors"],
    )

    # Cross-game calibration: prediction-weighted Brier and ECE.
    for checkpoint in (CHECKPOINT_PRE, CHECKPOINT_POST):
        def brier_obs(game, cp=checkpoint):
            return [
                (p - (1.0 if is_wolf else 0.0)) ** 2
                for snap_cp, p, is_wolf in game["belief"]["predictions"]
                if snap_cp == cp
            ]
        metrics[f"brier_{checkpoint}"] = pooled_obs(
            f"brier_{checkpoint}", brier_obs,
        )
        flat = [
            (p, is_wolf, game["game_id"], game["seed"])
            for game in games
            for snap_cp, p, is_wolf in game["belief"]["predictions"]
            if snap_cp == checkpoint
        ]
        bins, ece = _ece_bins(flat)
        metrics[f"ece_{checkpoint}"] = {
            "estimate": ece,
            "prediction_count": len(flat),
            "game_count": len({f[2] for f in flat}),
            "seed_count": len({f[3] for f in flat}),
            "bins": bins,
        }

    # Cost, tokens, latency.
    games_with_cost = [g for g in games
                       if g["usage"]["cost_usd"] is not None]
    total_cost = (
        sum(g["usage"]["cost_usd"] for g in games_with_cost)
        if games_with_cost else None
    )
    metrics["cost"] = {
        "total_usd": total_cost,
        "cost_per_game_usd": (
            total_cost / len(games_with_cost)
            if games_with_cost and total_cost is not None else None
        ),
        "games_with_cost": len(games_with_cost),
        "games_with_incomplete_cost": sum(
            1 for g in games if not g["usage"]["cost_complete"]
        ),
        "cost_complete": all(
            g["usage"]["cost_complete"] for g in games
        ) and len(games_with_cost) == total,
    }
    tokens: dict = defaultdict(int)
    for game in games:
        for key, value in game["usage"]["tokens"].items():
            tokens[key] += value
    metrics["tokens"] = dict(tokens)

    latencies = [v for g in games for v in g["usage"]["latencies"]]
    attempted = sum(g["usage"]["api_calls"] for g in games)
    metrics["latency"] = {
        "mean_ms": mean_statistic(latencies),
        "calls_with_latency": len(latencies),
        "total_attempted_calls": attempted,
        "coverage_fraction": (
            len(latencies) / attempted if attempted else None
        ),
    }
    return metrics


# --------------------------------------------------------------------------
# Comparisons
# --------------------------------------------------------------------------

def _game_metric_value(game: dict, metric_id: str) -> Optional[float]:
    if metric_id == "village_win_rate":
        return 1.0 if game["winner"] == "village" else 0.0
    if metric_id == "wolf_win_rate":
        return 1.0 if game["winner"] == "wolf" else 0.0
    return None


def _run_comparison(
    comparison: dict, views: dict, bootstrap: dict,
) -> dict:
    result = {**comparison, "estimate": None, "ci_low": None,
              "ci_high": None, "interval_status": None,
              "n_seeds": 0, "excluded_pairs": {}, "status": None}
    view_games = views.get(comparison["analysis_view"])
    if view_games is None:
        result["status"] = "unknown_analysis_view"
        return result
    if comparison["metric_id"] not in COMPARABLE_METRICS:
        result["status"] = "unsupported_metric"
        return result

    def games_by_seed(condition_id) -> dict:
        out = defaultdict(list)
        for game in view_games:
            if game["condition_id"] == condition_id:
                out[game["seed"]].append(game)
        return out

    a_games = games_by_seed(comparison["condition_a"])
    b_games = games_by_seed(comparison["condition_b"])

    if comparison["design"] != "paired":
        result["status"] = "unsupported_design"
        return result

    # Pair by seed; require eligible observations on both sides and
    # matching game-rule and role-map hashes. Incomplete pairs are
    # excluded and counted — NEVER downgraded to independent.
    excluded: dict = defaultdict(int)
    paired: dict = {}
    for seed in sorted(set(a_games) | set(b_games)):
        a, b = a_games.get(seed), b_games.get(seed)
        if not a or not b:
            excluded["missing_condition_observation"] += 1
            continue
        hashes = {
            (g["game_rules_hash"], g["role_map_hash"]) for g in a + b
        }
        if len(hashes) != 1:
            excluded["mismatched_rules_or_role_map"] += 1
            continue
        a_values = [_game_metric_value(g, comparison["metric_id"])
                    for g in a]
        b_values = [_game_metric_value(g, comparison["metric_id"])
                    for g in b]
        paired[seed] = (a_values, b_values)

    result["excluded_pairs"] = dict(excluded)
    if not paired:
        result["status"] = "no_shared_paired_seeds"
        return result

    sign = 1.0 if comparison["direction"] == "a_minus_b" else -1.0

    def statistic(clusters: list) -> Optional[float]:
        a_all = [v for a_values, _ in clusters for v in a_values]
        b_all = [v for _, b_values in clusters for v in b_values]
        a_mean, b_mean = mean_statistic(a_all), mean_statistic(b_all)
        if a_mean is None or b_mean is None:
            return None
        return sign * (a_mean - b_mean)

    interval = cluster_bootstrap(
        {seed: [pair] for seed, pair in paired.items()},
        statistic,
        n_boot=bootstrap["n_boot"],
        alpha=bootstrap["alpha"],
        rng_seed=derive_rng_seed(
            bootstrap["rng_seed"], "comparison",
            comparison["comparison_id"],
        ),
    )
    result["status"] = "ok"
    result.update(interval or {})
    result["note"] = (
        "Descriptive interval; not a significance declaration. No "
        "multiplicity correction is applied in version 1."
    )
    return result


# --------------------------------------------------------------------------
# Entry point (registered as aggregate-1)
# --------------------------------------------------------------------------

def analyze_v1(
    *, manifest, analysis_contract, sources, lifecycle_records,
    replay_state,
) -> dict:
    bootstrap = {**DEFAULT_BOOTSTRAP,
                 **(analysis_contract.get("bootstrap") or {})}

    completed = [s for s in sources if s.record_type == "trial_completed"]
    eligible_games = [extract_game_evidence(s)
                      for s in completed if s.verified]
    ineligible = [
        {
            "trial_id": s.trial_id,
            "attempt_id": s.attempt_id,
            "game_id": s.game_id,
            "condition_id": s.condition_id,
            "seed": s.seed,
            "source_status": s.source_status,
            "reason": s.source_status,
        }
        for s in completed if not s.verified
    ]

    views = {
        VIEW_ALL_COMPLETED: eligible_games,
        VIEW_CLEAN: [g for g in eligible_games if g["clean"]],
        VIEW_NOT_CLEAN: [g for g in eligible_games if not g["clean"]],
    }
    view_metrics = {
        name: {
            "per_condition": {
                condition_id: _view_metrics(
                    [g for g in games if g["condition_id"] == condition_id],
                    bootstrap, f"{name}:{condition_id}",
                )
                for condition_id in sorted(
                    manifest["execution_contract"]["conditions"]
                )
            },
            "overall": _view_metrics(games, bootstrap, f"{name}:overall"),
        }
        for name, games in views.items()
    }

    comparisons = [
        _run_comparison(comparison, views, bootstrap)
        for comparison in manifest.get("comparisons", [])
    ]

    # Operational accounting: every attempt, all failed work, all
    # health probes. Drifted evidence is excluded from authoritative
    # totals and reported as incomplete.
    attempt_counts = defaultdict(int)
    open_attempts = 0
    for trial in replay_state.trials.values():
        for attempt in trial.attempts:
            terminal = attempt["terminal"]
            if terminal is None:
                open_attempts += 1
            else:
                attempt_counts[terminal["record_type"]] += 1

    cost_by_type: dict = defaultdict(float)
    cost_known_by_type: dict = defaultdict(int)
    incomplete_sources = 0
    excluded_from_totals = 0
    for source in sources:
        if not source.verified:
            excluded_from_totals += 1
            continue
        usage = _usage_evidence(source.rows or [])
        if usage["cost_usd"] is not None:
            cost_by_type[source.record_type] += usage["cost_usd"]
            cost_known_by_type[source.record_type] += 1
        if not usage["cost_complete"]:
            incomplete_sources += 1

    health_records = [r for r in lifecycle_records
                      if r.get("record_type") == "health_check"]
    health_cost = 0.0
    health_cost_known = 0
    for record in health_records:
        usd = as_mapping(record.get("cost")).get("usd")
        if isinstance(usd, (int, float)) and not isinstance(usd, bool):
            health_cost += usd
            health_cost_known += 1

    operational = {
        "attempts": {
            **dict(attempt_counts),
            "open_or_abandoned": open_attempts,
            "total": sum(attempt_counts.values()) + open_attempts,
        },
        "scheduled_trials": len(
            manifest["execution_contract"]["schedule"]
        ),
        "completed_trials": sum(
            1 for trial in replay_state.trials.values() if trial.completed
        ),
        "cost": {
            "by_record_type_usd": dict(cost_by_type),
            "attempts_with_known_cost": dict(cost_known_by_type),
            "health_checks_usd": (
                health_cost if health_cost_known else None
            ),
            "health_checks": len(health_records),
            "sources_with_incomplete_cost": incomplete_sources,
            "sources_excluded_from_totals": excluded_from_totals,
            "complete": incomplete_sources == 0
            and excluded_from_totals == 0,
        },
    }

    return {
        "aggregate_analysis_version": AGGREGATE_ANALYSIS_VERSION,
        "comparison_method_version": COMPARISON_METHOD_VERSION,
        "metric_weighting": METRIC_WEIGHTING,
        "bootstrap": bootstrap,
        "views": view_metrics,
        "view_membership_note": (
            "Views overlap: clean_eligible and "
            "completed_not_clean_eligible are both subsets of "
            "all_completed; nothing completed is hidden."
        ),
        "games": [
            {k: game[k] for k in (
                "trial_id", "attempt_id", "game_id", "condition_id",
                "seed", "repetition", "winner", "rounds", "recovered",
                "clean", "violations",
            )}
            for game in eligible_games
        ],
        "analytically_ineligible": ineligible,
        "comparisons": comparisons,
        "operational": operational,
    }
