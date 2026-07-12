"""Seed-level bootstrap statistics for benchmark reporting.

Player-rounds within a game are correlated, and repetitions of the same
seed share a role assignment - so confidence intervals must be
bootstrapped over game SEEDS, not over individual observations. When two
conditions ran the same seeds, differences use a paired bootstrap.

All functions are deterministic given rng_seed.
"""
from __future__ import annotations

import random
from typing import Optional

DEFAULT_N_BOOT = 2000


def _seed_means(values_by_seed: dict) -> dict:
    """Collapse repetitions: one mean value per seed."""
    return {
        seed: sum(values) / len(values)
        for seed, values in values_by_seed.items() if values
    }


def bootstrap_ci(
    values_by_seed: dict,
    n_boot: int = DEFAULT_N_BOOT,
    alpha: float = 0.05,
    rng_seed: int = 0,
) -> Optional[dict]:
    """Percentile bootstrap CI for the mean, resampling seeds.

    values_by_seed: {seed: [value per repetition]}. Returns
    {"estimate", "ci_low", "ci_high", "n_seeds", "n_boot"} or None when
    there is no data.
    """
    means = _seed_means(values_by_seed)
    if not means:
        return None
    seeds = sorted(means)
    estimate = sum(means[s] for s in seeds) / len(seeds)
    if len(seeds) == 1:
        return {"estimate": estimate, "ci_low": estimate,
                "ci_high": estimate, "n_seeds": 1, "n_boot": 0}

    rng = random.Random(rng_seed)
    stats = []
    for _ in range(n_boot):
        sample = [means[rng.choice(seeds)] for _ in seeds]
        stats.append(sum(sample) / len(sample))
    stats.sort()
    lo_index = int((alpha / 2) * n_boot)
    hi_index = min(n_boot - 1, int((1 - alpha / 2) * n_boot))
    return {
        "estimate": estimate,
        "ci_low": stats[lo_index],
        "ci_high": stats[hi_index],
        "n_seeds": len(seeds),
        "n_boot": n_boot,
    }


def paired_bootstrap_diff(
    a_by_seed: dict,
    b_by_seed: dict,
    n_boot: int = DEFAULT_N_BOOT,
    alpha: float = 0.05,
    rng_seed: int = 0,
) -> Optional[dict]:
    """Paired bootstrap for mean(A) - mean(B) over the seeds both
    conditions share. Pairing removes between-seed variance (same seed =
    same role assignment), giving much tighter comparisons."""
    a_means, b_means = _seed_means(a_by_seed), _seed_means(b_by_seed)
    common = sorted(set(a_means) & set(b_means))
    if not common:
        return None
    diffs = {seed: a_means[seed] - b_means[seed] for seed in common}
    result = bootstrap_ci(
        {seed: [d] for seed, d in diffs.items()},
        n_boot=n_boot, alpha=alpha, rng_seed=rng_seed,
    )
    result["n_common_seeds"] = len(common)
    return result
