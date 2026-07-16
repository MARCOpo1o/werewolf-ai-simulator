"""Aggregate multi-game analysis (versioned).

Version constants live here so manifests can pin them; the metric,
view, and comparison implementations arrive with the summarizer.
"""
from __future__ import annotations

AGGREGATE_ANALYSIS_VERSION = 1
COMPARISON_METHOD_VERSION = 1

DEFAULT_BOOTSTRAP = {"n_boot": 2000, "alpha": 0.05, "rng_seed": 0}
METRIC_WEIGHTING = "aggregate-1"
