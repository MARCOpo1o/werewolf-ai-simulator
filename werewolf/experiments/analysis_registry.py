"""Registry of supported historical analysis implementations.

A pinned summarization must run the EXACT implementation registered for
the manifest's analysis-policy versions. When that combination is no
longer shipped, summarization fails with `analysis_policy_unavailable`
instead of silently substituting current logic: a summary that claims
policy X must have been produced by policy X.

`--analysis-policy current` is the intentional escape hatch: it selects
the current versions and produces a new immutable summary revision.
"""
from __future__ import annotations

from importlib import import_module

from werewolf.experiments.aggregate import (
    AGGREGATE_ANALYSIS_VERSION,
    DEFAULT_BOOTSTRAP,
    METRIC_WEIGHTING,
)


class AnalysisPolicyUnavailable(RuntimeError):
    error_code = "analysis_policy_unavailable"


# (report, validity, belief, aggregate) -> dotted implementation path.
# Implementations are resolved lazily so importing the registry never
# drags in the whole analysis stack.
ANALYSIS_IMPLEMENTATIONS = {
    ("report-12", "validity-4", "belief-2", "aggregate-1"):
        "werewolf.experiments.aggregate:analyze_v1",
}


def registry_key(analysis_contract: dict) -> tuple:
    return (
        f"report-{analysis_contract.get('report_build_version')}",
        f"validity-{analysis_contract.get('validity_policy_version')}",
        f"belief-{analysis_contract.get('belief_metrics_version')}",
        f"aggregate-{analysis_contract.get('aggregate_analysis_version')}",
    )


def resolve_analysis_implementation(analysis_contract: dict):
    """Return the registered analyze callable for a pinned contract, or
    raise AnalysisPolicyUnavailable."""
    key = registry_key(analysis_contract)
    dotted = ANALYSIS_IMPLEMENTATIONS.get(key)
    if dotted is None:
        raise AnalysisPolicyUnavailable(
            f"analysis_policy_unavailable: no registered implementation "
            f"for {key}; supported: "
            f"{sorted(ANALYSIS_IMPLEMENTATIONS)}. Use --analysis-policy "
            "current to summarize with current logic as a new revision."
        )
    module_name, _, attribute = dotted.partition(":")
    try:
        implementation = getattr(import_module(module_name), attribute)
    except (ImportError, AttributeError) as exc:
        raise AnalysisPolicyUnavailable(
            f"analysis_policy_unavailable: registered implementation "
            f"{dotted} for {key} could not be loaded: {exc}"
        )
    return implementation


def current_analysis_contract(bootstrap: dict = None) -> dict:
    """The analysis contract `--analysis-policy current` selects."""
    from werewolf.evaluation.belief_metrics import METRICS_VERSION
    from werewolf.evaluation.validity import VALIDITY_POLICY_VERSION
    from werewolf.reporting.builder import REPORT_BUILD_VERSION
    from werewolf.experiments.aggregate import COMPARISON_METHOD_VERSION
    from werewolf.experiments.runtime_hash import analysis_runtime_hash

    return {
        "report_build_version": REPORT_BUILD_VERSION,
        "validity_policy_version": VALIDITY_POLICY_VERSION,
        "belief_metrics_version": METRICS_VERSION,
        "aggregate_analysis_version": AGGREGATE_ANALYSIS_VERSION,
        "comparison_method_version": COMPARISON_METHOD_VERSION,
        "bootstrap": {**DEFAULT_BOOTSTRAP, **(bootstrap or {})},
        "metric_weighting": METRIC_WEIGHTING,
        "analysis_runtime_hash": analysis_runtime_hash(),
    }
