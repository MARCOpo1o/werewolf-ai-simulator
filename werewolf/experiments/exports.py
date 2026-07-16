"""CSV exports derived from immutable summary revisions.

Exports are rebuildable derivatives: deleting them loses nothing.
Every row carries full provenance (experiment, manifest hash, summary
revision and input hash, analysis contract and runtime hashes, view,
condition, and the seed/game/observation counts behind the row), so a
spreadsheet row can always be traced back to its exact evidence.
"""
from __future__ import annotations

import csv
from pathlib import Path

from werewolf.experiments import manifest as manifest_store

PROVENANCE_FIELDS = [
    "experiment_id",
    "manifest_content_sha256",
    "summary_revision",
    "summary_build_version",
    "summary_input_sha256",
    "analysis_contract_sha256",
    "analysis_runtime_hash",
    "analysis_view",
    "condition_id",
    "seed_count",
    "game_count",
    "observation_count",
]

# Metrics exported in long form from each view/condition scope.
_SCALAR_METRICS = (
    "village_win_rate", "wolf_win_rate", "clean_game_rate",
    "clean_eligible_completion_rate",
    "fallback_game_rate", "fallback_decision_group_rate", "retry_rate",
    "repair_rate", "parse_failure_rate", "invalid_action_rate",
    "probability_movement_toward_wolves", "initial_correctness",
    "harmful_revision", "beneficial_revision", "correct_belief_retention",
    "vote_belief_alignment", "wolf_suspicion_awareness_error",
    "brier_pre_discussion", "brier_post_discussion",
)


def exports_dir_for_revision(root, experiment_id: str, revision: int) -> Path:
    return (manifest_store.exports_dir(root, experiment_id)
            / f"summary_{revision:04d}")


def _provenance(summary: dict, *, view: str = "", condition: str = "",
                seed_count="", game_count="", observation_count="") -> dict:
    return {
        "experiment_id": summary["experiment_id"],
        "manifest_content_sha256": summary["manifest_content_sha256"],
        "summary_revision": summary["revision"],
        "summary_build_version": summary["summary_build_version"],
        "summary_input_sha256": summary["summary_input_sha256"],
        "analysis_contract_sha256": summary["analysis_contract_sha256"],
        "analysis_runtime_hash": summary["analysis_runtime_hash"],
        "analysis_view": view,
        "condition_id": condition,
        "seed_count": seed_count,
        "game_count": game_count,
        "observation_count": observation_count,
    }


def _write_csv(path: Path, fieldnames: list, rows: list) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _scopes(analysis: dict):
    for view_name, view in analysis["views"].items():
        yield view_name, "overall", view["overall"]
        for condition_id, metrics in view["per_condition"].items():
            yield view_name, condition_id, metrics


def write_summary_exports(root, experiment_id: str, summary: dict) -> Path:
    analysis = summary["analysis"]
    directory = exports_dir_for_revision(
        root, experiment_id, summary["revision"],
    )
    directory.mkdir(parents=True, exist_ok=True)

    # trials.csv: one row per analyzed game plus ineligible sources.
    trial_fields = PROVENANCE_FIELDS + [
        "trial_id", "attempt_id", "game_id", "seed", "repetition",
        "winner", "rounds", "recovered", "clean", "violations",
        "source_status",
    ]
    trial_rows = []
    for game in analysis["games"]:
        trial_rows.append({
            **_provenance(
                summary, view="all_completed",
                condition=game["condition_id"],
                seed_count=1, game_count=1, observation_count=1,
            ),
            "trial_id": game["trial_id"],
            "attempt_id": game["attempt_id"],
            "game_id": game["game_id"],
            "seed": game["seed"],
            "repetition": game["repetition"],
            "winner": game["winner"],
            "rounds": game["rounds"],
            "recovered": game["recovered"],
            "clean": game["clean"],
            "violations": ";".join(sorted(game["violations"])) or "",
            "source_status": "verified",
        })
    for source in analysis["analytically_ineligible"]:
        trial_rows.append({
            **_provenance(
                summary, view="", condition=source["condition_id"],
                seed_count=1, game_count=1, observation_count=0,
            ),
            "trial_id": source["trial_id"],
            "attempt_id": source["attempt_id"],
            "game_id": source["game_id"],
            "seed": source["seed"],
            "repetition": "",
            "winner": "",
            "rounds": "",
            "recovered": "",
            "clean": "",
            "violations": "",
            "source_status": source["source_status"],
        })
    _write_csv(directory / "trials.csv", trial_fields, trial_rows)

    # metrics.csv: long form, one row per scope x metric.
    metric_fields = PROVENANCE_FIELDS + [
        "metric_id", "estimate", "ci_low", "ci_high", "interval_status",
        "numerator", "denominator", "n_seeds", "n_boot",
    ]
    metric_rows = []
    for view_name, condition_id, metrics in _scopes(analysis):
        for metric_id in _SCALAR_METRICS:
            # A view with no eligible games still needs an explicit row for
            # every headline metric. Omitting it makes a zero-coverage view
            # indistinguishable from a view that was never exported.
            metric = metrics.get(metric_id)
            if not isinstance(metric, dict):
                metric = {}
            metric_rows.append({
                **_provenance(
                    summary, view=view_name, condition=condition_id,
                    seed_count=metrics.get("seed_count", ""),
                    game_count=metrics.get("games", ""),
                    observation_count=metric.get(
                        "observations", metric.get("denominator", ""),
                    ),
                ),
                "metric_id": metric_id,
                "estimate": metric.get("estimate"),
                "ci_low": metric.get("ci_low"),
                "ci_high": metric.get("ci_high"),
                "interval_status": metric.get("interval_status"),
                "numerator": metric.get("numerator"),
                "denominator": metric.get("denominator"),
                "n_seeds": metric.get("n_seeds"),
                "n_boot": metric.get("n_boot"),
            })
    _write_csv(directory / "metrics.csv", metric_fields, metric_rows)

    # comparisons.csv
    comparison_fields = PROVENANCE_FIELDS + [
        "comparison_id", "condition_a", "condition_b", "metric_id",
        "design", "effect", "direction", "status", "estimate",
        "ci_low", "ci_high", "interval_status", "n_seeds",
        "excluded_pairs",
    ]
    comparison_rows = []
    for comparison in analysis["comparisons"]:
        comparison_rows.append({
            **_provenance(
                summary, view=comparison["analysis_view"],
                condition="",
                seed_count=comparison.get("n_seeds", ""),
                game_count="", observation_count="",
            ),
            "comparison_id": comparison["comparison_id"],
            "condition_a": comparison["condition_a"],
            "condition_b": comparison["condition_b"],
            "metric_id": comparison["metric_id"],
            "design": comparison["design"],
            "effect": comparison["effect"],
            "direction": comparison["direction"],
            "status": comparison.get("status"),
            "estimate": comparison.get("estimate"),
            "ci_low": comparison.get("ci_low"),
            "ci_high": comparison.get("ci_high"),
            "interval_status": comparison.get("interval_status"),
            "n_seeds": comparison.get("n_seeds"),
            "excluded_pairs": ";".join(
                f"{reason}={count}" for reason, count in sorted(
                    (comparison.get("excluded_pairs") or {}).items()
                )
            ),
        })
    _write_csv(directory / "comparisons.csv", comparison_fields,
               comparison_rows)

    # calibration.csv: ECE bins per scope and checkpoint.
    calibration_fields = PROVENANCE_FIELDS + [
        "checkpoint", "bin", "prediction_count", "bin_game_count",
        "bin_seed_count", "mean_confidence", "empirical_frequency",
        "absolute_gap", "ece",
    ]
    calibration_rows = []
    for view_name, condition_id, metrics in _scopes(analysis):
        for checkpoint in ("pre_discussion", "post_discussion"):
            ece = metrics.get(f"ece_{checkpoint}")
            if not isinstance(ece, dict):
                continue
            for entry in ece["bins"]:
                calibration_rows.append({
                    **_provenance(
                        summary, view=view_name, condition=condition_id,
                        seed_count=ece.get("seed_count", ""),
                        game_count=ece.get("game_count", ""),
                        observation_count=ece.get("prediction_count", ""),
                    ),
                    "checkpoint": checkpoint,
                    "bin": entry["bin"],
                    "prediction_count": entry["prediction_count"],
                    "bin_game_count": entry["game_count"],
                    "bin_seed_count": entry["seed_count"],
                    "mean_confidence": entry["mean_confidence"],
                    "empirical_frequency": entry["empirical_frequency"],
                    "absolute_gap": entry["absolute_gap"],
                    "ece": ece["estimate"],
                })
    _write_csv(directory / "calibration.csv", calibration_fields,
               calibration_rows)
    return directory
