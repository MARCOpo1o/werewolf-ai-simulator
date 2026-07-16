"""Read-only access to persisted multi-game experiment artifacts.

Experiment execution is intentionally CLI-only.  This service exposes only
derived index data, immutable summary revisions, and already-written CSV
exports for the browser.
"""
from __future__ import annotations

from pathlib import Path

from werewolf.experiments.exports import exports_dir_for_revision
from werewolf.experiments.index import load_experiment_index
from werewolf.experiments.manifest import (
    ManifestError,
    load_verified_manifest,
    validate_experiment_id,
)
from werewolf.experiments.summaries import (
    SummaryError,
    load_summary_catalog,
    load_summary_revision,
)
from werewolf.reporting.repository import GameRepository

_EXPORTS = frozenset({
    "trials.csv", "attempts.csv", "metrics.csv", "comparisons.csv",
    "calibration.csv",
})


class ExperimentNotFound(ValueError):
    pass


def _validated_id(experiment_id: str) -> str:
    try:
        return validate_experiment_id(experiment_id)
    except ManifestError as exc:
        raise ExperimentNotFound("Experiment not found") from exc


def list_experiments(root: Path) -> dict:
    """Return the derived index without touching paid execution state."""
    return load_experiment_index(root)


def load_experiment(root: Path, experiment_id: str) -> dict:
    experiment_id = _validated_id(experiment_id)
    try:
        manifest = load_verified_manifest(root, experiment_id)
        catalog = load_summary_catalog(root, experiment_id)
    except (ManifestError, SummaryError) as exc:
        raise ExperimentNotFound("Experiment not found") from exc

    index = load_experiment_index(root)
    entry = next(
        (item for item in index.get("experiments", [])
         if item.get("experiment_id") == experiment_id),
        None,
    )
    return {
        "experiment_id": experiment_id,
        "manifest": manifest,
        "summary_catalog": catalog,
        "index_entry": entry,
        "links": {
            "summary": (
                f"/api/experiments/{experiment_id}/summaries/"
                "{revision}"
            ),
            "exports": (
                f"/api/experiments/{experiment_id}/exports/"
                "{revision}/{name}"
            ),
        },
    }


def load_experiment_summary(root: Path, experiment_id: str,
                            revision: int) -> dict:
    experiment_id = _validated_id(experiment_id)
    if isinstance(revision, bool) or not isinstance(revision, int) \
            or revision < 1:
        raise ExperimentNotFound("Summary not found")
    try:
        return load_summary_revision(root, experiment_id, revision)
    except (ManifestError, SummaryError) as exc:
        raise ExperimentNotFound("Summary not found") from exc


def experiment_game_repository(root: Path, experiment_id: str) -> GameRepository:
    """Return a report repository scoped to one experiment's game logs."""
    experiment_id = _validated_id(experiment_id)
    try:
        load_verified_manifest(root, experiment_id)
    except ManifestError as exc:
        raise ExperimentNotFound("Experiment not found") from exc
    return GameRepository(Path(root) / experiment_id / "games")


def export_path(root: Path, experiment_id: str, revision: int,
                filename: str) -> Path:
    experiment_id = _validated_id(experiment_id)
    if isinstance(revision, bool) or not isinstance(revision, int) \
            or revision < 1 or filename not in _EXPORTS:
        raise ExperimentNotFound("Export not found")
    path = exports_dir_for_revision(root, experiment_id, revision) / filename
    if not path.is_file():
        raise ExperimentNotFound("Export not found")
    return path
