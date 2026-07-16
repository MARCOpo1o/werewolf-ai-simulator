"""Derived experiment index.

`outputs/experiments/index.json` accelerates history requests and is
never canonical: it is rebuilt entirely by scanning manifests and
lifecycle journals, so deleting it loses no information.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from werewolf.experiments import manifest as manifest_store
from werewolf.experiments.journal import (
    JournalIntegrityError,
    read_journal,
    replay,
)
from werewolf.experiments.manifest import (
    ManifestError,
    atomic_write_json,
    load_manifest,
    validate_manifest,
)
from werewolf.experiments.summaries import SummaryError, list_summary_revisions

INDEX_VERSION = 1


def index_path(root) -> Path:
    return Path(root) / "index.json"


def _experiment_entry(root, experiment_id: str) -> dict:
    entry = {
        "experiment_id": experiment_id,
        "status": "ok",
        "problems": [],
    }
    try:
        manifest = load_manifest(root, experiment_id)
        errors = validate_manifest(manifest)
        if errors:
            entry["status"] = "invalid_manifest"
            entry["problems"].extend(errors)
    except ManifestError as exc:
        return {**entry, "status": "invalid_manifest",
                "problems": [str(exc)]}

    execution = manifest.get("execution_contract") or {}
    entry.update({
        "manifest_content_sha256": manifest.get("manifest_content_sha256"),
        "created_at": manifest.get("created_at"),
        "description": manifest.get("description"),
        "conditions": sorted(execution.get("conditions") or {}),
        "seed_count": len(execution.get("seeds") or []),
        "repetitions": execution.get("repetitions"),
        "scheduled_trials": len(execution.get("schedule") or []),
        "comparisons": len(manifest.get("comparisons") or []),
    })

    counts = {"completed": 0, "failed_attempts": 0,
              "interrupted_attempts": 0, "open_attempts": 0,
              "attempts": 0, "health_checks": 0, "sessions": 0}
    last_activity = None
    try:
        snapshot = read_journal(
            manifest_store.journal_path(root, experiment_id),
        )
        state = replay(snapshot.records)
        for trial in state.trials.values():
            if trial.completed:
                counts["completed"] += 1
            for attempt in trial.attempts:
                counts["attempts"] += 1
                terminal = attempt["terminal"]
                if terminal is None:
                    counts["open_attempts"] += 1
                elif terminal["record_type"] == "trial_failed":
                    counts["failed_attempts"] += 1
                elif terminal["record_type"] == "trial_interrupted":
                    counts["interrupted_attempts"] += 1
        counts["health_checks"] = len(state.health_checks)
        counts["sessions"] = len(state.sessions)
        if snapshot.records:
            last_activity = snapshot.records[-1].get("recorded_at")
    except JournalIntegrityError as exc:
        entry["status"] = "journal_integrity_error"
        entry["problems"].append(str(exc))
    entry["progress"] = counts
    entry["last_activity"] = last_activity

    try:
        revisions = list_summary_revisions(root, experiment_id)
        entry["summary_revisions"] = len(revisions)
        entry["current_summary_revision"] = (
            revisions[-1]["revision"] if revisions else None
        )
    except SummaryError as exc:
        entry["status"] = "summary_error"
        entry["problems"].append(str(exc))
        entry["summary_revisions"] = None
        entry["current_summary_revision"] = None
    return entry


def rebuild_experiment_index(root) -> dict:
    """Scan manifests and journals; the result is fully derived."""
    root = Path(root)
    experiments = []
    if root.is_dir():
        for child in sorted(root.iterdir()):
            if child.is_dir() and (child / "manifest.json").is_file():
                experiments.append(_experiment_entry(root, child.name))
    index = {
        "index_version": INDEX_VERSION,
        "rebuilt_at": datetime.now(timezone.utc).isoformat(),
        "experiments": experiments,
    }
    atomic_write_json(index_path(root), index)
    return index


def load_experiment_index(root, *, rebuild: bool = False) -> dict:
    path = index_path(root)
    if not rebuild:
        try:
            with open(path, encoding="utf-8") as f:
                index = json.load(f)
            if index.get("index_version") == INDEX_VERSION:
                return index
        except (FileNotFoundError, ValueError):
            pass
    return rebuild_experiment_index(root)
