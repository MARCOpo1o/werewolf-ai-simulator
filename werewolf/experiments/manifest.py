"""Canonical experiment manifests: identity, contracts, and storage.

A manifest is the single canonical configuration record of an
experiment. Its identity is `manifest_content_sha256`, an RFC 8785 JCS
SHA-256 over the manifest with the self-hash field omitted. Every
lifecycle record carries this hash, and the manifest becomes immutable
the moment the first lifecycle record exists: changing anything after
that point requires a new experiment ID.

Two nested contracts are hashed independently:

- `execution_contract` holds only behavior that can affect paid
  execution. Its hash (with the runtime hash inside) gates resume.
- `analysis_contract` holds derivation policy versions. Changing it
  never blocks resume; it only produces new summary revisions.

`repository_commit` / `working_tree_dirty` / `analysis_code_commit` are
recorded as informational metadata outside both contract hashes.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path

from werewolf.experiments.canonical import (
    CanonicalizationError,
    jcs_sha256,
)

MANIFEST_SCHEMA_VERSION = 1
DEFAULT_EXPERIMENTS_ROOT = Path("outputs/experiments")

_EXPERIMENT_ID_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_\-]{0,99}$")

_EXECUTION_CONTRACT_KEYS = frozenset({
    "conditions", "game", "seeds", "repetitions", "prompt_profile",
    "models", "generation", "predeclared_adjustments", "policies",
    "scheduler", "schedule", "execution_runtime_hash",
})
_ANALYSIS_CONTRACT_KEYS = frozenset({
    "report_build_version", "validity_policy_version",
    "belief_metrics_version", "aggregate_analysis_version",
    "comparison_method_version", "bootstrap", "metric_weighting",
    "analysis_runtime_hash",
})
_TOP_LEVEL_KEYS = frozenset({
    "manifest_schema_version", "experiment_id", "created_at",
    "description", "execution_contract", "analysis_contract",
    "comparisons", "metadata", "manifest_content_sha256",
})


class ManifestError(ValueError):
    pass


class ManifestImmutableError(ManifestError):
    """The manifest already governs paid lifecycle records."""


def validate_experiment_id(experiment_id) -> str:
    if not isinstance(experiment_id, str) or not _EXPERIMENT_ID_RE.match(
        experiment_id
    ):
        raise ManifestError(
            "experiment_id must be 1-100 chars of [a-zA-Z0-9_-] "
            "starting with an alphanumeric"
        )
    return experiment_id


def compute_manifest_hash(manifest: dict) -> str:
    """JCS SHA-256 with the self-hash field omitted."""
    content = {
        k: v for k, v in manifest.items() if k != "manifest_content_sha256"
    }
    return jcs_sha256(content)


def finalize_manifest(manifest: dict) -> dict:
    errors = validate_manifest(manifest, require_hash=False)
    if errors:
        raise ManifestError("Invalid manifest: " + "; ".join(errors))
    finalized = dict(manifest)
    finalized["manifest_content_sha256"] = compute_manifest_hash(manifest)
    return finalized


def execution_contract_sha256(manifest: dict) -> str:
    return jcs_sha256(manifest["execution_contract"])


def analysis_contract_sha256(manifest: dict) -> str:
    return jcs_sha256(manifest["analysis_contract"])


def validate_manifest(manifest, *, require_hash: bool = True) -> list:
    """Structural validation. Returns a list of error strings."""
    errors = []
    if not isinstance(manifest, dict):
        return ["manifest must be a JSON object"]

    unknown = set(manifest) - _TOP_LEVEL_KEYS
    if unknown:
        errors.append(f"unknown top-level keys: {sorted(unknown)}")
    if manifest.get("manifest_schema_version") != MANIFEST_SCHEMA_VERSION:
        errors.append(
            f"manifest_schema_version must be {MANIFEST_SCHEMA_VERSION}"
        )
    try:
        validate_experiment_id(manifest.get("experiment_id"))
    except ManifestError as exc:
        errors.append(str(exc))
    if not isinstance(manifest.get("created_at"), str):
        errors.append("created_at must be an ISO-8601 string")

    execution = manifest.get("execution_contract")
    if not isinstance(execution, dict):
        errors.append("execution_contract must be an object")
    else:
        missing = _EXECUTION_CONTRACT_KEYS - set(execution)
        if missing:
            errors.append(f"execution_contract missing keys: {sorted(missing)}")
        unknown = set(execution) - _EXECUTION_CONTRACT_KEYS
        if unknown:
            errors.append(f"execution_contract unknown keys: {sorted(unknown)}")
        seeds = execution.get("seeds")
        if not (
            isinstance(seeds, list) and seeds
            and all(isinstance(s, int) and not isinstance(s, bool) for s in seeds)
        ):
            errors.append("execution_contract.seeds must be a non-empty "
                          "list of integers")
        elif len(set(seeds)) != len(seeds):
            errors.append("execution_contract.seeds must be unique")
        repetitions = execution.get("repetitions")
        if not isinstance(repetitions, int) or isinstance(repetitions, bool) \
                or repetitions < 1:
            errors.append("execution_contract.repetitions must be a "
                          "positive integer")
        conditions = execution.get("conditions")
        if not (isinstance(conditions, dict) and conditions):
            errors.append("execution_contract.conditions must be a "
                          "non-empty object")
        if not isinstance(execution.get("execution_runtime_hash"), str):
            errors.append("execution_contract.execution_runtime_hash "
                          "must be a string")

    analysis = manifest.get("analysis_contract")
    if not isinstance(analysis, dict):
        errors.append("analysis_contract must be an object")
    else:
        missing = _ANALYSIS_CONTRACT_KEYS - set(analysis)
        if missing:
            errors.append(f"analysis_contract missing keys: {sorted(missing)}")
        unknown = set(analysis) - _ANALYSIS_CONTRACT_KEYS
        if unknown:
            errors.append(f"analysis_contract unknown keys: {sorted(unknown)}")

    if not isinstance(manifest.get("comparisons"), list):
        errors.append("comparisons must be a list")
    if not isinstance(manifest.get("metadata"), dict):
        errors.append("metadata must be an object")

    if require_hash:
        recorded = manifest.get("manifest_content_sha256")
        if not isinstance(recorded, str):
            errors.append("manifest_content_sha256 missing")
        elif not errors:
            try:
                expected = compute_manifest_hash(manifest)
            except CanonicalizationError as exc:
                errors.append(f"manifest is not canonicalizable: {exc}")
            else:
                if recorded != expected:
                    errors.append(
                        "manifest_content_sha256 does not match content "
                        f"(recorded {recorded[:12]}…, computed {expected[:12]}…)"
                    )
    else:
        try:
            jcs_sha256(
                {k: v for k, v in manifest.items()
                 if k != "manifest_content_sha256"}
            )
        except CanonicalizationError as exc:
            errors.append(f"manifest is not canonicalizable: {exc}")
    return errors


# --------------------------------------------------------------------------
# Storage layout
# --------------------------------------------------------------------------

def experiment_dir(root, experiment_id: str) -> Path:
    return Path(root) / validate_experiment_id(experiment_id)


def manifest_path(root, experiment_id: str) -> Path:
    return experiment_dir(root, experiment_id) / "manifest.json"


def journal_path(root, experiment_id: str) -> Path:
    return experiment_dir(root, experiment_id) / "trials.jsonl"


def games_dir(root, experiment_id: str) -> Path:
    return experiment_dir(root, experiment_id) / "games"


def summaries_dir(root, experiment_id: str) -> Path:
    return experiment_dir(root, experiment_id) / "summaries"


def exports_dir(root, experiment_id: str) -> Path:
    return experiment_dir(root, experiment_id) / "exports"


def summary_catalog_path(root, experiment_id: str) -> Path:
    return experiment_dir(root, experiment_id) / "summary.json"


def manifest_is_frozen(root, experiment_id: str) -> bool:
    """True once any lifecycle record exists; the manifest is then
    immutable and any change requires a new experiment ID."""
    journal = journal_path(root, experiment_id)
    try:
        return journal.stat().st_size > 0
    except OSError:
        return False


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=path.name, suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def write_manifest(root, manifest: dict) -> Path:
    """Persist a finalized manifest. Rewriting is allowed only while no
    lifecycle record exists AND only with identical content."""
    errors = validate_manifest(manifest)
    if errors:
        raise ManifestError("Invalid manifest: " + "; ".join(errors))
    experiment_id = manifest["experiment_id"]
    path = manifest_path(root, experiment_id)
    if path.exists():
        existing = load_manifest(root, experiment_id)
        if (existing.get("manifest_content_sha256")
                != manifest["manifest_content_sha256"]):
            if manifest_is_frozen(root, experiment_id):
                raise ManifestImmutableError(
                    f"Experiment {experiment_id} already has lifecycle "
                    "records; create a new experiment ID instead of "
                    "modifying its manifest."
                )
            raise ManifestError(
                f"Experiment {experiment_id} already exists with different "
                "content; choose a new experiment ID."
            )
        return path
    atomic_write_json(path, manifest)
    return path


def load_manifest(root, experiment_id: str) -> dict:
    path = manifest_path(root, experiment_id)
    try:
        with open(path, encoding="utf-8") as f:
            manifest = json.load(f)
    except FileNotFoundError:
        raise ManifestError(f"No manifest found for {experiment_id}")
    except json.JSONDecodeError as exc:
        raise ManifestError(f"Manifest for {experiment_id} is corrupt: {exc}")
    return manifest


def load_verified_manifest(root, experiment_id: str) -> dict:
    manifest = load_manifest(root, experiment_id)
    errors = validate_manifest(manifest)
    if errors:
        raise ManifestError(
            f"Manifest for {experiment_id} failed validation: "
            + "; ".join(errors)
        )
    return manifest
