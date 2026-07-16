"""Immutable aggregate summary revisions.

A summary revision is an immutable artifact identified by
`summary_input_sha256`: the JCS SHA-256 of the analysis contract, the
analysis runtime hash, the lifecycle snapshot hash, and the sorted game
source identities. Summarization is a no-op only when that exact input
hash already exists — additional completed trials, source drift, or an
analysis change each produce a NEW revision; nothing is ever rewritten.

`summary.json` is only a revision catalog / current pointer. Deleting
it loses nothing: it is rebuilt by scanning summaries/summary_*.json.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from werewolf.experiments import manifest as manifest_store
from werewolf.experiments.analysis_registry import (
    AnalysisPolicyUnavailable,
    current_analysis_contract,
    registry_key,
    resolve_analysis_implementation,
)
from werewolf.experiments.canonical import jcs_sha256
from werewolf.experiments.journal import replay
from werewolf.experiments.locks import analysis_lock
from werewolf.experiments.manifest import (
    atomic_write_json,
    load_verified_manifest,
)
from werewolf.experiments.runtime_hash import analysis_runtime_hash
from werewolf.experiments.snapshot import (
    capture_game_sources,
    capture_lifecycle_snapshot,
)

SUMMARY_BUILD_VERSION = 1

_SUMMARY_FILE_RE = re.compile(r"^summary_(\d{4,})\.json$")


class SummaryError(RuntimeError):
    pass


def summary_revision_path(root, experiment_id: str, revision: int) -> Path:
    return (manifest_store.summaries_dir(root, experiment_id)
            / f"summary_{revision:04d}.json")


def compute_summary_input_sha256(
    *, analysis_contract: dict, analysis_runtime: str,
    lifecycle_snapshot_sha256: str, sources: list,
) -> str:
    return jcs_sha256({
        "analysis_contract_sha256": jcs_sha256(analysis_contract),
        "analysis_runtime_hash": analysis_runtime,
        "lifecycle_snapshot_sha256": lifecycle_snapshot_sha256,
        "sorted_game_sources": [source.identity() for source in sources],
    })


def _revision_header(payload: dict) -> dict:
    return {
        "revision": payload["revision"],
        "summary_build_version": payload["summary_build_version"],
        "summary_input_sha256": payload["summary_input_sha256"],
        "analysis_policy": payload["analysis_policy"],
        "analysis_contract_sha256": payload["analysis_contract_sha256"],
        "analysis_runtime_hash": payload["analysis_runtime_hash"],
        "created_at": payload["created_at"],
        "lifecycle": payload["lifecycle"],
    }


def list_summary_revisions(root, experiment_id: str) -> list:
    """Scan summaries/ directly; the catalog is derived from this."""
    directory = manifest_store.summaries_dir(root, experiment_id)
    revisions = []
    if directory.is_dir():
        for path in sorted(directory.iterdir()):
            match = _SUMMARY_FILE_RE.match(path.name)
            if not match:
                continue
            try:
                with open(path, encoding="utf-8") as f:
                    payload = json.load(f)
            except (OSError, ValueError) as exc:
                raise SummaryError(
                    f"Summary revision {path} is unreadable: {exc}"
                )
            revisions.append(_revision_header(payload))
    revisions.sort(key=lambda r: r["revision"])
    return revisions


def rebuild_summary_catalog(root, experiment_id: str) -> dict:
    revisions = list_summary_revisions(root, experiment_id)
    catalog = {
        "experiment_id": experiment_id,
        "catalog_rebuilt_at": datetime.now(timezone.utc).isoformat(),
        "current_revision": (
            revisions[-1]["revision"] if revisions else None
        ),
        "revisions": revisions,
    }
    atomic_write_json(
        manifest_store.summary_catalog_path(root, experiment_id), catalog,
    )
    return catalog


def load_summary_catalog(root, experiment_id: str) -> dict:
    path = manifest_store.summary_catalog_path(root, experiment_id)
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, ValueError):
        # The catalog is derived and rebuildable at any time.
        return rebuild_summary_catalog(root, experiment_id)


def load_summary_revision(root, experiment_id: str, revision: int) -> dict:
    path = summary_revision_path(root, experiment_id, revision)
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        raise SummaryError(
            f"Summary revision {revision} of {experiment_id} does not exist"
        )


def summarize_experiment(
    root,
    experiment_id: str,
    *,
    analysis_policy: str = "pinned",
    analyze: Optional[Callable] = None,
    exporter: Optional[Callable] = None,
) -> dict:
    """Build (or no-op to) an immutable summary revision.

    The analysis lock covers snapshotting, source reading, revision
    allocation, the immutable write, and the catalog update, so two
    concurrent summarizers can never interleave.
    """
    if analysis_policy not in ("pinned", "current"):
        raise SummaryError(
            f"Unknown analysis policy: {analysis_policy!r}"
        )
    manifest = load_verified_manifest(root, experiment_id)
    experiment_dir = manifest_store.experiment_dir(root, experiment_id)

    with analysis_lock(experiment_dir, experiment_id):
        # 1-2. Lifecycle snapshot, then single-read game sources.
        lifecycle = capture_lifecycle_snapshot(
            manifest_store.journal_path(root, experiment_id),
        )
        sources = capture_game_sources(
            manifest_store.games_dir(root, experiment_id), lifecycle,
        )

        current_runtime = analysis_runtime_hash()
        if analysis_policy == "pinned":
            analysis_contract = manifest["analysis_contract"]
            pinned_runtime = analysis_contract.get("analysis_runtime_hash")
            if pinned_runtime != current_runtime:
                raise AnalysisPolicyUnavailable(
                    "analysis_policy_unavailable: pinned analysis runtime "
                    "does not match this checkout; run from the recorded "
                    "analysis implementation or use --analysis-policy current."
                )
        else:
            analysis_contract = current_analysis_contract(
                bootstrap=manifest["analysis_contract"].get("bootstrap"),
            )
            # The current-policy revision must self-describe the exact
            # implementation selected now, even when a caller injects a
            # deterministic runtime hash in tests.
            analysis_contract = {
                **analysis_contract,
                "analysis_runtime_hash": current_runtime,
            }
        if analyze is None:
            analyze = resolve_analysis_implementation(analysis_contract)

        input_sha = compute_summary_input_sha256(
            analysis_contract=analysis_contract,
            analysis_runtime=current_runtime,
            lifecycle_snapshot_sha256=(
                lifecycle.lifecycle_snapshot_sha256
            ),
            sources=sources,
        )

        # No-op only on an exact input-hash match.
        existing = list_summary_revisions(root, experiment_id)
        for revision in existing:
            if revision["summary_input_sha256"] == input_sha:
                # Exports are derived, not part of immutable summary
                # identity. Rebuild them on a no-op summary so a deleted or
                # previously interrupted export set is recoverable.
                if exporter is not None:
                    exporter(
                        root, experiment_id,
                        load_summary_revision(
                            root, experiment_id, revision["revision"],
                        ),
                    )
                catalog = load_summary_catalog(root, experiment_id)
                return {
                    "revision": revision["revision"],
                    "created": False,
                    "summary_input_sha256": input_sha,
                    "catalog": catalog,
                }

        analysis = analyze(
            manifest=manifest,
            analysis_contract=analysis_contract,
            sources=sources,
            lifecycle_records=lifecycle.records,
            replay_state=replay(lifecycle.records),
        )

        # 3. Allocate the next revision and write immutably ('x' mode:
        # an existing file is a hard error, revisions are never
        # rewritten).
        revision_number = (
            existing[-1]["revision"] + 1 if existing else 1
        )
        payload = {
            "experiment_id": experiment_id,
            "manifest_content_sha256": manifest["manifest_content_sha256"],
            "revision": revision_number,
            "summary_build_version": SUMMARY_BUILD_VERSION,
            "summary_input_sha256": input_sha,
            "analysis_policy": analysis_policy,
            "analysis_registry_key": list(registry_key(analysis_contract)),
            "analysis_contract": analysis_contract,
            "analysis_contract_sha256": jcs_sha256(analysis_contract),
            "analysis_runtime_hash": current_runtime,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "lifecycle": lifecycle.meta(),
            "sources": [source.identity() for source in sources],
            "analysis": analysis,
        }
        path = summary_revision_path(root, experiment_id, revision_number)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():  # Defensive: the analysis lock should prevent it.
            raise SummaryError(
                f"Refusing to replace immutable summary revision {path}"
            )
        # atomic_write_json flushes/fsyncs a same-directory temporary file
        # before publication. The revision remains immutable after publish.
        atomic_write_json(path, payload)

        if exporter is not None:
            exporter(root, experiment_id, payload)

        # 4. Atomically refresh the derived catalog.
        catalog = rebuild_summary_catalog(root, experiment_id)
        return {
            "revision": revision_number,
            "created": True,
            "summary_input_sha256": input_sha,
            "path": str(path),
            "catalog": catalog,
        }
