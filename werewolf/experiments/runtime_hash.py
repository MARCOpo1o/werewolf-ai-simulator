"""Versioned runtime hashes for the execution and analysis code paths.

`execution_runtime_hash` covers exactly the components whose behavior
can change what a paid trial does (engine, orchestration, providers,
prompts, retry/normalization policy, usage logging, execution-relevant
pinned dependencies). `analysis_runtime_hash` covers the code that
derives reports, metrics, and summaries from already-paid evidence.

Keeping the two hashes separate is what lets analysis-only changes land
without blocking resume of a partially executed paid experiment: resume
compares the manifest's pinned execution hash against the current one,
and summarization records the analysis hash it actually ran with.
"""
from __future__ import annotations

from pathlib import Path

from werewolf.experiments.canonical import jcs_sha256, sha256_file

EXECUTION_RUNTIME_VERSION = 1
ANALYSIS_RUNTIME_VERSION = 1

REPO_ROOT = Path(__file__).resolve().parents[2]

# Component -> repo-relative files or directories (directories include
# every *.py file recursively). Missing entries are allowed so earlier
# commits hash cleanly before later modules exist; adding a module
# changes the hash, which is the point.
_EXECUTION_COMPONENTS: dict = {
    "engine_and_rules": ("werewolf/engine",),
    "agent_orchestration": ("werewolf/agents", "werewolf/roles"),
    "providers_and_usage": ("werewolf/llm", "werewolf/json_safety.py"),
    "experiment_execution": (
        "werewolf/experiments/canonical.py",
        "werewolf/experiments/runtime_hash.py",
        "werewolf/experiments/manifest.py",
        "werewolf/experiments/profiles.py",
        "werewolf/experiments/conditions.py",
        "werewolf/experiments/journal.py",
        "werewolf/experiments/scheduler.py",
        "werewolf/experiments/locks.py",
        "werewolf/experiments/health.py",
        "werewolf/experiments/verifier.py",
        "werewolf/experiments/runner.py",
    ),
}
# Pinned dependency entries that alter provider behavior.
_EXECUTION_DEPENDENCIES = ("xai-sdk", "litellm", "python-dotenv")

_ANALYSIS_COMPONENTS: dict = {
    "reporting": ("werewolf/reporting",),
    "evaluation": ("werewolf/evaluation", "werewolf/json_safety.py"),
    "experiment_analysis": (
        "werewolf/experiments/canonical.py",
        "werewolf/experiments/analysis_registry.py",
        "werewolf/experiments/snapshot.py",
        "werewolf/experiments/aggregate.py",
        "werewolf/experiments/summaries.py",
    ),
}
_ANALYSIS_DEPENDENCIES: tuple = ()


def _component_files(entries, root: Path) -> list:
    files = []
    for entry in entries:
        path = root / entry
        if path.is_dir():
            files.extend(
                p for p in sorted(path.rglob("*.py"))
                if "__pycache__" not in p.parts
            )
        elif path.is_file():
            files.append(path)
    return files


def _dependency_pins(names, root: Path) -> list:
    requirements = root / "requirements.txt"
    pins = []
    if requirements.is_file():
        for line in requirements.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            package = (
                line.split("==")[0].split(">=")[0].split("<=")[0].strip()
            )
            if package in names:
                pins.append({"package": package, "pin": line})
    return sorted(pins, key=lambda p: p["package"])


def _runtime_components(components: dict, root: Path) -> list:
    hashed = []
    for name in sorted(components):
        files = _component_files(components[name], root)
        hashed.append({
            "component": name,
            "sha256": jcs_sha256([
                {
                    "path": file.relative_to(root).as_posix(),
                    "sha256": sha256_file(file),
                }
                for file in files
            ]),
            "files": len(files),
        })
    return hashed


def execution_runtime_components(root: Path = REPO_ROOT) -> dict:
    return {
        "runtime_hash_version": EXECUTION_RUNTIME_VERSION,
        "components": _runtime_components(_EXECUTION_COMPONENTS, root),
        "pinned_dependencies": _dependency_pins(_EXECUTION_DEPENDENCIES, root),
    }


def execution_runtime_hash(root: Path = REPO_ROOT) -> str:
    return jcs_sha256(execution_runtime_components(root))


def analysis_runtime_components(root: Path = REPO_ROOT) -> dict:
    return {
        "runtime_hash_version": ANALYSIS_RUNTIME_VERSION,
        "components": _runtime_components(_ANALYSIS_COMPONENTS, root),
        "pinned_dependencies": _dependency_pins(_ANALYSIS_DEPENDENCIES, root),
    }


def analysis_runtime_hash(root: Path = REPO_ROOT) -> str:
    return jcs_sha256(analysis_runtime_components(root))


def repository_commit(root: Path = REPO_ROOT) -> dict:
    """Informational only: never part of any contract hash."""
    import subprocess
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root,
            capture_output=True, text=True, timeout=5,
        ).stdout.strip() or None
        dirty_probe = subprocess.run(
            ["git", "status", "--porcelain"], cwd=root,
            capture_output=True, text=True, timeout=5,
        )
        dirty = (
            bool(dirty_probe.stdout.strip())
            if dirty_probe.returncode == 0 else None
        )
    except Exception:
        commit, dirty = None, None
    return {"repository_commit": commit, "working_tree_dirty": dirty}
