"""Small, stable runtime fingerprint for reproducible game configs."""
from __future__ import annotations

import hashlib
import platform
from functools import lru_cache
from importlib import metadata
from pathlib import Path


def _version(distribution: str):
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


@lru_cache(maxsize=1)
def collect_runtime_metadata() -> dict:
    root = Path(__file__).resolve().parents[2]
    lock_path = root / "requirements.txt"
    lock_hash = None
    if lock_path.exists():
        lock_hash = hashlib.sha256(lock_path.read_bytes()).hexdigest()
    return {
        "python": platform.python_version(),
        "xai_sdk": _version("xai-sdk"),
        "litellm": _version("litellm"),
        "flask": _version("flask"),
        "requirements_sha256": lock_hash,
    }

