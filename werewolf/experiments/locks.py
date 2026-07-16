"""Nonblocking OS-level advisory locks for execution and analysis.

flock(2) ownership is authoritative: it cannot leak across crashes
(the OS releases it when the process dies) and it conflicts across
processes on the same host. The JSON metadata written into the lock
file (pid, hostname, experiment, acquisition time) is diagnostics
only — it may be stale after a crash and is never used to decide
whether the lock is held.

Multi-process safety is deliberately limited to local advisory locks;
distributed execution is out of scope.
"""
from __future__ import annotations

import fcntl
import json
import os
import socket
from datetime import datetime, timezone
from pathlib import Path


class LockError(RuntimeError):
    pass


class LockHeldError(LockError):
    """Another process currently holds the lock."""


class AdvisoryLock:
    def __init__(self, path, *, purpose: str, experiment_id: str):
        self.path = Path(path)
        self.purpose = purpose
        self.experiment_id = experiment_id
        self._file = None

    @property
    def held(self) -> bool:
        return self._file is not None

    def acquire(self) -> "AdvisoryLock":
        if self._file is not None:
            raise LockError("Lock is already held by this object")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = open(self.path, "a+", encoding="utf-8")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            diagnostics = self._read_diagnostics(handle)
            handle.close()
            holder = ""
            if diagnostics:
                holder = (
                    f" (last holder: pid {diagnostics.get('pid')} on "
                    f"{diagnostics.get('hostname')} since "
                    f"{diagnostics.get('acquired_at')})"
                )
            raise LockHeldError(
                f"The {self.purpose} lock for experiment "
                f"{self.experiment_id} is held by another process"
                f"{holder}."
            )
        handle.seek(0)
        handle.truncate()
        json.dump({
            "purpose": self.purpose,
            "experiment_id": self.experiment_id,
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "acquired_at": datetime.now(timezone.utc).isoformat(),
        }, handle, indent=2)
        handle.write("\n")
        handle.flush()
        self._file = handle
        return self

    @staticmethod
    def _read_diagnostics(handle):
        try:
            handle.seek(0)
            return json.load(handle)
        except (ValueError, OSError):
            return None

    def release(self) -> None:
        if self._file is None:
            return
        try:
            fcntl.flock(self._file.fileno(), fcntl.LOCK_UN)
        finally:
            self._file.close()
            self._file = None

    def __enter__(self) -> "AdvisoryLock":
        return self.acquire()

    def __exit__(self, *exc_info) -> None:
        self.release()


def execution_lock(experiment_dir, experiment_id: str) -> AdvisoryLock:
    return AdvisoryLock(
        Path(experiment_dir) / "execution.lock",
        purpose="execution", experiment_id=experiment_id,
    )


def analysis_lock(experiment_dir, experiment_id: str) -> AdvisoryLock:
    return AdvisoryLock(
        Path(experiment_dir) / "analysis.lock",
        purpose="analysis", experiment_id=experiment_id,
    )
