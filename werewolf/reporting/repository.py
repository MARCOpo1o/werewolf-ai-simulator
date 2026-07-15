"""Rebuildable game-log index and atomic derived-file storage.

JSONL logs are canonical. Metadata sidecars and ``index.json`` are caches
that may be deleted and reconstructed without losing game information.
The lock here protects threads in one process only; multi-process writers
are deliberately outside the PR 2 storage contract.
"""
from __future__ import annotations

import base64
import json
import os
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from werewolf.reporting.builder import (
    REPORT_BUILD_VERSION,
    REPORT_SCHEMA_VERSION,
    build_full_report,
    build_history_summary,
)
from werewolf.reporting.parser import ParsedGameLog, parse_game_log


INDEX_SCHEMA_VERSION = 1
META_SCHEMA_VERSION = 2
_GAME_ID = re.compile(r"^game_[A-Za-z0-9_-]{1,190}$")
_CREATED_SOURCE_RANK = {
    "filesystem": 1,
    "record_timestamp": 2,
    "config_record": 3,
}


class InvalidGameId(ValueError):
    pass


class InvalidCursor(ValueError):
    pass


def validate_game_id(game_id: str) -> str:
    if not isinstance(game_id, str) or not _GAME_ID.fullmatch(game_id):
        raise InvalidGameId("Invalid game ID")
    return game_id


def _utc_iso(value: Any) -> Optional[str]:
    try:
        if isinstance(value, bool) or value is None:
            return None
        if isinstance(value, (int, float)):
            dt = datetime.fromtimestamp(float(value), tz=timezone.utc)
        elif isinstance(value, str):
            text = value.strip().replace("Z", "+00:00")
            dt = datetime.fromisoformat(text)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            dt = dt.astimezone(timezone.utc)
        else:
            return None
        return dt.isoformat().replace("+00:00", "Z")
    except (OverflowError, OSError, TypeError, ValueError):
        return None


def atomic_json_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except (AttributeError, OSError):
            pass  # best-effort directory durability on supported systems
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def _read_json(path: Path) -> Optional[dict]:
    try:
        with open(path, encoding="utf-8") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def _cursor_encode(entry: dict) -> str:
    raw = json.dumps({
        "v": 1,
        "created_at": entry["created_at"],
        "game_id": entry["game_id"],
    }, separators=(",", ":"), sort_keys=True).encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _cursor_decode(cursor: str) -> tuple[str, str]:
    try:
        padded = cursor + "=" * (-len(cursor) % 4)
        value = json.loads(base64.urlsafe_b64decode(padded).decode())
        if value.get("v") != 1:
            raise ValueError
        created_at = _utc_iso(value.get("created_at"))
        game_id = validate_game_id(value.get("game_id"))
        if not created_at:
            raise ValueError
        return created_at, game_id
    except Exception as exc:
        raise InvalidCursor("Invalid history cursor") from exc


class GameRepository:
    def __init__(self, root: str | Path = "outputs/games"):
        self.root = Path(root)
        self.index_path = self.root / "index.json"
        self._lock = threading.RLock()
        self._entries: dict[str, dict] = {}
        self._reconciled = False

    def log_path(self, game_id: str) -> Path:
        return self.root / f"{validate_game_id(game_id)}.jsonl"

    def meta_path(self, game_id: str) -> Path:
        return self.root / f"{validate_game_id(game_id)}.meta.json"

    def report_path(self, game_id: str) -> Path:
        return self.root / f"{validate_game_id(game_id)}.report.json"

    def ensure_reconciled(self) -> None:
        if self._reconciled:
            return
        self.rebuild()

    def rebuild(self) -> list[dict]:
        """Explicit full reconciliation for startup, tests, and maintenance."""
        with self._lock:
            self.root.mkdir(parents=True, exist_ok=True)
            entries: dict[str, dict] = {}
            live_logs = set()
            for log_path in sorted(self.root.glob("game_*.jsonl")):
                if not _GAME_ID.fullmatch(log_path.stem):
                    continue
                live_logs.add(log_path.stem)
                entry = self._refresh_path(log_path)
                if entry is not None:
                    entries[entry["game_id"]] = entry

            for suffix in (".meta.json", ".report.json"):
                for derived in self.root.glob(f"game_*{suffix}"):
                    game_id = derived.name[:-len(suffix)]
                    if game_id not in live_logs:
                        try:
                            derived.unlink()
                        except OSError:
                            pass

            self._entries = entries
            self._write_index()
            self._reconciled = True
            return self._sorted_entries()

    def refresh_game(self, game_id: str) -> Optional[dict]:
        """Incrementally reconcile one game, normally after a phase/completion."""
        with self._lock:
            self.ensure_reconciled()
            log_path = self.log_path(game_id)
            if not log_path.exists():
                self._entries.pop(game_id, None)
                self._write_index()
                return None
            entry = self._refresh_path(log_path)
            if entry is not None:
                self._entries[game_id] = entry
            self._write_index()
            return entry

    def list_games(
        self,
        *,
        limit: int = 50,
        cursor: Optional[str] = None,
        active_game_id: Optional[str] = None,
    ) -> dict:
        with self._lock:
            self.ensure_reconciled()
            limit = max(1, min(int(limit), 100))
            entries = self._sorted_entries()
            if cursor:
                marker = _cursor_decode(cursor)
                entries = [
                    entry for entry in entries
                    if (entry["created_at"], entry["game_id"]) < marker
                ]
            page = entries[:limit]
            display = []
            for entry in page:
                item = dict(entry)
                item["display_status"] = (
                    "active" if item["game_id"] == active_game_id
                    else item["completion_status"]
                )
                display.append(item)
            next_cursor = (
                _cursor_encode(page[-1]) if len(entries) > limit else None
            )
            return {"games": display, "next_cursor": next_cursor}

    def get_entry(
        self, game_id: str, *, active_game_id: Optional[str] = None,
    ) -> Optional[dict]:
        with self._lock:
            self.ensure_reconciled()
            entry = self._entries.get(validate_game_id(game_id))
            if entry is None:
                return None
            item = dict(entry)
            item["display_status"] = (
                "active" if game_id == active_game_id
                else item["completion_status"]
            )
            return item

    def _sorted_entries(self) -> list[dict]:
        return sorted(
            (dict(entry) for entry in self._entries.values()),
            key=lambda entry: (entry["created_at"], entry["game_id"]),
            reverse=True,
        )

    def _write_index(self) -> None:
        atomic_json_write(self.index_path, {
            "index_schema_version": INDEX_SCHEMA_VERSION,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "games": self._sorted_entries(),
        })

    def _refresh_path(self, log_path: Path) -> Optional[dict]:
        stat = log_path.stat()
        game_id = log_path.stem
        old = _read_json(self.meta_path(game_id)) or {}
        if (
            old.get("meta_schema_version") == META_SCHEMA_VERSION
            and old.get("report_schema_version") == REPORT_SCHEMA_VERSION
            and old.get("report_build_version") == REPORT_BUILD_VERSION
            and old.get("source_size") == stat.st_size
            and old.get("source_mtime_ns") == stat.st_mtime_ns
            and self._report_sidecar_matches(game_id, stat)
        ):
            return old

        try:
            parsed = parse_game_log(log_path)
        except OSError:
            return None

        config = parsed.config or {}
        timestamps = self._record_timestamps(parsed)
        config_created = _utc_iso(config.get("created_at"))
        if config_created:
            created_at, created_source = config_created, "config_record"
        elif timestamps:
            created_at, created_source = min(timestamps), "record_timestamp"
        else:
            fs_timestamp = getattr(stat, "st_birthtime", stat.st_mtime)
            created_at = _utc_iso(fs_timestamp)
            created_source = "filesystem"

        old_source = old.get("created_at_source")
        if (
            old.get("created_at")
            and _CREATED_SOURCE_RANK.get(old_source, 0)
            >= _CREATED_SOURCE_RANK[created_source]
        ):
            created_at = old["created_at"]
            created_source = old_source

        report = build_full_report(parsed, metadata={
            "game_id": game_id,
            "created_at": created_at,
            "created_at_source": created_source,
        })
        atomic_json_write(self.report_path(game_id), report)
        entry = {
            "meta_schema_version": META_SCHEMA_VERSION,
            **build_history_summary(report),
            "log_name": log_path.name,
            "created_at": created_at,
            "created_at_source": created_source,
            "source_size": stat.st_size,
            "source_mtime_ns": stat.st_mtime_ns,
        }
        atomic_json_write(self.meta_path(game_id), entry)
        return entry

    def _report_sidecar_matches(self, game_id: str, stat: os.stat_result) -> bool:
        report = _read_json(self.report_path(game_id))
        if report is None:
            return False
        source = report.get("source")
        if not isinstance(source, dict):
            return False
        return (
            report.get("report_schema_version") == REPORT_SCHEMA_VERSION
            and report.get("report_build_version") == REPORT_BUILD_VERSION
            and source.get("size_bytes") == stat.st_size
            and source.get("mtime_ns") == stat.st_mtime_ns
        )

    @staticmethod
    def _record_timestamps(parsed: ParsedGameLog) -> list[str]:
        timestamps = []
        for wrapped in parsed.rows:
            row = wrapped["record"]
            event = row.get("event") if isinstance(row.get("event"), dict) else {}
            for candidate in (row.get("created_at"), row.get("ts"), event.get("t")):
                timestamp = _utc_iso(candidate)
                if timestamp:
                    timestamps.append(timestamp)
        return timestamps

    def update_from_report(self, game_id: str, report: dict) -> None:
        """Copy report headline fields into the derived history metadata."""
        with self._lock:
            self.ensure_reconciled()
            entry = self._entries.get(validate_game_id(game_id))
            if entry is None:
                return
            entry.update(build_history_summary(report))
            atomic_json_write(self.meta_path(game_id), entry)
            self._write_index()


__all__ = [
    "GameRepository", "InvalidCursor", "InvalidGameId",
    "INDEX_SCHEMA_VERSION", "META_SCHEMA_VERSION", "atomic_json_write",
    "validate_game_id",
]
