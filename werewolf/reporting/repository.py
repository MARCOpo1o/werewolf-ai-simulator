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


INDEX_SCHEMA_VERSION = 1
META_SCHEMA_VERSION = 1
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


def _atomic_json_write(path: Path, payload: dict) -> None:
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
        _atomic_json_write(self.index_path, {
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
            and old.get("source_size") == stat.st_size
            and old.get("source_mtime_ns") == stat.st_mtime_ns
        ):
            return old

        config = outcome = usage_summary = None
        timestamps: list[str] = []
        warnings = 0
        record_count = 0
        try:
            with open(log_path, encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        warnings += 1
                        continue
                    if not isinstance(row, dict):
                        warnings += 1
                        continue
                    record_count += 1
                    row_type = row.get("type")
                    if row_type == "config" and config is None:
                        config = row
                    elif row_type == "outcome":
                        outcome = row
                    elif row_type == "usage_summary":
                        usage_summary = row.get("usage")
                    for candidate in (
                        row.get("created_at"), row.get("ts"),
                        (row.get("event") or {}).get("t")
                        if isinstance(row.get("event"), dict) else None,
                    ):
                        timestamp = _utc_iso(candidate)
                        if timestamp:
                            timestamps.append(timestamp)

        except OSError:
            return None

        config = config or {}
        configured_id = config.get("game_id")
        if configured_id and configured_id != game_id:
            warnings += 1
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

        models = []
        role_models = config.get("role_models")
        if isinstance(role_models, dict):
            models = sorted({
                info.get("alias") or info.get("requested_model")
                for info in role_models.values() if isinstance(info, dict)
            } - {None})
        elif config.get("model_alias") or config.get("model"):
            models = [config.get("model_alias") or config.get("model")]

        entry = {
            "meta_schema_version": META_SCHEMA_VERSION,
            "game_id": game_id,
            "log_name": log_path.name,
            "completion_status": "completed" if outcome else "incomplete",
            "integrity_status": (
                "corrupt" if not config and record_count == 0
                else "warnings" if warnings else "clean"
            ),
            "analysis_eligibility": (
                "ineligible" if not config and record_count == 0
                else "limited" if not outcome else "eligible"
            ),
            "analysis_exclusion_reasons": (
                ["unrecoverable_log"] if not config and record_count == 0
                else ["game_incomplete"] if not outcome else []
            ),
            "created_at": created_at,
            "created_at_source": created_source,
            "winner": outcome.get("winner") if outcome else None,
            "rounds": outcome.get("rounds") if outcome else None,
            "seed": config.get("seed"),
            "n_players": config.get("n_players"),
            "models": models,
            "known_cost_usd": (
                usage_summary.get("cost_usd_total")
                if isinstance(usage_summary, dict) else None
            ),
            "warning_count": warnings,
            "record_count": record_count,
            "source_size": stat.st_size,
            "source_mtime_ns": stat.st_mtime_ns,
        }
        _atomic_json_write(self.meta_path(game_id), entry)
        return entry


__all__ = [
    "GameRepository", "InvalidCursor", "InvalidGameId",
    "INDEX_SCHEMA_VERSION", "META_SCHEMA_VERSION", "validate_game_id",
]
