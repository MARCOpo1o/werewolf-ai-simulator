import json
import os
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock

from werewolf.reporting.repository import (
    GameRepository,
    InvalidCursor,
    InvalidGameId,
)


def write_rows(root: Path, game_id: str, rows: list[dict]) -> Path:
    path = root / f"{game_id}.jsonl"
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return path


def config(game_id: str, created_at=None, seed=1) -> dict:
    row = {
        "type": "config", "game_id": game_id, "seed": seed,
        "n_players": 4, "model_alias": "fast", "model": "grok-4.3",
    }
    if created_at:
        row["created_at"] = created_at
    return row


class GameRepositoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_rebuild_creates_index_and_sidecars_from_jsonl(self):
        game_id = "game_1_alpha"
        write_rows(self.root, game_id, [
            config(game_id, "2026-07-14T10:00:00Z"),
            {"type": "outcome", "winner": "village", "rounds": 2},
        ])
        repository = GameRepository(self.root)
        entries = repository.rebuild()

        self.assertEqual([e["game_id"] for e in entries], [game_id])
        self.assertTrue((self.root / "index.json").exists())
        self.assertTrue(repository.meta_path(game_id).exists())
        self.assertEqual(entries[0]["completion_status"], "completed")
        self.assertEqual(entries[0]["created_at_source"], "config_record")

        os.unlink(self.root / "index.json")
        rebuilt = GameRepository(self.root).list_games()
        self.assertEqual(rebuilt["games"][0]["game_id"], game_id)

    def test_active_is_runtime_overlay_never_persisted(self):
        game_id = "game_2_active"
        write_rows(self.root, game_id, [
            config(game_id, "2026-07-14T11:00:00Z"),
        ])
        repository = GameRepository(self.root)
        active = repository.list_games(active_game_id=game_id)["games"][0]
        self.assertEqual(active["completion_status"], "incomplete")
        self.assertEqual(active["display_status"], "active")

        with open(repository.meta_path(game_id), encoding="utf-8") as handle:
            stored = json.load(handle)
        self.assertNotIn("display_status", stored)
        self.assertEqual(stored["completion_status"], "incomplete")

        after_restart = GameRepository(self.root).list_games()["games"][0]
        self.assertEqual(after_restart["display_status"], "incomplete")

    def test_incremental_refresh_marks_completion(self):
        game_id = "game_3_refresh"
        path = write_rows(self.root, game_id, [
            config(game_id, "2026-07-14T12:00:00Z"),
        ])
        repository = GameRepository(self.root)
        repository.rebuild()
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "type": "outcome", "winner": "wolf", "rounds": 1,
            }) + "\n")
        refreshed = repository.refresh_game(game_id)
        self.assertEqual(refreshed["completion_status"], "completed")
        self.assertEqual(refreshed["winner"], "wolf")

    def test_created_at_precedence_upgrades_but_never_downgrades(self):
        game_id = "game_4_time"
        path = write_rows(self.root, game_id, [
            config(game_id),
            {"type": "llm_call", "ts": "2026-07-14T13:00:00Z"},
        ])
        repository = GameRepository(self.root)
        first = repository.rebuild()[0]
        self.assertEqual(first["created_at_source"], "record_timestamp")

        rows = [
            config(game_id, "2026-07-14T09:00:00Z"),
            {"type": "llm_call", "ts": "2026-07-14T13:00:00Z"},
        ]
        write_rows(self.root, game_id, rows)
        upgraded = repository.refresh_game(game_id)
        self.assertEqual(upgraded["created_at"], "2026-07-14T09:00:00Z")
        self.assertEqual(upgraded["created_at_source"], "config_record")

        # A changed filesystem timestamp cannot displace canonical creation time.
        os.utime(path, None)
        retained = repository.refresh_game(game_id)
        self.assertEqual(retained["created_at"], upgraded["created_at"])
        self.assertEqual(retained["created_at_source"], "config_record")

    def test_stable_cursor_uses_created_at_and_game_id(self):
        for number in range(3):
            game_id = f"game_{number}_cursor"
            write_rows(self.root, game_id, [
                config(game_id, f"2026-07-14T0{number}:00:00Z", number),
            ])
        repository = GameRepository(self.root)
        first = repository.list_games(limit=2)
        second = repository.list_games(limit=2, cursor=first["next_cursor"])
        self.assertEqual(len(first["games"]), 2)
        self.assertEqual(len(second["games"]), 1)
        ids = [g["game_id"] for g in first["games"] + second["games"]]
        self.assertEqual(len(ids), len(set(ids)))
        with self.assertRaises(InvalidCursor):
            repository.list_games(cursor="not-a-cursor")

    def test_orphaned_derived_files_are_removed(self):
        orphan = self.root / "game_9_orphan.meta.json"
        orphan.write_text("{}", encoding="utf-8")
        report = self.root / "game_9_orphan.report.json"
        report.write_text("{}", encoding="utf-8")
        GameRepository(self.root).rebuild()
        self.assertFalse(orphan.exists())
        self.assertFalse(report.exists())

    def test_invalid_game_id_cannot_become_a_path(self):
        repository = GameRepository(self.root)
        with self.assertRaises(InvalidGameId):
            repository.log_path("../../secret")

    def test_normal_list_requests_do_not_rescan_jsonl(self):
        game_id = "game_10_once"
        write_rows(self.root, game_id, [config(game_id, "2026-07-14T10:00:00Z")])
        repository = GameRepository(self.root)
        repository.list_games()
        with mock.patch.object(
            repository, "_refresh_path", wraps=repository._refresh_path,
        ) as refresh:
            repository.list_games()
        refresh.assert_not_called()

    def test_threaded_incremental_updates_are_serialized_in_process(self):
        game_id = "game_11_threads"
        write_rows(self.root, game_id, [config(game_id, "2026-07-14T10:00:00Z")])
        repository = GameRepository(self.root)
        repository.rebuild()
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(
                lambda _: repository.refresh_game(game_id), range(8),
            ))
        self.assertTrue(all(result["game_id"] == game_id for result in results))
        with open(repository.index_path, encoding="utf-8") as handle:
            index = json.load(handle)
        self.assertEqual(len(index["games"]), 1)
        self.assertFalse(list(self.root.glob("*.tmp")))


if __name__ == "__main__":
    unittest.main()
