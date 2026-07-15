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
from werewolf.reporting.builder import REPORT_BUILD_VERSION, REPORT_SCHEMA_VERSION


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


def llm_call(**overrides) -> dict:
    row = {
        "type": "llm_call", "schema_version": 2, "call_id": "call-1",
        "attempt": 1, "api_attempted": True, "api_ok": True,
        "parse_ok": True, "validation_ok": True,
        "error_category": "completed", "player_id": 0,
        "player_role": "villager", "phase": "day_discuss", "round": 1,
        "required_action": "speak_public", "requested_model": "model-a",
        "resolved_model": "model-a", "usage": {"total_tokens": 1},
        "cost": {"source": "provider_reported", "usd": 0.1},
    }
    row.update(overrides)
    return row


def completed_rows(game_id: str) -> list[dict]:
    cfg = config(game_id, "2026-07-14T10:00:00Z")
    cfg["event_schema_version"] = 2
    return [
        cfg,
        llm_call(),
        {"type": "event", "event": {
            "id": 0, "event_id": "evt_000000", "round": 1,
            "phase": "day_discuss", "type": "message", "channel": "public",
            "speaker_id": 0, "source_call_id": "call-1",
            "discussion_cycle": 1, "payload": {"text": "hello"},
        }},
        {"type": "usage_summary", "usage": {
            "calls": 1, "retries": 0, "fallbacks": 0,
            "api_failures": 0, "parse_failures": 0,
            "validation_failures": 0, "recovered_parses": 0,
            "tokens": {
                "input_tokens": 0, "cached_input_tokens": 0,
                "output_tokens": 0, "reasoning_tokens": 0,
                "total_tokens": 1,
            },
            "cost_usd_total": 0.1,
        }},
        {"type": "outcome", "winner": "village", "rounds": 1},
    ]


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

    def test_fresh_metadata_with_stale_build_version_is_rebuilt(self):
        game_id = "game_1_stale_build"
        write_rows(self.root, game_id, completed_rows(game_id))
        repository = GameRepository(self.root)
        repository.rebuild()
        for path in (
            repository.meta_path(game_id), repository.report_path(game_id),
        ):
            with open(path, encoding="utf-8") as handle:
                payload = json.load(handle)
            payload["report_build_version"] = REPORT_BUILD_VERSION - 1
            if path == repository.meta_path(game_id):
                payload["analysis_eligibility"] = "stale-value"
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)

        history = GameRepository(self.root).list_games()["games"][0]
        self.assertEqual(history["report_schema_version"], REPORT_SCHEMA_VERSION)
        self.assertEqual(history["report_build_version"], REPORT_BUILD_VERSION)
        self.assertNotEqual(history["analysis_eligibility"], "stale-value")

    def test_rebuild_regenerates_missing_report_sidecar(self):
        game_id = "game_1_missing_report"
        write_rows(self.root, game_id, completed_rows(game_id))
        repository = GameRepository(self.root)
        repository.rebuild()
        repository.report_path(game_id).unlink()
        self.assertTrue(repository.meta_path(game_id).exists())

        repository.rebuild()
        self.assertTrue(repository.report_path(game_id).exists())
        with open(repository.report_path(game_id), encoding="utf-8") as handle:
            report = json.load(handle)
        self.assertEqual(report["report_build_version"], REPORT_BUILD_VERSION)

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

    def test_history_uses_canonical_report_semantics_before_report_request(self):
        cases = {}

        fallback = completed_rows("game_12_fallback")
        fallback.insert(-2, llm_call(
            call_id="call-fallback", api_attempted=False, api_ok=False,
            parse_ok=False, validation_ok=False, required_action="vote",
            error_category="fallback_used", cost={"source": "unavailable"},
        ))
        cases["game_12_fallback"] = fallback

        mismatch = completed_rows("game_13_model")
        mismatch[1]["resolved_model"] = "model-b"
        cases["game_13_model"] = mismatch

        terminal_mismatch = completed_rows("game_14_usage")
        terminal_mismatch[-2]["usage"]["cost_usd_total"] = 99.0
        cases["game_14_usage"] = terminal_mismatch

        missing_call = completed_rows("game_15_missing_call")
        missing_call[2]["event"]["source_call_id"] = "call-does-not-exist"
        cases["game_15_missing_call"] = missing_call

        for game_id, rows in cases.items():
            write_rows(self.root, game_id, rows)

        malformed_id = "game_16_malformed"
        malformed_path = write_rows(
            self.root, malformed_id, completed_rows(malformed_id),
        )
        with open(malformed_path, "a", encoding="utf-8") as handle:
            handle.write("{not-json\n")

        history = {
            row["game_id"]: row
            for row in GameRepository(self.root).list_games()["games"]
        }
        self.assertEqual(history["game_12_fallback"]["analysis_eligibility"], "ineligible")
        self.assertIn(
            "validity:fallback_vote",
            history["game_12_fallback"]["analysis_exclusion_reasons"],
        )
        self.assertEqual(history["game_13_model"]["analysis_eligibility"], "ineligible")
        self.assertIn(
            "validity:resolved_model_mismatch",
            history["game_13_model"]["analysis_exclusion_reasons"],
        )
        self.assertEqual(history["game_14_usage"]["known_cost_usd"], 0.1)
        self.assertEqual(history["game_14_usage"]["usage_reliability"], "inconsistent")
        self.assertEqual(history["game_14_usage"]["integrity_status"], "warnings")
        self.assertEqual(history["game_15_missing_call"]["analysis_eligibility"], "ineligible")
        self.assertIn(
            "missing_strategic_call_evidence",
            history["game_15_missing_call"]["analysis_exclusion_reasons"],
        )
        self.assertEqual(history[malformed_id]["integrity_status"], "warnings")
        self.assertEqual(history[malformed_id]["analysis_eligibility"], "eligible")


if __name__ == "__main__":
    unittest.main()
