import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from werewolf.reporting.privacy import build_public_report
from werewolf.reporting.repository import GameRepository
from werewolf.web import app as web_app


SENTINEL = "NEVER_APPEAR_IN_PUBLIC_RESPONSE"


def write_private_game(root: Path, game_id="game_7_private"):
    rows = [
        {
            "type": "config", "game_id": game_id,
            "created_at": "2026-07-14T10:00:00Z",
            "seed": 7, "n_players": 3, "n_wolves": 1, "n_seers": 1,
            "event_schema_version": 2, "belief_snapshots": True,
            "belief_schema_version": 1,
            "model": "model", "model_alias": "model",
            "role_map": {
                "0": {"role": "villager", "team": "village"},
                "1": {"role": "werewolf", "team": "wolf"},
                "2": {"role": "seer", "team": "village"},
            },
        },
        {
            "type": "llm_call", "call_id": "private-call", "attempt": 1,
            "api_attempted": True, "api_ok": True, "parse_ok": True,
            "validation_ok": True, "error_category": "completed",
            "player_id": 1, "player_role": "werewolf", "round": 1,
            "phase": "night_wolf_chat", "required_action": "wolf_chat",
            "requested_model": SENTINEL, "resolved_model": SENTINEL,
            "usage": {"total_tokens": 1},
            "cost": {"source": "unavailable", "usd": None, "ticks": None},
        },
        {"type": "event", "event": {
            "id": 0, "event_id": "evt_000000", "round": 1,
            "phase": "night_wolf_chat", "type": "thought",
            "channel": "moderator_only", "speaker_id": 1,
            "source_call_id": "private-call", "discussion_cycle": None,
            "payload": {"thought": SENTINEL},
        }},
        {"type": "event", "event": {
            "id": 1, "event_id": "evt_000001", "round": 1,
            "phase": "night_wolf_chat", "type": "message",
            "channel": "werewolf", "speaker_id": 1,
            "source_call_id": "private-call", "discussion_cycle": None,
            "payload": {"text": SENTINEL},
        }},
        {"type": "event", "event": {
            "id": 2, "event_id": "evt_000002", "round": 1,
            "phase": "night_seer", "type": "divine_result",
            "channel": "seer_private", "speaker_id": 2,
            "source_call_id": "private-call", "discussion_cycle": None,
            "payload": {"target_id": 1, "is_werewolf": True, "secret": SENTINEL},
        }},
        {"type": "event", "event": {
            "id": 3, "event_id": "evt_000003", "round": 1,
            "phase": "day_announce", "type": "game_status",
            "channel": "public", "speaker_id": None,
            "source_call_id": None, "discussion_cycle": None,
            "payload": {"alive_wolves": 1, "alive_villagers": 2},
        }},
    ]
    path = root / f"{game_id}.jsonl"
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return path


class PublicProjectionTests(unittest.TestCase):
    def test_allowlist_drops_private_sentinels_at_every_depth(self):
        full = {
            "report_schema_version": 1,
            "source": {"log_name": "x.jsonl", "warnings": []},
            "overview": {
                "game_id": "game_1_x", "role_assignment": {"0": SENTINEL},
                "usage": {"by_player": {"0": SENTINEL}},
            },
            "players": [{"id": 0, "role": SENTINEL, "team": SENTINEL}],
            "timeline": [
                {"channel": "moderator_only", "payload": {"thought": SENTINEL}},
                {"channel": "werewolf", "payload": {"text": SENTINEL}},
                {"channel": "seer_private", "payload": {"secret": SENTINEL}},
            ],
            "beliefs": {"snapshots": [SENTINEL]},
            "decisions": {"attempt_groups": [{"model": SENTINEL}]},
            "manipulation_signals": {"episodes": [SENTINEL]},
            "usage": {"by_player": {"0": SENTINEL}},
            "reproducibility": {}, "links": {},
        }
        serialized = json.dumps(build_public_report(full))
        self.assertNotIn(SENTINEL, serialized)


class ReportApiTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.game_id = "game_7_private"
        write_private_game(self.root, self.game_id)
        self.old_repository = web_app.game_repository
        self.old_engine = web_app.game_engine
        web_app.game_repository = GameRepository(self.root)
        web_app.game_engine = None
        web_app.app.config["TESTING"] = True
        self.client = web_app.app.test_client()

    def tearDown(self):
        web_app.game_repository = self.old_repository
        web_app.game_engine = self.old_engine
        self.tmp.cleanup()

    def test_default_report_is_server_side_public_projection(self):
        response = self.client.get(f"/api/games/{self.game_id}/report")
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertFalse(body["privacy"]["include_private"])
        self.assertNotIn(SENTINEL, response.get_data(as_text=True))
        self.assertEqual(body["timeline"][0]["type"], "game_status")
        self.assertEqual(body["players"], [{"id": 0}, {"id": 1}, {"id": 2}])
        self.assertFalse(body["beliefs"]["available"])

    def test_private_report_refetch_and_raw_are_no_store(self):
        private = self.client.get(
            f"/api/games/{self.game_id}/report?include_private=true"
        )
        self.assertEqual(private.status_code, 200)
        self.assertEqual(private.headers["Cache-Control"], "no-store")
        self.assertIn(SENTINEL, private.get_data(as_text=True))

        raw = self.client.get(f"/api/games/{self.game_id}/raw")
        self.assertEqual(raw.status_code, 200)
        self.assertEqual(raw.headers["Cache-Control"], "no-store")
        self.assertIn("attachment", raw.headers["Content-Disposition"])
        self.assertTrue(raw.mimetype in {"application/x-ndjson", "application/ndjson"})
        raw.close()

    def test_history_overlays_active_without_persisting_it(self):
        web_app.game_engine = SimpleNamespace(
            state=SimpleNamespace(game_id=self.game_id)
        )
        response = self.client.get("/api/games")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["games"][0]["display_status"], "active")
        with open(web_app.game_repository.meta_path(self.game_id), encoding="utf-8") as f:
            self.assertNotIn("display_status", json.load(f))

    def test_new_game_appears_in_already_reconciled_history(self):
        # Reconcile first, matching the browser sequence from the audit.
        self.client.get("/api/games")
        new_game_id = "game_8_new_active"
        path = web_app.game_repository.log_path(new_game_id)
        path.write_text(json.dumps({
            "type": "config", "game_id": new_game_id,
            "created_at": "2026-07-14T11:00:00Z", "seed": 8,
            "n_players": 4, "n_wolves": 1, "n_seers": 0,
            "event_schema_version": 3, "model": "model",
            "model_alias": "model",
        }) + "\n", encoding="utf-8")
        engine = mock.Mock()
        engine.state = SimpleNamespace(game_id=new_game_id)
        engine.get_state_dict.return_value = {"game_id": new_game_id}
        with mock.patch.object(
            web_app, "create_engine_from_payload", return_value=engine,
        ):
            created = self.client.post("/api/new", json={"model": "fast"})
        self.assertEqual(created.status_code, 200)
        games = self.client.get("/api/games").get_json()["games"]
        active = next(game for game in games if game["game_id"] == new_game_id)
        self.assertEqual(active["completion_status"], "incomplete")
        self.assertEqual(active["display_status"], "active")

    def test_report_survives_without_active_engine_and_is_cached(self):
        response = self.client.get(
            f"/api/games/{self.game_id}/report?include_private=true"
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(web_app.game_repository.report_path(self.game_id).exists())
        self.assertEqual(response.get_json()["overview"]["display_status"], "incomplete")

    def test_report_cache_invalidates_when_jsonl_changes(self):
        first = self.client.get(
            f"/api/games/{self.game_id}/report?include_private=true"
        ).get_json()
        self.assertEqual(first["overview"]["completion_status"], "incomplete")
        with open(web_app.game_repository.log_path(self.game_id), "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "type": "outcome", "winner": "wolf", "rounds": 1,
                "remaining": [1],
            }) + "\n")
        second = self.client.get(
            f"/api/games/{self.game_id}/report?include_private=true"
        ).get_json()
        self.assertEqual(second["overview"]["completion_status"], "completed")
        self.assertEqual(second["overview"]["winner"], "wolf")

    def test_report_cache_invalidates_when_build_version_changes(self):
        first = self.client.get(
            f"/api/games/{self.game_id}/report?include_private=true"
        ).get_json()
        report_path = web_app.game_repository.report_path(self.game_id)
        with open(report_path, encoding="utf-8") as handle:
            cached = json.load(handle)
        cached["report_build_version"] = -1
        cached["overview"]["winner"] = "stale-winner"
        with open(report_path, "w", encoding="utf-8") as handle:
            json.dump(cached, handle)
        second = self.client.get(
            f"/api/games/{self.game_id}/report?include_private=true"
        ).get_json()
        self.assertEqual(second["report_build_version"], first["report_build_version"])
        self.assertIsNone(second["overview"]["winner"])

    def test_invalid_paths_and_query_are_rejected(self):
        self.assertEqual(
            self.client.get("/api/games/not-a-game/report").status_code, 404
        )
        self.assertEqual(
            self.client.get(
                f"/api/games/{self.game_id}/report?include_private=maybe"
            ).status_code,
            400,
        )

    def test_history_page_loads_persistent_history_client(self):
        history = self.client.get("/games")
        self.assertEqual(history.status_code, 200)
        text = history.get_data(as_text=True)
        self.assertIn("Game History", text)
        self.assertIn("/static/history.js", text)
        self.assertIn("history-status", text)

        live = self.client.get("/").get_data(as_text=True)
        self.assertIn('href="/games"', live)
        javascript = (
            Path(__file__).parents[1] / "werewolf" / "web" / "static" / "app.js"
        ).read_text(encoding="utf-8")
        self.assertIn("Open forensic report", javascript)
        self.assertIn("display.replaceChildren(result, report)", javascript)

    def test_report_page_has_forensic_sections_without_calibration_claims(self):
        response = self.client.get(f"/games/{self.game_id}")
        self.assertEqual(response.status_code, 200)
        text = response.get_data(as_text=True)
        for section in (
            "overview", "timeline", "beliefs", "decisions", "manipulation",
            "usage", "reproducibility",
        ):
            self.assertIn(f'id="{section}"', text)
        self.assertIn("/static/report.js", text)
        self.assertNotIn("calibration", text.lower())
        javascript = (
            Path(__file__).parents[1] / "werewolf" / "web" / "static" / "report.js"
        ).read_text(encoding="utf-8")
        self.assertIn("flushSegment", javascript)
        self.assertIn("point.snapshot_valid ? 'valid' : 'invalid'", javascript)
        self.assertIn("item.evidence_quality", javascript)
        self.assertIn("beliefs.checkpoints || []", javascript)
        self.assertIn("checkpoint-marker", javascript)


if __name__ == "__main__":
    unittest.main()
