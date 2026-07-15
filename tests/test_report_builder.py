import json
import tempfile
import unittest
from pathlib import Path

from werewolf.reporting.builder import build_full_report_from_file
from werewolf.reporting.parser import parse_game_log


def call(call_id="call-1", *, cost=0.1, source="provider_reported"):
    return {
        "type": "llm_call", "schema_version": 2,
        "call_id": call_id, "attempt": 1,
        "api_attempted": True, "api_ok": True,
        "parse_ok": True, "parse_method": "direct", "validation_ok": True,
        "error_category": "completed", "player_id": 0,
        "player_role": "villager", "phase": "day_discuss", "round": 1,
        "required_action": "speak_public", "requested_model": "model-a",
        "resolved_model": "model-a",
        "usage": {
            "input_tokens": 10, "cached_input_tokens": 0,
            "output_tokens": 5, "reasoning_tokens": 0, "total_tokens": 15,
        },
        "cost": {"source": source, "ticks": None, "usd": cost},
    }


def terminal(cost=0.1):
    return {
        "type": "usage_summary",
        "usage": {
            "calls": 1, "retries": 0, "fallbacks": 0,
            "api_failures": 0, "parse_failures": 0,
            "validation_failures": 0, "recovered_parses": 0,
            "tokens": {
                "input_tokens": 10, "cached_input_tokens": 0,
                "output_tokens": 5, "reasoning_tokens": 0,
                "total_tokens": 15,
            },
            "cost_usd_total": cost,
        },
    }


def fixture_rows(game_id="game_1_report"):
    return [
        {
            "type": "config", "game_id": game_id,
            "created_at": "2026-07-14T10:00:00Z",
            "seed": 1, "n_players": 4, "n_wolves": 1, "n_seers": 0,
            "event_schema_version": 2, "log_schema_version": 2,
            "role_map": {
                "0": {"role": "villager", "team": "village"},
                "1": {"role": "werewolf", "team": "wolf"},
            },
        },
        call(),
        {
            "type": "event",
            "event": {
                "id": 0, "event_id": "evt_000000", "round": 1,
                "phase": "day_discuss", "type": "message", "channel": "public",
                "speaker_id": 0, "source_call_id": "call-1",
                "discussion_cycle": 1, "payload": {"text": "hello"},
            },
        },
        terminal(),
        {"type": "outcome", "winner": "village", "rounds": 1, "remaining": [0]},
    ]


class ReportBuilderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, rows, game_id="game_1_report", malformed=None):
        path = self.root / f"{game_id}.jsonl"
        with open(path, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
            if malformed:
                handle.write(malformed + "\n")
        return path

    def test_completed_current_log_is_eligible(self):
        report = build_full_report_from_file(self.write(fixture_rows()))
        overview = report["overview"]
        self.assertEqual(overview["completion_status"], "completed")
        self.assertEqual(overview["integrity_status"], "clean")
        self.assertEqual(overview["analysis_eligibility"], "eligible")
        self.assertEqual(overview["usage_reliability"], "reliable")
        self.assertEqual(report["usage"]["known_cost_usd"], 0.1)
        self.assertEqual(report["timeline"][0]["link_quality"], "exact")
        self.assertEqual(report["timeline"][0]["source_line"], 3)

    def test_terminal_mismatch_warns_but_does_not_change_eligibility(self):
        rows = fixture_rows()
        rows[-2] = terminal(cost=99.0)
        report = build_full_report_from_file(self.write(rows))
        self.assertEqual(report["overview"]["integrity_status"], "warnings")
        self.assertEqual(report["overview"]["usage_reliability"], "inconsistent")
        self.assertEqual(report["overview"]["analysis_eligibility"], "eligible")
        self.assertEqual(report["usage"]["known_cost_usd"], 0.1)
        self.assertEqual(
            report["usage"]["terminal_consistency"]["status"], "mismatched",
        )

    def test_missing_canonical_call_evidence_is_ineligible(self):
        rows = fixture_rows()
        rows[2]["event"]["source_call_id"] = "missing-call"
        report = build_full_report_from_file(self.write(rows))
        self.assertEqual(report["overview"]["analysis_eligibility"], "ineligible")
        self.assertIn(
            "missing_strategic_call_evidence",
            report["overview"]["analysis_exclusion_reasons"],
        )

    def test_exact_link_requires_semantically_matching_call(self):
        rows = fixture_rows()
        rows[1]["phase"] = "night_wolf_chat"
        report = build_full_report_from_file(self.write(rows))
        self.assertEqual(report["timeline"][0]["link_quality"], "mismatched")
        self.assertEqual(report["overview"]["analysis_eligibility"], "ineligible")
        self.assertIn(
            "missing_strategic_call_evidence",
            report["overview"]["analysis_exclusion_reasons"],
        )

    def test_kill_vote_call_mappings_are_validated_and_exposed(self):
        game_id = "game_19_kill_links"
        config = fixture_rows(game_id)[0]
        config["event_schema_version"] = 3
        wolf_one = call("call-w1", cost=0.1)
        wolf_one.update({
            "player_id": 1, "player_role": "werewolf",
            "phase": "night_wolf_kill", "required_action": "choose_wolf_kill",
        })
        wolf_two = call("call-w3", cost=0.1)
        wolf_two.update({
            "player_id": 3, "player_role": "werewolf",
            "phase": "night_wolf_kill", "required_action": "choose_wolf_kill",
        })
        kill = {"type": "event", "event": {
            "id": 0, "event_id": "evt_000000", "round": 1,
            "phase": "night_wolf_kill", "type": "kill",
            "channel": "moderator_only", "speaker_id": None,
            "source_call_id": None, "payload": {
                "victim_id": 0, "votes": {"1": 0, "3": 0},
                "vote_source_call_ids": {"1": "call-w1", "3": "call-w3"},
            },
        }}
        rows = [
            config, wolf_one, wolf_two, kill,
            {"type": "outcome", "winner": "wolf", "rounds": 1},
        ]
        report = build_full_report_from_file(self.write(rows, game_id=game_id))
        event = report["timeline"][0]
        self.assertEqual(event["link_quality"], "exact")
        self.assertEqual(
            [link["link_quality"] for link in event["vote_source_links"]],
            ["exact", "exact"],
        )
        self.assertEqual(report["overview"]["analysis_eligibility"], "eligible")

        rows[2]["required_action"] = "speak_public"
        damaged = build_full_report_from_file(self.write(rows, game_id=game_id))
        self.assertEqual(damaged["timeline"][0]["link_quality"], "mismatched")
        self.assertEqual(damaged["overview"]["analysis_eligibility"], "ineligible")

    def test_malformed_line_is_preserved_as_integrity_warning(self):
        path = self.write(fixture_rows(), malformed="{definitely-not-json")
        parsed = parse_game_log(path)
        self.assertEqual(parsed.warnings[0].source_line, 6)
        report = build_full_report_from_file(path)
        self.assertEqual(report["overview"]["integrity_status"], "warnings")
        self.assertEqual(report["overview"]["analysis_eligibility"], "eligible")

    def test_mixed_cost_sources_are_explicit_and_partial(self):
        rows = fixture_rows()
        second = call("call-2", cost=None, source="unavailable")
        rows.insert(2, second)
        rows[-2]["usage"]["calls"] = 2
        # Terminal total still agrees with the known portion.
        report = build_full_report_from_file(self.write(rows))
        usage = report["usage"]
        self.assertEqual(usage["known_cost_usd"], 0.1)
        self.assertEqual(usage["calls_with_known_cost"], 1)
        self.assertEqual(usage["calls_without_known_cost"], 1)
        self.assertEqual(usage["cost_completeness"], "partial")
        self.assertEqual(
            usage["cost_sources"], ["provider_reported", "unavailable"],
        )

    def test_legacy_completed_log_is_limited_not_corrupt(self):
        rows = fixture_rows()
        rows[0].pop("event_schema_version")
        rows[2]["event"].pop("event_id")
        rows[2]["event"].pop("source_call_id")
        report = build_full_report_from_file(self.write(rows))
        self.assertEqual(report["overview"]["analysis_eligibility"], "limited")
        self.assertEqual(report["timeline"][0]["event_id"], "evt_000000")

    def test_all_malformed_log_is_corrupt(self):
        path = self.root / "game_9_corrupt.jsonl"
        path.write_text("not-json\n", encoding="utf-8")
        report = build_full_report_from_file(path)
        self.assertEqual(report["overview"]["integrity_status"], "corrupt")
        self.assertEqual(report["overview"]["analysis_eligibility"], "ineligible")

    def test_incomplete_instrumented_log_remains_limited_while_validity_is_provisional(self):
        rows = fixture_rows()[:-2]
        rows[0]["belief_snapshots"] = True
        report = build_full_report_from_file(self.write(rows))
        self.assertEqual(report["overview"]["completion_status"], "incomplete")
        self.assertEqual(report["overview"]["analysis_eligibility"], "limited")
        self.assertTrue(report["overview"]["validity"]["provisional"])

    def test_filename_game_id_is_canonical_when_config_disagrees(self):
        rows = fixture_rows("game_wrong_config")
        path = self.write(rows, game_id="game_17_canonical")
        report = build_full_report_from_file(path)
        self.assertEqual(report["overview"]["game_id"], "game_17_canonical")
        self.assertEqual(
            report["links"]["report"], "/games/game_17_canonical",
        )
        warnings = report["source"]["warnings"]
        self.assertTrue(any(w["code"] == "game_id_mismatch" for w in warnings))
        self.assertEqual(report["overview"]["integrity_status"], "warnings")

    def test_structurally_damaged_nested_records_warn_without_crashing(self):
        game_id = "game_18_nested_damage"
        rows = [
            {
                "type": "config", "game_id": game_id,
                "event_schema_version": 2, "role_map": [], "role_models": [],
            },
            {
                "type": "llm_call", "call_id": ["broken"], "attempt": [],
                "api_attempted": True, "usage": "broken", "cost": [],
                "requested_model": ["bad"], "resolved_model": ["bad"],
            },
            {
                "type": "llm_call", "call_id": "call-negative", "attempt": 1,
                "api_attempted": True, "usage": {"total_tokens": -3},
                "cost": {"source": "provider_reported", "usd": -1.0},
                "requested_model": "model", "resolved_model": "model",
            },
            {"type": "event", "event": {
                "id": 0, "event_id": "evt_000000", "round": 1,
                "phase": "day_discuss", "type": "message", "channel": "public",
                "speaker_id": 0, "source_call_id": ["broken"], "payload": [],
            }},
            {"type": "usage_summary", "usage": {
                "calls": 2, "tokens": {"total_tokens": -1},
                "cost_usd_total": -2.0,
            }},
            {"type": "outcome", "winner": "wolf", "rounds": 1},
        ]
        report = build_full_report_from_file(self.write(rows, game_id=game_id))
        codes = {warning["code"] for warning in report["source"]["warnings"]}
        self.assertTrue({
            "malformed_role_map", "malformed_role_models",
            "malformed_llm_usage", "malformed_llm_cost",
            "invalid_llm_tokens", "invalid_llm_cost",
            "malformed_event_payload", "invalid_terminal_tokens",
            "invalid_terminal_cost",
        } <= codes)
        self.assertEqual(report["overview"]["integrity_status"], "warnings")
        self.assertEqual(report["usage"]["attempts"], 2)
        self.assertIsNone(report["usage"]["known_cost_usd"])
        self.assertEqual(report["usage"]["calls_without_known_cost"], 2)
        self.assertEqual(report["usage"]["tokens"]["total_tokens"], 0)
        self.assertEqual(
            report["usage"]["terminal_consistency"]["status"], "mismatched",
        )

    def test_unreported_optional_tokens_do_not_warn(self):
        rows = fixture_rows("game_19_null_tokens")
        rows[1]["usage"]["cached_input_tokens"] = None
        rows[1]["usage"]["reasoning_tokens"] = None
        rows[3]["usage"]["tokens"]["cached_input_tokens"] = None
        rows[3]["usage"]["tokens"]["reasoning_tokens"] = None
        report = build_full_report_from_file(self.write(
            rows, game_id="game_19_null_tokens",
        ))
        codes = {warning["code"] for warning in report["source"]["warnings"]}
        self.assertNotIn("invalid_llm_tokens", codes)
        self.assertNotIn("invalid_terminal_tokens", codes)
        self.assertEqual(
            report["usage"]["terminal_consistency"]["status"], "matched",
        )
        self.assertEqual(report["overview"]["usage_reliability"], "reliable")


if __name__ == "__main__":
    unittest.main()
