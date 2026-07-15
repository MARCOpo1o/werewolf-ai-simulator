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
        "player_role": "villager", "phase": "day_discuss",
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


if __name__ == "__main__":
    unittest.main()
