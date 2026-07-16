import json
import unittest

from werewolf.experiments.aggregate import (
    _ece_bins,
    analyze_v1,
    cluster_bootstrap,
    derive_rng_seed,
    extract_game_evidence,
    mean_statistic,
    ratio_statistic,
)
from werewolf.experiments.journal import replay
from werewolf.experiments.runner import build_experiment_manifest
from werewolf.experiments.snapshot import GameSource

TWO_CONDITIONS = {
    "cond_a": {"role_models": {
        "werewolf": "fast", "villager": "gemini_flash_lite",
        "seer": "gemini_flash_lite",
    }},
    "cond_b": {"role_models": {
        "werewolf": "gemini_flash_lite", "villager": "fast",
        "seer": "fast",
    }},
}

COMPARISON = {
    "comparison_id": "a_vs_b_clean_win_rate",
    "condition_a": "cond_a",
    "condition_b": "cond_b",
    "metric_id": "village_win_rate",
    "analysis_view": "clean_eligible",
    "design": "paired",
    "effect": "difference",
    "direction": "a_minus_b",
}


def snapshot_event(round_num, speaker, checkpoint, wolf_probabilities,
                   suspicion=None):
    return {"type": "event", "event": {
        "type": "belief_snapshot",
        "round": round_num,
        "speaker_id": speaker,
        "payload": {
            "valid": True,
            "checkpoint": checkpoint,
            "wolf_probabilities": {
                str(k): v for k, v in wolf_probabilities.items()
            },
            "intended_vote": None,
            "estimated_suspicion_of_me": (
                {str(k): v for k, v in suspicion.items()}
                if suspicion else None
            ),
        },
    }}


def vote_event(round_num, voter, target):
    return {"type": "event", "event": {
        "type": "vote", "round": round_num,
        "payload": {"voter_id": voter, "target_id": target},
    }}


def call_rows(call_id, action, *, attempts=1, fallback=False,
              repaired=False, usd=0.001, latency_ms=100):
    rows = []
    for attempt in range(1, attempts + 1):
        final = attempt == attempts
        ok = final and not fallback
        rows.append({
            "type": "llm_call",
            "call_id": call_id,
            "attempt": attempt,
            "api_attempted": True,
            "required_action": action,
            "error_category": (
                "completed" if ok
                else "provider_error" if not final or not fallback
                else "provider_error"
            ),
            "parse_method": (
                ("repaired" if repaired else "direct") if ok else None
            ),
            "validation_ok": ok or None,
            "cost": {"source": "provider_reported", "usd": usd,
                     "ticks": int(usd * 10_000_000_000)},
            "usage": {"total_tokens": 100, "input_tokens": 80,
                      "output_tokens": 20},
            "latency_ms": latency_ms,
        })
    if fallback:
        rows.append({
            "type": "llm_call",
            "call_id": call_id,
            "attempt": attempts,
            "api_attempted": False,
            "required_action": action,
            "error_category": "fallback_used",
            "cost": {"source": "unavailable", "usd": None, "ticks": None},
            "usage": {},
            "latency_ms": None,
        })
    return rows


def game_rows(game_id, seed, *, winner="village", fallback=False,
              beliefs=True, calls_usd=0.001):
    """4 players: 0 is the wolf, 1-3 are villagers."""
    role_map = {
        "0": {"role": "werewolf", "team": "wolf"},
        "1": {"role": "villager", "team": "village"},
        "2": {"role": "villager", "team": "village"},
        "3": {"role": "villager", "team": "village"},
    }
    rows = [{
        "type": "config",
        "game_id": game_id,
        "seed": seed,
        "n_players": 4, "n_wolves": 1, "n_seers": 0,
        "event_schema_version": 1,
        "belief_snapshots": beliefs,
        "role_map": role_map,
        "role_models": {
            "werewolf": {"requested": "fast"},
            "villager": {"requested": "gemini_flash_lite"},
        },
    }]
    rows += call_rows(f"{game_id}_c1", "vote", usd=calls_usd)
    rows += call_rows(f"{game_id}_c2", "speak_public", attempts=2,
                      usd=calls_usd)
    rows += call_rows(f"{game_id}_c3", "vote", repaired=True, usd=calls_usd)
    if fallback:
        rows += call_rows(f"{game_id}_c4", "vote", attempts=3,
                          fallback=True, usd=calls_usd)
    if beliefs:
        # P1: pre top-suspect wolf (correct), post stays correct, votes 0.
        rows.append(snapshot_event(1, 1, "pre_discussion",
                                   {0: 0.8, 2: 0.2, 3: 0.1}))
        rows.append(snapshot_event(1, 1, "post_discussion",
                                   {0: 0.9, 2: 0.2, 3: 0.1}))
        # P2: pre top-suspect villager (wrong), post revises to the wolf.
        rows.append(snapshot_event(1, 2, "pre_discussion",
                                   {0: 0.2, 1: 0.6, 3: 0.1}))
        rows.append(snapshot_event(1, 2, "post_discussion",
                                   {0: 0.7, 1: 0.3, 3: 0.1}))
        # Wolf estimates suspicion of itself.
        rows.append(snapshot_event(1, 0, "post_discussion",
                                   {1: 0.3, 2: 0.4, 3: 0.2},
                                   suspicion={1: 0.8, 2: 0.7}))
        rows.append(vote_event(1, 1, 0))
        rows.append(vote_event(1, 2, 3))  # misaligned with top suspect
    remaining = [1, 2, 3] if winner == "village" else [0]
    rows.append({"type": "usage_summary", "usage": {}})
    rows.append({"type": "outcome", "winner": winner, "rounds": 1,
                 "remaining": remaining})
    return rows


def make_source(condition_id, seed, repetition=0, *, winner="village",
                record_type="trial_completed", verified=True,
                fallback=False, beliefs=True, usd=0.001):
    game_id = f"game_{seed}_{condition_id}_{repetition}"
    trial_id = f"trial_{condition_id}_{seed}_{repetition}"
    rows = game_rows(game_id, seed, winner=winner, fallback=fallback,
                     beliefs=beliefs, calls_usd=usd)
    terminal = {
        "record_type": record_type,
        "trial_id": trial_id,
        "attempt_id": f"{trial_id}_a1",
        "attempt_number": 1,
        "condition_id": condition_id,
        "seed": seed,
        "repetition": repetition,
        "game_id": game_id,
        "winner": winner if record_type == "trial_completed" else None,
        "rounds": 1,
    }
    return GameSource(
        trial_id=trial_id,
        attempt_id=f"{trial_id}_a1",
        attempt_number=1,
        record_type=record_type,
        condition_id=condition_id,
        seed=seed,
        repetition=repetition,
        game_id=game_id,
        recorded_game_sha256="a" * 64 if verified else "b" * 64,
        observed_game_sha256="a" * 64,
        source_status=(
            "verified" if verified else "source_modified_after_completion"
        ),
        terminal_record=terminal,
        data=b"".join(
            json.dumps(row, sort_keys=True).encode("utf-8") + b"\n"
            for row in rows
        ) if verified else None,
        rows=rows if verified else None,
    )


def make_manifest(comparisons=None, seeds=(1, 2, 3, 4, 5, 6)):
    return build_experiment_manifest(
        experiment_id="agg",
        conditions=TWO_CONDITIONS,
        seeds=list(seeds),
        repetitions=1,
        game={"n_players": 4, "n_wolves": 1, "n_seers": 0},
        comparisons=comparisons or [COMPARISON],
    )


def run_analysis(sources, manifest=None):
    manifest = manifest or make_manifest()
    records = []
    for source in sources:
        base = {
            "trial_id": source.trial_id,
            "attempt_id": source.attempt_id,
            "attempt_number": 1,
            "trial_index": 0,
            "scheduler_position": 0,
            "condition_id": source.condition_id,
            "seed": source.seed,
            "repetition": source.repetition,
            "game_id": source.game_id,
        }
        records.append({"record_type": "trial_started", **base})
        records.append({
            "record_type": source.record_type, **base,
            "recorded_game_sha256": source.recorded_game_sha256,
            "source_status": "recorded",
        })
    return analyze_v1(
        manifest=manifest,
        analysis_contract=manifest["analysis_contract"],
        sources=sources,
        lifecycle_records=[],
        replay_state=replay(records),
    )


class BootstrapTests(unittest.TestCase):
    def test_deterministic_and_cluster_resampled(self):
        obs = {seed: [float(seed % 2)] * 2 for seed in range(10)}
        a = cluster_bootstrap(obs, mean_statistic, n_boot=200,
                              alpha=0.05, rng_seed=42)
        b = cluster_bootstrap(obs, mean_statistic, n_boot=200,
                              alpha=0.05, rng_seed=42)
        self.assertEqual(a, b)
        self.assertEqual(a["estimate"], 0.5)
        self.assertEqual(a["n_seeds"], 10)
        self.assertEqual(a["interval_status"], "ok")
        self.assertLessEqual(a["ci_low"], 0.5)
        self.assertGreaterEqual(a["ci_high"], 0.5)

    def test_sparse_clusters_report_estimate_without_interval(self):
        obs = {1: [1.0], 2: [0.0], 3: [1.0], 4: [0.0]}
        result = cluster_bootstrap(obs, mean_statistic, n_boot=200,
                                   alpha=0.05, rng_seed=0)
        self.assertEqual(result["estimate"], 0.5)
        self.assertEqual(result["interval_status"],
                         "insufficient_clusters")
        self.assertIsNone(result["ci_low"])

    def test_empty_data_returns_none(self):
        self.assertIsNone(cluster_bootstrap(
            {}, mean_statistic, n_boot=10, alpha=0.05, rng_seed=0,
        ))
        self.assertIsNone(ratio_statistic([(0, 0)]))

    def test_rng_seed_derivation_is_label_sensitive(self):
        self.assertNotEqual(derive_rng_seed(0, "a"), derive_rng_seed(0, "b"))
        self.assertEqual(derive_rng_seed(0, "a"), derive_rng_seed(0, "a"))


class EceTests(unittest.TestCase):
    def test_bins_and_boundaries(self):
        predictions = [
            (0.0, False, "g1", 1),    # bin 0
            (0.05, False, "g1", 1),   # bin 0
            (0.1, False, "g1", 1),    # bin 1 (left-closed)
            (0.95, True, "g2", 2),    # bin 9
            (1.0, True, "g2", 2),     # bin 9 (right-closed last bin)
        ]
        bins, ece = _ece_bins(predictions)
        self.assertEqual(len(bins), 10)
        self.assertEqual(bins[0]["prediction_count"], 2)
        self.assertEqual(bins[1]["prediction_count"], 1)
        self.assertEqual(bins[9]["prediction_count"], 2)
        self.assertEqual(bins[9]["empirical_frequency"], 1.0)
        self.assertEqual(bins[9]["game_count"], 1)
        self.assertEqual(bins[9]["seed_count"], 1)
        self.assertIsNone(bins[5]["mean_confidence"])  # empty bin
        self.assertIsNotNone(ece)

    def test_no_predictions_yields_null(self):
        bins, ece = _ece_bins([])
        self.assertIsNone(ece)
        self.assertTrue(all(b["prediction_count"] == 0 for b in bins))


class EvidenceExtractionTests(unittest.TestCase):
    def test_decision_group_accounting(self):
        source = make_source("cond_a", 1, fallback=True)
        evidence = extract_game_evidence(source)
        groups = evidence["decision_groups"]
        self.assertEqual(len(groups), 4)
        self.assertEqual(
            sum(1 for g in groups if g["ended_in_fallback"]), 1,
        )
        self.assertEqual(sum(1 for g in groups if g["api_attempts"] > 1), 2)
        self.assertEqual(
            sum(1 for g in groups if g["repaired"]), 1,
        )
        self.assertFalse(evidence["clean"])

    def test_belief_observation_extraction(self):
        evidence = extract_game_evidence(make_source("cond_a", 1))
        belief = evidence["belief"]
        self.assertEqual(belief["initially_correct"], 1)
        self.assertEqual(belief["harmful"], 0)
        self.assertEqual(belief["initially_wrong"], 1)
        self.assertEqual(belief["beneficial"], 1)
        self.assertEqual(belief["alignment_n"], 2)
        self.assertEqual(belief["aligned"], 1)
        self.assertEqual(len(belief["awareness_errors"]), 2)
        # movement toward wolf: P1 +0.1, P2 +0.5
        movement = sorted(belief["movement"])
        self.assertEqual(len(movement), 2)
        self.assertAlmostEqual(movement[0], 0.1)
        self.assertAlmostEqual(movement[1], 0.5)

    def test_usage_extraction(self):
        evidence = extract_game_evidence(make_source("cond_a", 1))
        usage = evidence["usage"]
        self.assertEqual(usage["api_calls"], 4)
        self.assertAlmostEqual(usage["cost_usd"], 0.004)
        self.assertTrue(usage["cost_complete"])
        self.assertEqual(usage["tokens"]["total_tokens"], 400)
        self.assertEqual(len(usage["latencies"]), 4)

    def test_usage_extraction_reuses_forensic_value_validation(self):
        source = make_source("cond_a", 1)
        call = next(row for row in source.rows
                    if row.get("type") == "llm_call")
        call["cost"] = {"source": "provider_reported", "usd": -2.0}
        call["usage"] = {"total_tokens": -500}
        source.data = b"".join(
            json.dumps(row, sort_keys=True).encode("utf-8") + b"\n"
            for row in source.rows
        )
        usage = extract_game_evidence(source)["usage"]
        self.assertAlmostEqual(usage["cost_usd"], 0.003)
        self.assertFalse(usage["cost_complete"])
        self.assertEqual(usage["calls_with_unavailable_cost"], 1)
        self.assertEqual(usage["tokens"]["total_tokens"], 300)

    def test_clean_requires_pr2_strategic_eligibility(self):
        source = make_source("cond_a", 1)
        config = source.rows[0]
        config["event_schema_version"] = 3
        # Version 3 provenance requires linked events. This synthetic game
        # remains validity-clean but must not enter clean_eligible metrics.
        source.data = b"".join(
            json.dumps(row, sort_keys=True).encode("utf-8") + b"\n"
            for row in source.rows
        )
        evidence = extract_game_evidence(source)
        self.assertTrue(evidence["clean"])
        self.assertEqual(evidence["analysis_eligibility"], "ineligible")


class AnalyzeV1Tests(unittest.TestCase):
    def _sources(self):
        sources = []
        for seed in (1, 2, 3, 4, 5, 6):
            # cond_a: village wins except seed 6; cond_b: village wins
            # only on even seeds.
            sources.append(make_source(
                "cond_a", seed,
                winner="village" if seed != 6 else "wolf",
            ))
            sources.append(make_source(
                "cond_b", seed,
                winner="village" if seed % 2 == 0 else "wolf",
            ))
        return sources

    def test_views_overlap_and_membership(self):
        sources = self._sources()
        # one dirty game (fallback) in cond_a
        sources[0] = make_source("cond_a", 1, fallback=True)
        analysis = run_analysis(sources)
        views = analysis["views"]
        all_completed = views["all_completed"]["overall"]
        clean = views["clean_eligible"]["overall"]
        not_clean = views["completed_not_clean_eligible"]["overall"]
        self.assertEqual(all_completed["games"], 12)
        self.assertEqual(clean["games"], 11)
        self.assertEqual(not_clean["games"], 1)
        self.assertEqual(
            all_completed["clean_game_rate"]["numerator"], 11,
        )
        self.assertEqual(
            all_completed["fallback_game_rate"]["numerator"], 1,
        )

    def test_win_rates_have_exact_denominators(self):
        analysis = run_analysis(self._sources())
        cond_a = (analysis["views"]["all_completed"]["per_condition"]
                  ["cond_a"])
        self.assertEqual(cond_a["village_win_rate"]["numerator"], 5)
        self.assertEqual(cond_a["village_win_rate"]["denominator"], 6)
        self.assertAlmostEqual(
            cond_a["village_win_rate"]["estimate"], 5 / 6,
        )
        self.assertEqual(cond_a["village_win_rate"]["n_seeds"], 6)
        self.assertEqual(cond_a["village_win_rate"]["interval_status"],
                         "ok")

    def test_pooled_group_and_belief_rates(self):
        analysis = run_analysis(self._sources())
        overall = analysis["views"]["all_completed"]["overall"]
        # 3 groups per game, 12 games, none fallback
        self.assertEqual(
            overall["fallback_decision_group_rate"]["denominator"], 36,
        )
        self.assertEqual(overall["retry_rate"]["numerator"], 12)
        self.assertEqual(overall["repair_rate"]["numerator"], 12)
        self.assertEqual(overall["repair_rate"]["denominator"], 36)
        belief = overall["harmful_revision"]
        self.assertEqual(belief["estimate"], 0.0)
        self.assertEqual(belief["observations"], 12)
        self.assertEqual(belief["games"], 12)
        self.assertEqual(belief["seeds"], 6)
        retention = overall["correct_belief_retention"]
        self.assertEqual(retention["estimate"], 1.0)
        self.assertEqual(overall["initial_correctness"]["estimate"], 0.5)
        self.assertEqual(overall["beneficial_revision"]["estimate"], 1.0)
        alignment = overall["vote_belief_alignment"]
        self.assertEqual(alignment["estimate"], 0.5)
        self.assertEqual(alignment["eligible_votes"], 24)

    def test_calibration_reports_counts_and_bins(self):
        analysis = run_analysis(self._sources())
        overall = analysis["views"]["all_completed"]["overall"]
        brier = overall["brier_post_discussion"]
        self.assertIsNotNone(brier["estimate"])
        self.assertEqual(brier["observations"], 12 * 6)
        self.assertEqual(brier["games"], 12)
        self.assertEqual(brier["seeds"], 6)
        ece = overall["ece_post_discussion"]
        self.assertEqual(ece["prediction_count"], 72)
        self.assertEqual(len(ece["bins"]), 10)
        self.assertIsNotNone(ece["estimate"])
        self.assertEqual(ece["interval_status"], "ok")

    def test_cost_tokens_latency(self):
        analysis = run_analysis(self._sources())
        overall = analysis["views"]["all_completed"]["overall"]
        self.assertAlmostEqual(overall["cost"]["total_usd"], 0.004 * 12)
        self.assertAlmostEqual(
            overall["cost"]["cost_per_game_usd"], 0.004,
        )
        self.assertTrue(overall["cost"]["cost_complete"])
        latency = overall["latency"]
        self.assertEqual(latency["calls_with_latency"], 48)
        self.assertEqual(latency["total_attempted_calls"], 48)
        self.assertEqual(latency["coverage_fraction"], 1.0)
        self.assertEqual(latency["median_ms"], 100)
        self.assertEqual(latency["p90_ms"], 100)

    def test_modified_sources_are_ineligible_but_visible(self):
        sources = self._sources()
        sources[3] = make_source(
            "cond_b", 2, winner="village", verified=False,
        )
        analysis = run_analysis(sources)
        self.assertEqual(
            analysis["views"]["all_completed"]["overall"]["games"], 11,
        )
        ineligible = analysis["analytically_ineligible"]
        self.assertEqual(len(ineligible), 1)
        self.assertEqual(ineligible[0]["source_status"],
                         "source_modified_after_completion")
        self.assertEqual(
            analysis["operational"]["cost"]
            ["sources_excluded_from_totals"], 1,
        )

    def test_paired_comparison(self):
        analysis = run_analysis(self._sources())
        comparison = analysis["comparisons"][0]
        self.assertEqual(comparison["status"], "ok")
        self.assertEqual(comparison["n_seeds"], 6)
        # a: 5/6 village wins; b: 3/6
        self.assertAlmostEqual(comparison["estimate"], 5 / 6 - 3 / 6)
        self.assertEqual(comparison["excluded_pairs"], {})
        self.assertIn("not a significance declaration",
                      comparison["note"])

    def test_incomplete_pairs_are_excluded_and_reported(self):
        sources = [s for s in self._sources()
                   if not (s.condition_id == "cond_b" and s.seed == 6)]
        analysis = run_analysis(sources)
        comparison = analysis["comparisons"][0]
        self.assertEqual(comparison["n_seeds"], 5)
        self.assertEqual(
            comparison["excluded_pairs"],
            {"missing_condition_observation": 1},
        )

    def test_partial_repetition_set_excludes_entire_seed(self):
        manifest = build_experiment_manifest(
            experiment_id="agg_repetitions",
            conditions=TWO_CONDITIONS,
            seeds=[1, 2, 3, 4, 5],
            repetitions=2,
            game={"n_players": 4, "n_wolves": 1, "n_seers": 0},
            comparisons=[COMPARISON],
        )
        sources = []
        for seed in (1, 2, 3, 4, 5):
            sources.append(make_source("cond_a", seed, repetition=0))
            sources.append(make_source("cond_a", seed, repetition=1))
            sources.append(make_source("cond_b", seed, repetition=0))
            if seed != 5:
                sources.append(make_source("cond_b", seed, repetition=1))
        comparison = run_analysis(sources, manifest)["comparisons"][0]
        self.assertEqual(comparison["n_seeds"], 4)
        self.assertEqual(
            comparison["excluded_pairs"], {"incomplete_repetitions": 1},
        )

    def test_zero_shared_seeds_yields_null(self):
        sources = [make_source("cond_a", 1), make_source("cond_b", 2)]
        analysis = run_analysis(sources)
        comparison = analysis["comparisons"][0]
        self.assertEqual(comparison["status"], "no_shared_paired_seeds")
        self.assertIsNone(comparison["estimate"])

    def test_operational_attempt_accounting(self):
        analysis = run_analysis(self._sources())
        operational = analysis["operational"]
        self.assertEqual(operational["attempts"]["trial_completed"], 12)
        self.assertEqual(operational["attempts"]["total"], 12)
        self.assertEqual(operational["completed_trials"], 12)
        self.assertAlmostEqual(
            operational["cost"]["by_record_type_usd"]["trial_completed"],
            0.048,
        )

    def test_failed_work_cost_is_included(self):
        sources = self._sources()
        sources.append(make_source(
            "cond_a", 1, repetition=1, record_type="trial_failed",
            usd=0.01,
        ))
        analysis = run_analysis(sources)
        operational = analysis["operational"]
        self.assertEqual(operational["attempts"]["trial_failed"], 1)
        self.assertAlmostEqual(
            operational["cost"]["by_record_type_usd"]["trial_failed"],
            0.04,  # 4 attempted calls at $0.01 in the failed attempt
        )
        # failed work never enters the analysis views
        self.assertEqual(
            analysis["views"]["all_completed"]["overall"]["games"], 12,
        )

    def test_no_composite_score_is_exposed(self):
        analysis = run_analysis(self._sources())
        text = str(analysis).lower()
        self.assertNotIn("composite_score", text)
        self.assertNotIn("universal_score", text)
        self.assertNotIn("overall_model_score", text)


if __name__ == "__main__":
    unittest.main()
