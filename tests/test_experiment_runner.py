import json
import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from werewolf.experiments.health import probe_model, unique_health_targets
from werewolf.engine.game import GameEngine
from werewolf.experiments.journal import (
    JournalWriter,
    TRIAL_COMPLETED,
    TRIAL_STARTED,
    read_journal,
)
from werewolf.experiments.manifest import (
    ManifestError,
    execution_contract_sha256,
    games_dir,
    journal_path,
    manifest_is_frozen,
    write_manifest,
    finalize_manifest,
)
from werewolf.experiments.runner import (
    ExecutionRuntimeChanged,
    ExperimentRunError,
    _default_engine_factory,
    build_experiment_manifest,
    default_crossed_comparisons,
    run_experiment,
    validate_manifest_for_execution,
)
from werewolf.llm.fake_provider import (
    FakeProvider,
    error_result,
    success_result,
)
from werewolf.llm.records import ErrorCategory

CONDITIONS = {
    "solo": {"role_models": {
        "werewolf": "fast", "villager": "fast", "seer": "fast",
    }},
}
OFFLINE_POLICIES = {
    "allow_provider_fallback": True,
    "action_failure_policy": "fallback",
}


def make_manifest(experiment_id="exp1", seeds=(9001, 9002), **overrides):
    kwargs = dict(
        experiment_id=experiment_id,
        conditions=CONDITIONS,
        seeds=list(seeds),
        repetitions=1,
        policies=OFFLINE_POLICIES,
        game={"belief_snapshots": False},
    )
    kwargs.update(overrides)
    return build_experiment_manifest(**kwargs)


def ready_prober(target):
    return probe_model(target, provider=FakeProvider([success_result(
        {"health": "ok"}, resolved_model=target["requested_model"],
    )]))


def failed_prober(target):
    return probe_model(target, provider=FakeProvider([error_result(
        ErrorCategory.PROVIDER_ERROR, cost_ticks=7_000_000,
    )]))


def adjusted_prober(target):
    result = success_result(
        {"health": "ok"}, resolved_model=target["requested_model"],
    )
    result.provider_metadata = {"generation_dropped": ["provider_seed"]}
    return probe_model(target, provider=FakeProvider([result]))


class BoomEngine:
    def __init__(self, game_id: str):
        self.state = SimpleNamespace(game_id=game_id)

    def run(self):
        raise RuntimeError("simulated provider explosion")

    def close(self):
        pass


def boom_factory(entry, manifest, games_directory):
    return BoomEngine(f"game_boom_{entry['trial_id'][:12]}")


def offline_engine_factory(entry, manifest, games_directory):
    """Explicit no-provider factory for tests, even with local .env keys."""
    execution = manifest["execution_contract"]
    game = execution["game"]
    policies = execution["policies"]
    roles = execution["conditions"][entry["condition_id"]]["role_models"]
    return GameEngine(
        n_players=game["n_players"], n_wolves=game["n_wolves"],
        n_seers=game["n_seers"], seed=entry["seed"],
        output_dir=str(games_directory), api_key="", model=roles["villager"],
        role_models=roles,
        role_providers={role: None for role in roles},
        allow_provider_fallback=True,
        action_failure_policy=policies["action_failure_policy"],
        max_rounds=policies["max_rounds"],
        agent_action_max_attempts=policies["agent_action_max_attempts"],
        retryable_error_categories=policies["retryable_errors"],
        retry_backoff=policies["retry_backoff"],
        request_timeout_seconds=policies["request_timeout_seconds"],
        transcript_enabled=False, show_all_channels=False,
        belief_snapshots=game["belief_snapshots"],
        discussion_cycles=game["discussion_cycles"],
        batch_id=f"fixture/{entry['condition_id']}",
        trial_index=entry["trial_index"],
    )


def quiet(*args, **kwargs):
    pass


def run(tmp, experiment_id="exp1", **kwargs):
    kwargs.setdefault("health_prober", ready_prober)
    kwargs.setdefault("progress", quiet)
    return run_experiment(tmp, experiment_id, **kwargs)


class RunnerHappyPathTests(unittest.TestCase):
    def test_fresh_run_completes_all_trials(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_manifest(tmp, make_manifest())
            counts = run(tmp)
            self.assertEqual(counts["completed"], 2)
            self.assertEqual(counts["failed"], 0)
            snapshot = read_journal(journal_path(tmp, "exp1"))
            types = [r["record_type"] for r in snapshot.records]
            self.assertEqual(types.count("execution_session_started"), 1)
            self.assertEqual(types.count("health_check"), 1)
            self.assertEqual(types.count(TRIAL_STARTED), 2)
            self.assertEqual(types.count(TRIAL_COMPLETED), 2)
            self.assertEqual(types.count("execution_session_finished"), 1)
            self.assertTrue(manifest_is_frozen(tmp, "exp1"))

    def test_completed_attempts_store_verified_source_hashes(self):
        import hashlib
        with tempfile.TemporaryDirectory() as tmp:
            write_manifest(tmp, make_manifest())
            run(tmp)
            snapshot = read_journal(journal_path(tmp, "exp1"))
            completed = [r for r in snapshot.records
                         if r["record_type"] == TRIAL_COMPLETED]
            self.assertEqual(len(completed), 2)
            for record in completed:
                log = games_dir(tmp, "exp1") / f"{record['game_id']}.jsonl"
                self.assertEqual(
                    record["recorded_game_sha256"],
                    hashlib.sha256(log.read_bytes()).hexdigest(),
                )
                self.assertIn(record["winner"], ("wolf", "village"))
                self.assertTrue(record["verifier"]["checks"]
                                ["victory_predicate"])

    def test_second_run_requires_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_manifest(tmp, make_manifest())
            run(tmp)
            with self.assertRaises(ExperimentRunError):
                run(tmp)
            counts = run(tmp, resume=True)
            self.assertEqual(counts["completed"], 0)
            self.assertEqual(counts["skipped_exhausted"], 0)

    def test_noop_resume_appends_no_session_or_health_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_manifest(tmp, make_manifest())
            run(tmp)
            before = len(read_journal(journal_path(tmp, "exp1")).records)
            run(tmp, resume=True)
            after = len(read_journal(journal_path(tmp, "exp1")).records)
            self.assertEqual(before, after)


class RunnerRecoveryTests(unittest.TestCase):
    def _open_attempt(self, tmp, manifest, entry, *, with_log: bool):
        """Simulate a crash: trial_started journaled, no terminal."""
        directory = games_dir(tmp, manifest["experiment_id"])
        directory.mkdir(parents=True, exist_ok=True)
        if with_log:
            engine = _default_engine_factory(entry, manifest, directory)
            engine.run()
            game_id = engine.state.game_id
        else:
            game_id = "game_vanished_1_aa"
        writer = JournalWriter(
            journal_path(tmp, manifest["experiment_id"]),
            manifest_content_sha256=manifest["manifest_content_sha256"],
            execution_contract_sha256=execution_contract_sha256(manifest),
        )
        writer.append(TRIAL_STARTED, {
            "trial_id": entry["trial_id"],
            "attempt_id": f"{entry['trial_id']}_a1",
            "attempt_number": 1,
            "trial_index": entry["trial_index"],
            "scheduler_position": entry["scheduler_position"],
            "condition_id": entry["condition_id"],
            "seed": entry["seed"],
            "repetition": entry["repetition"],
            "game_id": game_id,
        })
        return game_id

    def test_completed_paid_game_is_recovered_exactly_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = make_manifest()
            write_manifest(tmp, manifest)
            entry = manifest["execution_contract"]["schedule"][0]
            self._open_attempt(tmp, manifest, entry, with_log=True)

            counts = run(tmp, resume=True)
            self.assertEqual(counts["recovered"], 1)
            self.assertEqual(counts["completed"], 1)  # the other seed
            snapshot = read_journal(journal_path(tmp, "exp1"))
            recovered = [r for r in snapshot.records
                         if r["record_type"] == TRIAL_COMPLETED
                         and r.get("recovered")]
            self.assertEqual(len(recovered), 1)
            self.assertEqual(recovered[0]["trial_id"], entry["trial_id"])

            counts = run(tmp, resume=True)
            self.assertEqual(counts["recovered"], 0)
            self.assertEqual(counts["completed"], 0)

    def test_missing_game_log_interrupts_then_reruns(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = make_manifest(seeds=(9001,))
            write_manifest(tmp, manifest)
            entry = manifest["execution_contract"]["schedule"][0]
            self._open_attempt(tmp, manifest, entry, with_log=False)

            counts = run(tmp, resume=True)
            self.assertEqual(counts["reconciled_interrupted"], 1)
            self.assertEqual(counts["completed"], 1)
            snapshot = read_journal(journal_path(tmp, "exp1"))
            interrupted = [r for r in snapshot.records
                           if r["record_type"] == "trial_interrupted"]
            self.assertEqual(len(interrupted), 1)
            self.assertIsNone(interrupted[0]["recorded_game_sha256"])
            self.assertEqual(interrupted[0]["source_status"],
                             "missing_game_log")
            trial = [r for r in snapshot.records
                     if r["record_type"] == TRIAL_COMPLETED][0]
            self.assertEqual(trial["attempt_number"], 2)

    def test_incomplete_game_log_is_hashed_and_interrupted(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = make_manifest(seeds=(9001,))
            write_manifest(tmp, manifest)
            entry = manifest["execution_contract"]["schedule"][0]
            game_id = self._open_attempt(tmp, manifest, entry, with_log=True)
            log = games_dir(tmp, "exp1") / f"{game_id}.jsonl"
            lines = log.read_text(encoding="utf-8").splitlines()
            log.write_text(
                "\n".join(l for l in lines if '"outcome"' not in l) + "\n",
                encoding="utf-8",
            )
            counts = run(tmp, resume=True)
            self.assertEqual(counts["reconciled_interrupted"], 1)
            snapshot = read_journal(journal_path(tmp, "exp1"))
            interrupted = [r for r in snapshot.records
                           if r["record_type"] == "trial_interrupted"][0]
            self.assertEqual(interrupted["source_status"], "recorded")
            self.assertIsNotNone(interrupted["recorded_game_sha256"])

    def test_crashed_explicit_failure_does_not_auto_rerun(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = make_manifest(seeds=(9001,))
            write_manifest(tmp, manifest)
            entry = manifest["execution_contract"]["schedule"][0]
            directory = games_dir(tmp, "exp1")
            directory.mkdir(parents=True, exist_ok=True)
            engine = _default_engine_factory(entry, manifest, directory)
            game_id = engine.state.game_id
            writer = JournalWriter(
                journal_path(tmp, "exp1"),
                manifest_content_sha256=manifest["manifest_content_sha256"],
                execution_contract_sha256=execution_contract_sha256(manifest),
            )
            writer.append(TRIAL_STARTED, {
                "trial_id": entry["trial_id"],
                "attempt_id": f"{entry['trial_id']}_a1",
                "attempt_number": 1,
                "trial_index": entry["trial_index"],
                "scheduler_position": entry["scheduler_position"],
                "condition_id": entry["condition_id"],
                "seed": entry["seed"],
                "repetition": entry["repetition"],
                "game_id": game_id,
            })
            engine.abort("ActionFailureAbort")
            engine.close()

            counts = run(tmp, resume=True)
            self.assertEqual(counts["reconciled_failed"], 1)
            self.assertEqual(counts["completed"], 0)
            self.assertEqual(counts["skipped_exhausted"], 1)
            records = read_journal(journal_path(tmp, "exp1")).records
            failures = [r for r in records
                        if r["record_type"] == "trial_failed"]
            self.assertEqual(len(failures), 1)
            self.assertTrue(failures[0]["recovered"])


class RunnerRetryTests(unittest.TestCase):
    def test_failures_require_retry_failed_even_before_attempt_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_manifest(tmp, make_manifest(seeds=(9001,)))
            counts = run(tmp, engine_factory=boom_factory)
            self.assertEqual(counts["failed"], 1)
            # An explicit failure is not retried by ordinary resume.
            counts = run(tmp, resume=True, engine_factory=boom_factory)
            self.assertEqual(counts["skipped_exhausted"], 1)
            self.assertEqual(counts["completed"], 0)
            # --retry-failed grants a deliberate second attempt.
            counts = run(tmp, retry_failed=True, engine_factory=boom_factory)
            self.assertEqual(counts["failed"], 1)
            # It remains opt-in even after exhausting the normal budget.
            counts = run(tmp, resume=True)
            self.assertEqual(counts["skipped_exhausted"], 1)
            # A subsequent --retry-failed attempt can use a repaired setup.
            counts = run(tmp, retry_failed=True)
            self.assertEqual(counts["completed"], 1)
            snapshot = read_journal(journal_path(tmp, "exp1"))
            completed = [r for r in snapshot.records
                         if r["record_type"] == TRIAL_COMPLETED][0]
            self.assertEqual(completed["attempt_number"], 3)

    def test_failed_attempts_record_sanitized_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_manifest(tmp, make_manifest(seeds=(9001,)))
            run(tmp, engine_factory=boom_factory)
            snapshot = read_journal(journal_path(tmp, "exp1"))
            failed = [r for r in snapshot.records
                      if r["record_type"] == "trial_failed"][0]
            self.assertIn("simulated provider explosion",
                          failed["sanitized_error"])
            self.assertEqual(failed["source_status"], "missing_game_log")

    def test_failed_real_engine_log_has_abort_and_usage_evidence(self):
        def limited_factory(entry, manifest, game_directory):
            engine = _default_engine_factory(entry, manifest, game_directory)
            engine.max_rounds = 1
            return engine

        with tempfile.TemporaryDirectory() as tmp:
            write_manifest(tmp, make_manifest(seeds=(9001,)))
            counts = run(tmp, engine_factory=limited_factory)
            self.assertEqual(counts["failed"], 1)
            failed = [record for record in read_journal(
                journal_path(tmp, "exp1"),
            ).records if record["record_type"] == "trial_failed"][0]
            log = games_dir(tmp, "exp1") / f"{failed['game_id']}.jsonl"
            rows = [json.loads(line) for line in log.read_text().splitlines()]
            self.assertEqual([row["type"] for row in rows].count("abort"), 1)
            self.assertEqual(
                [row["type"] for row in rows].count("usage_summary"), 1,
            )


class RunnerGateTests(unittest.TestCase):
    def test_zero_seer_run_skips_inactive_model_preflight(self):
        conditions = {
            "zero_seer": {"role_models": {
                "werewolf": "fast", "villager": "fast",
                "seer": "gemini_flash_lite",
            }},
        }
        seen = []

        def recording_prober(target):
            seen.append(target["model_alias"])
            return ready_prober(target)

        with tempfile.TemporaryDirectory() as tmp:
            write_manifest(tmp, make_manifest(
                seeds=(9001,), conditions=conditions,
                game={"belief_snapshots": False, "n_seers": 0},
            ))
            counts = run(
                tmp, health_prober=recording_prober,
                engine_factory=offline_engine_factory,
            )
            self.assertEqual(counts["completed"], 1)
            self.assertEqual(seen, ["fast"])

    def test_execution_runtime_change_blocks_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_manifest(tmp, make_manifest())
            with mock.patch(
                "werewolf.experiments.runner.execution_runtime_hash",
                return_value="0" * 64,
            ):
                with self.assertRaises(ExecutionRuntimeChanged):
                    run(tmp)

    def test_analysis_only_change_never_blocks_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch(
                "werewolf.experiments.runner.analysis_runtime_hash",
                return_value="f" * 64,
            ):
                manifest = make_manifest()
            self.assertEqual(
                manifest["analysis_contract"]["analysis_runtime_hash"],
                "f" * 64,
            )
            write_manifest(tmp, manifest)
            counts = run(tmp)  # real analysis hash differs; still runs
            self.assertEqual(counts["completed"], 2)

    def test_failed_health_blocks_and_preserves_cost(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_manifest(tmp, make_manifest())
            with self.assertRaises(ExperimentRunError):
                run(tmp, health_prober=failed_prober)
            snapshot = read_journal(journal_path(tmp, "exp1"))
            types = [r["record_type"] for r in snapshot.records]
            self.assertIn("health_check", types)
            self.assertIn("execution_session_aborted", types)
            self.assertNotIn(TRIAL_STARTED, types)
            health = [r for r in snapshot.records
                      if r["record_type"] == "health_check"][0]
            self.assertEqual(health["status"], "failed")
            self.assertIsNotNone(health["cost"])

    def test_undeclared_adjustment_blocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_manifest(tmp, make_manifest())
            with self.assertRaises(ExperimentRunError):
                run(tmp, health_prober=adjusted_prober,
                    allow_adjusted_health=True)

    def test_predeclared_adjustment_with_flag_runs(self):
        base_manifest = make_manifest()
        execution = base_manifest["execution_contract"]
        target = unique_health_targets(
            execution["conditions"], execution["generation"],
            request_timeout_seconds=(
                execution["policies"]["request_timeout_seconds"]
            ),
        )[0]
        target_probe = adjusted_prober(target)
        fingerprint = target_probe["adjustment_fingerprint"]
        with tempfile.TemporaryDirectory() as tmp:
            write_manifest(tmp, make_manifest(
                predeclared_adjustments=[{
                    "description": "provider drops provider_seed",
                    "fingerprint": fingerprint,
                }],
            ))
            with self.assertRaises(ExperimentRunError):
                run(tmp, health_prober=adjusted_prober)  # flag missing
            counts = run(tmp, health_prober=adjusted_prober,
                         allow_adjusted_health=True, resume=True)
            self.assertEqual(counts["completed"], 2)


class ManifestBuildTests(unittest.TestCase):
    def test_formal_defaults_are_pinned(self):
        manifest = make_manifest()
        policies = manifest["execution_contract"]["policies"]
        self.assertEqual(policies["max_trial_attempts"], 2)
        self.assertEqual(policies["max_rounds"], 20)
        self.assertEqual(policies["agent_action_max_attempts"], 3)
        self.assertEqual(policies["request_timeout_seconds"], 120)
        self.assertEqual(policies["retry_backoff"], "none")
        self.assertIn("malformed_json", policies["retryable_errors"])
        self.assertIn("invalid_game_action", policies["retryable_errors"])
        self.assertEqual(policies["public_message_limit"], 800)
        self.assertEqual(policies["memory_limit"], 2500)
        self.assertEqual(
            manifest["execution_contract"]["generation"]
            ["max_output_tokens"], 4096,
        )
        self.assertEqual(
            manifest["execution_contract"]["game"]["discussion_cycles"], 2,
        )
        self.assertEqual(validate_manifest_for_execution(manifest), [])

    def test_default_factory_applies_pinned_execution_policies(self):
        manifest = make_manifest(policies={
            **OFFLINE_POLICIES,
            "agent_action_max_attempts": 1,
            "retryable_errors": ["timeout"],
            "retry_backoff": "none",
            "request_timeout_seconds": 37,
        })
        entry = manifest["execution_contract"]["schedule"][0]
        with tempfile.TemporaryDirectory() as tmp:
            engine = _default_engine_factory(entry, manifest, Path(tmp))
            self.assertEqual(engine.agent_action_max_attempts, 1)
            self.assertEqual(engine.retryable_error_categories, ["timeout"])
            self.assertEqual(engine.retry_backoff, "none")
            self.assertEqual(engine.request_timeout_seconds, 37)
            engine.close()

    def test_formal_manifest_defaults_to_fail_closed_execution(self):
        manifest = build_experiment_manifest(
            experiment_id="formal",
            conditions=CONDITIONS,
            seeds=[1],
            repetitions=1,
        )
        policies = manifest["execution_contract"]["policies"]
        self.assertFalse(policies["allow_provider_fallback"])
        self.assertEqual(policies["action_failure_policy"], "abort_game")

    def test_crossed_comparisons_validate(self):
        from werewolf.experiments.conditions import build_crossed_conditions
        manifest = build_experiment_manifest(
            experiment_id="crossed",
            conditions=build_crossed_conditions("fast", "gemini_flash_lite"),
            seeds=[1, 2],
            repetitions=2,
            comparisons=default_crossed_comparisons(),
        )
        self.assertEqual(validate_manifest_for_execution(manifest), [])
        self.assertEqual(len(manifest["comparisons"]), 3)

    def test_independent_comparison_is_rejected_in_v1(self):
        comparison = {**default_crossed_comparisons()[0],
                      "design": "independent"}
        with self.assertRaises(ManifestError):
            build_experiment_manifest(
                experiment_id="independent",
                conditions=CONDITIONS,
                seeds=[1], repetitions=1,
                comparisons=[comparison],
            )

    def test_unknown_policy_rejected(self):
        from werewolf.experiments.manifest import ManifestError
        with self.assertRaises(ManifestError):
            make_manifest(policies={"warp_speed": True,
                                    **OFFLINE_POLICIES})

    def test_deep_execution_validation_blocks_before_health_preflight(self):
        manifest = copy.deepcopy(make_manifest())
        manifest["execution_contract"]["game"]["n_wolves"] = 99
        manifest["execution_contract"]["generation"]["top_p"] = 2
        manifest["execution_contract"]["policies"]["request_timeout_seconds"] = 0
        manifest["execution_contract"]["policies"]["retryable_errors"] = [
            "not_a_category",
        ]
        manifest["analysis_contract"]["bootstrap"]["alpha"] = 1
        manifest = finalize_manifest(manifest)
        errors = validate_manifest_for_execution(manifest)
        self.assertTrue(any("n_wolves" in error for error in errors))
        self.assertTrue(any("top_p" in error for error in errors))
        self.assertTrue(any("request_timeout" in error for error in errors))
        self.assertTrue(any("retryable_errors" in error for error in errors))
        self.assertTrue(any("bootstrap.alpha" in error for error in errors))

    def test_invalid_manifest_does_not_run_paid_health_preflight(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = copy.deepcopy(make_manifest())
            manifest["execution_contract"]["game"]["n_players"] = 2
            write_manifest(tmp, finalize_manifest(manifest))
            prober = mock.Mock(side_effect=AssertionError("must not probe"))
            with self.assertRaises(ExperimentRunError):
                run(tmp, health_prober=prober)
            prober.assert_not_called()


if __name__ == "__main__":
    unittest.main()
