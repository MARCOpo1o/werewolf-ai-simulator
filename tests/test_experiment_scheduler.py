import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

from werewolf.experiments.canonical import canonical_object_digest, jcs_sha256
from werewolf.experiments.conditions import build_crossed_conditions
from werewolf.experiments.locks import (
    LockHeldError,
    analysis_lock,
    execution_lock,
)
from werewolf.experiments.scheduler import (
    SCHEDULER_VERSION,
    materialize_schedule,
    schedule_order_key,
    trial_id,
    verify_schedule,
)

CONDITIONS = build_crossed_conditions("fast", "gemini_flash_lite")


def make_schedule(scheduler_seed=7, seeds=(5000, 5001, 5002), repetitions=2):
    return materialize_schedule(
        experiment_id="exp",
        conditions=CONDITIONS,
        seeds=list(seeds),
        repetitions=repetitions,
        scheduler_seed=scheduler_seed,
    )


class SchedulerTests(unittest.TestCase):
    def test_schedule_is_deterministic_and_complete(self):
        first, second = make_schedule(), make_schedule()
        self.assertEqual(first, second)
        self.assertEqual(len(first), 4 * 3 * 2)
        combos = {(e["condition_id"], e["seed"], e["repetition"])
                  for e in first}
        self.assertEqual(len(combos), len(first))
        self.assertEqual([e["trial_index"] for e in first],
                         list(range(len(first))))

    def test_block_order_matches_documented_hash_formula(self):
        schedule = make_schedule()
        for seed in (5000, 5001, 5002):
            for repetition in (0, 1):
                block = [e for e in schedule
                         if e["seed"] == seed and e["repetition"] == repetition]
                expected = sorted(
                    (e["condition_id"] for e in block),
                    key=lambda c: jcs_sha256({
                        "scheduler_seed": 7,
                        "seed": seed,
                        "repetition": repetition,
                        "condition_id": c,
                    }),
                )
                self.assertEqual([e["condition_id"] for e in block], expected)
                self.assertEqual([e["scheduler_position"] for e in block],
                                 list(range(len(block))))

    def test_condition_order_varies_across_blocks(self):
        schedule = make_schedule(seeds=range(5000, 5010), repetitions=2)
        orders = set()
        for seed in range(5000, 5010):
            for repetition in (0, 1):
                orders.add(tuple(
                    e["condition_id"] for e in schedule
                    if e["seed"] == seed and e["repetition"] == repetition
                ))
        self.assertGreater(len(orders), 1)

    def test_scheduler_seed_changes_order(self):
        self.assertNotEqual(
            [e["trial_id"] for e in make_schedule(scheduler_seed=7)],
            [e["trial_id"] for e in make_schedule(scheduler_seed=8)],
        )

    def test_trial_ids_are_stable_and_domain_separated(self):
        a = trial_id("exp", "a_homogeneous", 5000, 0)
        self.assertEqual(a, trial_id("exp", "a_homogeneous", 5000, 0))
        self.assertNotEqual(a, trial_id("exp2", "a_homogeneous", 5000, 0))
        self.assertTrue(a.startswith("trial_"))
        # The ordering key hashes the same logical fields but must never
        # equal the trial identity digest.
        self.assertNotIn(
            schedule_order_key(7, 5000, 0, "a_homogeneous")[:32],
            a,
        )
        self.assertNotEqual(
            canonical_object_digest("werewolf-experiment-trial-id-v1", {
                "experiment_id": "exp", "condition_id": "a_homogeneous",
                "seed": 5000, "repetition": 0,
            }),
            jcs_sha256({
                "experiment_id": "exp", "condition_id": "a_homogeneous",
                "seed": 5000, "repetition": 0,
            }),
        )

    def test_verify_schedule_detects_tampering(self):
        schedule = make_schedule()
        manifest = {
            "experiment_id": "exp",
            "execution_contract": {
                "conditions": CONDITIONS,
                "seeds": [5000, 5001, 5002],
                "repetitions": 2,
                "scheduler": {"version": SCHEDULER_VERSION,
                              "scheduler_seed": 7},
                "schedule": schedule,
            },
        }
        self.assertEqual(verify_schedule(manifest), [])
        tampered = [dict(e) for e in schedule]
        tampered[0], tampered[1] = tampered[1], tampered[0]
        manifest["execution_contract"]["schedule"] = tampered
        self.assertTrue(verify_schedule(manifest))
        manifest["execution_contract"]["schedule"] = schedule
        manifest["execution_contract"]["scheduler"]["version"] = 99
        self.assertTrue(verify_schedule(manifest))


class AdvisoryLockTests(unittest.TestCase):
    def test_lock_excludes_other_holders(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = execution_lock(tmp, "exp").acquire()
            try:
                with self.assertRaises(LockHeldError):
                    execution_lock(tmp, "exp").acquire()
            finally:
                first.release()
            second = execution_lock(tmp, "exp").acquire()
            second.release()

    def test_execution_and_analysis_locks_are_independent(self):
        with tempfile.TemporaryDirectory() as tmp:
            with execution_lock(tmp, "exp"):
                with analysis_lock(tmp, "exp"):
                    pass

    def test_lock_metadata_is_diagnostic(self):
        with tempfile.TemporaryDirectory() as tmp:
            with execution_lock(tmp, "exp") as lock:
                text = Path(lock.path).read_text(encoding="utf-8")
                self.assertIn("\"pid\"", text)
                self.assertIn("\"experiment_id\"", text)

    def test_lock_conflicts_across_processes(self):
        with tempfile.TemporaryDirectory() as tmp:
            with execution_lock(tmp, "exp"):
                probe = textwrap.dedent(f"""
                    import sys
                    sys.path.insert(0, {str(Path.cwd())!r})
                    from werewolf.experiments.locks import (
                        LockHeldError, execution_lock,
                    )
                    try:
                        execution_lock({tmp!r}, "exp").acquire()
                    except LockHeldError:
                        sys.exit(42)
                    sys.exit(0)
                """)
                result = subprocess.run(
                    [sys.executable, "-c", probe],
                    capture_output=True, text=True, timeout=30,
                )
                self.assertEqual(result.returncode, 42, result.stderr)


if __name__ == "__main__":
    unittest.main()
