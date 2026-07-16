import json
import tempfile
import unittest
from pathlib import Path

from werewolf.experiments.canonical import (
    CanonicalizationError,
    canonical_object_digest,
    jcs_canonicalize,
    jcs_sha256,
)
from werewolf.experiments.manifest import (
    ManifestError,
    ManifestImmutableError,
    analysis_contract_sha256,
    compute_manifest_hash,
    execution_contract_sha256,
    finalize_manifest,
    journal_path,
    load_verified_manifest,
    manifest_is_frozen,
    validate_manifest,
    write_manifest,
)
from werewolf.experiments.runtime_hash import (
    analysis_runtime_hash,
    execution_runtime_hash,
)


def make_manifest(experiment_id="exp_test", **overrides) -> dict:
    manifest = {
        "manifest_schema_version": 1,
        "experiment_id": experiment_id,
        "created_at": "2026-07-15T00:00:00+00:00",
        "description": "test experiment",
        "execution_contract": {
            "conditions": {
                "a_homogeneous": {
                    "role_models": {
                        "werewolf": "fast", "villager": "fast", "seer": "fast",
                    },
                },
            },
            "game": {"n_players": 7, "n_wolves": 2, "n_seers": 1,
                     "discussion_cycles": 2, "max_rounds": 20},
            "seeds": [5000, 5001],
            "repetitions": 2,
            "prompt_profile": {"name": "baseline_v1", "version": "x"},
            "models": {"fast": {"provider": "xai", "requested_model": "grok-4.3"}},
            "generation": {"max_output_tokens": 4096},
            "predeclared_adjustments": [],
            "policies": {"allow_provider_fallback": False,
                         "max_trial_attempts": 2},
            "scheduler": {"version": 1, "scheduler_seed": 7},
            "schedule": [],
            "execution_runtime_hash": "e" * 64,
        },
        "analysis_contract": {
            "report_build_version": 12,
            "validity_policy_version": 4,
            "belief_metrics_version": 2,
            "aggregate_analysis_version": 1,
            "comparison_method_version": 1,
            "bootstrap": {"n_boot": 2000, "alpha": 0.05, "rng_seed": 0},
            "metric_weighting": "aggregate-1",
            "analysis_runtime_hash": "a" * 64,
        },
        "comparisons": [],
        "metadata": {"repository_commit": None, "working_tree_dirty": None},
    }
    manifest.update(overrides)
    return manifest


class JCSCanonicalizationTests(unittest.TestCase):
    def test_rfc8785_appendix_style_vector(self):
        value = {
            "numbers": [333333333.33333329, 1e30, 4.50, 2e-3,
                        0.000000000000000000000000001],
            "string": "€$\nA'B\"\\\\\"/",
            "literals": [None, True, False],
        }
        expected = (
            '{"literals":[null,true,false],'
            '"numbers":[333333333.3333333,1e+30,4.5,0.002,1e-27],'
            '"string":"€$\\u000f\\nA\'B\\"\\\\\\\\\\"/"}'
        )
        self.assertEqual(jcs_canonicalize(value), expected)

    def test_number_formatting(self):
        cases = {
            0.0: "0",
            -0.0: "0",
            1.0: "1",
            -1.5: "-1.5",
            123.0: "123",
            0.5: "0.5",
            1e21: "1e+21",
            1e20: "100000000000000000000",
            1e-6: "0.000001",
            1e-7: "1e-7",
            5e-324: "5e-324",
            1.7976931348623157e308: "1.7976931348623157e+308",
        }
        for value, expected in cases.items():
            self.assertEqual(jcs_canonicalize(value), expected, msg=repr(value))
        self.assertEqual(jcs_canonicalize(10), "10")

    def test_utf16_key_ordering(self):
        # U+1F600 is a surrogate pair starting 0xD83D in UTF-16, which
        # sorts BEFORE U+FF5A even though its code point is larger.
        text = jcs_canonicalize({"ｚ": 1, "\U0001f600": 2})
        self.assertEqual(text, '{"\U0001f600":2,"ｚ":1}')

    def test_rejects_uncanonicalizable_values(self):
        for bad in (float("nan"), float("inf"), {1: "x"}, {"x": object()},
                    2**53 + 1):
            with self.assertRaises(CanonicalizationError, msg=repr(bad)):
                jcs_canonicalize(bad)

    def test_canonicalization_is_input_order_independent(self):
        a = {"b": 1, "a": [1, 2.0, "z"]}
        b = {"a": [1, 2.0, "z"], "b": 1}
        self.assertEqual(jcs_sha256(a), jcs_sha256(b))

    def test_domain_separation(self):
        payload = {"seed": 1, "condition_id": "a"}
        self.assertNotEqual(
            canonical_object_digest("trial-id-v1", payload),
            canonical_object_digest("schedule-order-v1", payload),
        )


class ManifestHashTests(unittest.TestCase):
    def test_self_hash_is_omitted_from_hash_input(self):
        manifest = finalize_manifest(make_manifest())
        recorded = manifest["manifest_content_sha256"]
        # Recomputing over the finalized manifest (which now contains the
        # hash field) must ignore that field and reproduce the same hash.
        self.assertEqual(compute_manifest_hash(manifest), recorded)
        self.assertEqual(validate_manifest(manifest), [])

    def test_content_change_changes_hash(self):
        base = finalize_manifest(make_manifest())
        changed = finalize_manifest(make_manifest(description="other"))
        self.assertNotEqual(
            base["manifest_content_sha256"],
            changed["manifest_content_sha256"],
        )

    def test_tampered_manifest_fails_validation(self):
        manifest = finalize_manifest(make_manifest())
        manifest["description"] = "tampered"
        errors = validate_manifest(manifest)
        self.assertTrue(any("manifest_content_sha256" in e for e in errors))

    def test_analysis_changes_do_not_move_execution_contract_hash(self):
        base = make_manifest()
        modified = make_manifest()
        modified["analysis_contract"] = dict(
            modified["analysis_contract"], aggregate_analysis_version=2,
        )
        self.assertEqual(
            execution_contract_sha256(base),
            execution_contract_sha256(modified),
        )
        self.assertNotEqual(
            analysis_contract_sha256(base),
            analysis_contract_sha256(modified),
        )
        self.assertNotEqual(
            finalize_manifest(base)["manifest_content_sha256"],
            finalize_manifest(modified)["manifest_content_sha256"],
        )

    def test_execution_policy_change_moves_execution_contract_hash(self):
        base = make_manifest()
        modified = make_manifest()
        modified["execution_contract"] = dict(
            modified["execution_contract"],
            policies={"allow_provider_fallback": True,
                      "max_trial_attempts": 2},
        )
        self.assertNotEqual(
            execution_contract_sha256(base),
            execution_contract_sha256(modified),
        )

    def test_finalize_rejects_structurally_invalid_manifests(self):
        broken = make_manifest()
        del broken["execution_contract"]["seeds"]
        with self.assertRaises(ManifestError):
            finalize_manifest(broken)
        with self.assertRaises(ManifestError):
            finalize_manifest(make_manifest(experiment_id="../escape"))

    def test_duplicate_seeds_rejected(self):
        broken = make_manifest()
        broken["execution_contract"] = dict(
            broken["execution_contract"], seeds=[5000, 5000],
        )
        with self.assertRaises(ManifestError):
            finalize_manifest(broken)


class ManifestStorageTests(unittest.TestCase):
    def test_write_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = finalize_manifest(make_manifest())
            path = write_manifest(tmp, manifest)
            self.assertTrue(path.exists())
            loaded = load_verified_manifest(tmp, "exp_test")
            self.assertEqual(loaded, json.loads(json.dumps(manifest)))

    def test_identical_rewrite_is_a_noop(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = finalize_manifest(make_manifest())
            write_manifest(tmp, manifest)
            write_manifest(tmp, manifest)  # no error

    def test_conflicting_write_rejected_before_freeze(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_manifest(tmp, finalize_manifest(make_manifest()))
            changed = finalize_manifest(make_manifest(description="v2"))
            with self.assertRaises(ManifestError):
                write_manifest(tmp, changed)

    def test_manifest_immutable_after_first_lifecycle_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = finalize_manifest(make_manifest())
            write_manifest(tmp, manifest)
            self.assertFalse(manifest_is_frozen(tmp, "exp_test"))
            journal = journal_path(tmp, "exp_test")
            journal.write_text('{"record_type":"execution_session_started"}\n')
            self.assertTrue(manifest_is_frozen(tmp, "exp_test"))
            changed = finalize_manifest(make_manifest(description="v2"))
            with self.assertRaises(ManifestImmutableError):
                write_manifest(tmp, changed)


class RuntimeHashTests(unittest.TestCase):
    def _seed_repo(self, root: Path):
        (root / "werewolf" / "engine").mkdir(parents=True)
        (root / "werewolf" / "reporting").mkdir(parents=True)
        (root / "werewolf" / "engine" / "game.py").write_text("ENGINE = 1\n")
        (root / "werewolf" / "reporting" / "builder.py").write_text("R = 1\n")
        (root / "requirements.txt").write_text("litellm==1.92.0\nflask==3.1.3\n")

    def test_execution_change_moves_only_execution_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._seed_repo(root)
            exec_before = execution_runtime_hash(root)
            analysis_before = analysis_runtime_hash(root)
            (root / "werewolf" / "engine" / "game.py").write_text("ENGINE = 2\n")
            self.assertNotEqual(execution_runtime_hash(root), exec_before)
            self.assertEqual(analysis_runtime_hash(root), analysis_before)

    def test_analysis_change_moves_only_analysis_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._seed_repo(root)
            exec_before = execution_runtime_hash(root)
            analysis_before = analysis_runtime_hash(root)
            (root / "werewolf" / "reporting" / "builder.py").write_text("R = 2\n")
            self.assertEqual(execution_runtime_hash(root), exec_before)
            self.assertNotEqual(analysis_runtime_hash(root), analysis_before)

    def test_execution_dependency_pin_moves_execution_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._seed_repo(root)
            exec_before = execution_runtime_hash(root)
            (root / "requirements.txt").write_text(
                "litellm==1.93.0\nflask==3.1.3\n"
            )
            self.assertNotEqual(execution_runtime_hash(root), exec_before)

    def test_analysis_irrelevant_dependency_pin_is_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._seed_repo(root)
            analysis_before = analysis_runtime_hash(root)
            (root / "requirements.txt").write_text(
                "litellm==1.93.0\nflask==3.2.0\n"
            )
            self.assertEqual(analysis_runtime_hash(root), analysis_before)


if __name__ == "__main__":
    unittest.main()
