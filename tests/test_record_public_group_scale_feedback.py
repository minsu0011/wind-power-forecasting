from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts/record_public_group_scale_feedback.py"
FILES = {
    "base": PROJECT_ROOT / "artifacts/final_cf_fix/corrected_recent_v4.csv",
    "global092": PROJECT_ROOT
    / "artifacts/postgate/public_scale_probe/corrected_recent_v4_scale_092_2025.csv",
    "g3_only": PROJECT_ROOT
    / "artifacts/postgate/public_group_scale_probe/corrected_recent_v4_g3_only_scale_092_2025.csv",
    "g1_only": PROJECT_ROOT
    / "artifacts/postgate/public_group_scale_probe/corrected_recent_v4_g1_only_scale_092_2025.csv",
}
METRICS = {
    "base": (0.6122309211, 0.8518461479, 0.3726156943),
    "global092": (0.615, 0.855, 0.375),
    "g3_only": (0.613, 0.853, 0.373),
    "g1_only": (0.613, 0.852, 0.374),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def submission(label: str, metrics: tuple[float, float, float] | None = None) -> dict[str, object]:
    score, one_minus_nmae, ficr = metrics or METRICS[label]
    path = FILES[label]
    return {
        "label": label,
        "file": str(path.relative_to(PROJECT_ROOT)),
        "sha256": sha256(path),
        "score": score,
        "one_minus_nmae": one_minus_nmae,
        "ficr": ficr,
    }


class RecordPublicGroupScaleFeedbackCliTests(unittest.TestCase):
    def invoke(
        self,
        temporary: Path,
        labels: list[str],
        *,
        mutations: dict[str, object] | None = None,
        output_name: str = "decision_pack",
    ) -> tuple[subprocess.CompletedProcess[str], Path, Path]:
        records = [submission(label) for label in labels]
        if mutations:
            records[-1].update(mutations)
        feedback = {
            "schema_version": 1,
            "reported_at_kst": "2026-08-08T12:00:00+09:00",
            "source": "synthetic unit test; not a leaderboard result",
            "submissions": records,
        }
        feedback_path = temporary / f"feedback_{output_name}.json"
        feedback_path.write_text(
            json.dumps(feedback, indent=2) + "\n", encoding="utf-8"
        )
        out_dir = temporary / output_name
        completed = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--feedback",
                str(feedback_path),
                "--out-dir",
                str(out_dir),
            ],
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        return completed, feedback_path, out_dir

    def assert_success(self, completed: subprocess.CompletedProcess[str]) -> None:
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_cli_frozen_stage_transitions(self) -> None:
        cases = (
            (["base", "global092"], "global_gate_passed", "g3_only"),
            (["base", "global092", "g3_only"], "g3_observed", "g1_only"),
            (
                ["base", "global092", "g3_only", "g1_only"],
                "two_group_probes_observed_g2_inferred",
                None,
            ),
        )
        with tempfile.TemporaryDirectory() as raw_temporary:
            temporary = Path(raw_temporary)
            for index, (labels, expected_stage, expected_next) in enumerate(cases):
                with self.subTest(labels=labels):
                    completed, feedback_path, out_dir = self.invoke(
                        temporary, labels, output_name=f"stage_{index}"
                    )
                    self.assert_success(completed)
                    decision = json.loads(
                        (out_dir / "decision.json").read_text(encoding="utf-8")
                    )
                    state = decision["decision"]
                    self.assertEqual(state["stage"], expected_stage)
                    observed_next = (
                        None
                        if state["next_submission"] is None
                        else state["next_submission"]["label"]
                    )
                    self.assertEqual(observed_next, expected_next)
                    self.assertIsNone(state["automatic_final_scale_choice"])
                    self.assertFalse(decision["submission_performed"])
                    self.assertFalse(decision["upload_capability"])
                    self.assertEqual(
                        (out_dir / "feedback_input.json").read_bytes(),
                        feedback_path.read_bytes(),
                    )

    def test_cli_infers_g2_componentwise_without_final_choice(self) -> None:
        with tempfile.TemporaryDirectory() as raw_temporary:
            completed, _, out_dir = self.invoke(
                Path(raw_temporary),
                ["base", "global092", "g3_only", "g1_only"],
            )
            self.assert_success(completed)
            state = json.loads(
                (out_dir / "decision.json").read_text(encoding="utf-8")
            )["decision"]
            deltas = state["group_deltas"]
            self.assertTrue(state["g2_delta_inferred"])
            self.assertEqual(deltas["kpx_group_2"]["source"], "inferred_by_macro_separability")
            for component in ("score", "one_minus_nmae", "ficr"):
                observed_sum = sum(deltas[group][component] for group in deltas)
                self.assertAlmostEqual(
                    observed_sum, state["global_delta"][component], places=15
                )
            self.assertIsNone(state["next_submission"])
            self.assertIsNone(state["automatic_final_scale_choice"])

    def test_cli_gate_failure_has_no_next_submission(self) -> None:
        with tempfile.TemporaryDirectory() as raw_temporary:
            completed, _, out_dir = self.invoke(
                Path(raw_temporary),
                ["base", "global092"],
                mutations={
                    "score": 0.61,
                    "one_minus_nmae": 0.85,
                    "ficr": 0.37,
                },
            )
            self.assert_success(completed)
            state = json.loads(
                (out_dir / "decision.json").read_text(encoding="utf-8")
            )["decision"]
            self.assertEqual(state["stage"], "global_gate_failed")
            self.assertFalse(state["activation_gate"]["passed"])
            self.assertIsNone(state["next_submission"])

    def test_cli_gate_uses_frozen_base_at_divergent_tolerance_boundary(self) -> None:
        frozen_score, frozen_one_minus_nmae, frozen_ficr = METRICS["base"]
        cases = (
            (
                -4e-10,
                -2e-10,
                False,
                "global_gate_failed",
                None,
            ),
            (
                4e-10,
                2e-10,
                True,
                "global_gate_passed",
                "g3_only",
            ),
        )
        with tempfile.TemporaryDirectory() as raw_temporary:
            temporary = Path(raw_temporary)
            for index, (
                base_shift,
                global_shift,
                expected_pass,
                expected_stage,
                expected_next,
            ) in enumerate(cases):
                with self.subTest(base_shift=base_shift, global_shift=global_shift):
                    records = [
                        submission(
                            "base",
                            (
                                frozen_score + base_shift,
                                frozen_one_minus_nmae + base_shift,
                                frozen_ficr + base_shift,
                            ),
                        ),
                        submission(
                            "global092",
                            (
                                frozen_score + global_shift,
                                frozen_one_minus_nmae + global_shift,
                                frozen_ficr + global_shift,
                            ),
                        ),
                    ]
                    feedback_path = temporary / f"boundary_{index}.json"
                    feedback_path.write_text(
                        json.dumps({"schema_version": 1, "submissions": records}) + "\n",
                        encoding="utf-8",
                    )
                    out_dir = temporary / f"boundary_out_{index}"
                    completed = subprocess.run(
                        [sys.executable, str(SCRIPT), "--feedback", str(feedback_path), "--out-dir", str(out_dir)],
                        cwd=PROJECT_ROOT,
                        text=True,
                        capture_output=True,
                        check=False,
                    )
                    self.assert_success(completed)
                    state = json.loads(
                        (out_dir / "decision.json").read_text(encoding="utf-8")
                    )["decision"]
                    gate = state["activation_gate"]
                    self.assertEqual(gate["passed"], expected_pass)
                    self.assertEqual(gate["base_score"], frozen_score)
                    self.assertEqual(state["stage"], expected_stage)
                    observed_next = (
                        None
                        if state["next_submission"] is None
                        else state["next_submission"]["label"]
                    )
                    self.assertEqual(observed_next, expected_next)

    def test_cli_rejects_duplicate_submission(self) -> None:
        with tempfile.TemporaryDirectory() as raw_temporary:
            temporary = Path(raw_temporary)
            records = [submission("base"), submission("global092"), submission("base")]
            feedback_path = temporary / "duplicate.json"
            feedback_path.write_text(
                json.dumps({"schema_version": 1, "submissions": records}) + "\n",
                encoding="utf-8",
            )
            out_dir = temporary / "out"
            completed = subprocess.run(
                [sys.executable, str(SCRIPT), "--feedback", str(feedback_path), "--out-dir", str(out_dir)],
                cwd=PROJECT_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("duplicate submission", completed.stderr)
            self.assertFalse(out_dir.exists())

    def test_cli_rejects_bad_metric_identity_and_wrong_sha(self) -> None:
        cases = (
            ({"score": 0.7}, "score is inconsistent"),
            ({"sha256": "0" * 64}, "supplied SHA does not match"),
        )
        with tempfile.TemporaryDirectory() as raw_temporary:
            temporary = Path(raw_temporary)
            for index, (mutation, message) in enumerate(cases):
                with self.subTest(mutation=mutation):
                    completed, _, out_dir = self.invoke(
                        temporary,
                        ["base", "global092"],
                        mutations=mutation,
                        output_name=f"invalid_{index}",
                    )
                    self.assertNotEqual(completed.returncode, 0)
                    self.assertIn(message, completed.stderr)
                    self.assertFalse(out_dir.exists())

    def test_cli_rejects_metric_components_outside_unit_interval(self) -> None:
        cases = (
            ({"one_minus_nmae": 1.0000000001}, "one_minus_nmae must be within [0, 1]"),
            ({"ficr": -0.0000000001}, "ficr must be within [0, 1]"),
            ({"score": 1.0000000001}, "score must be within [0, 1]"),
        )
        with tempfile.TemporaryDirectory() as raw_temporary:
            temporary = Path(raw_temporary)
            for index, (mutation, message) in enumerate(cases):
                with self.subTest(mutation=mutation):
                    completed, _, out_dir = self.invoke(
                        temporary,
                        ["base", "global092"],
                        mutations=mutation,
                        output_name=f"range_{index}",
                    )
                    self.assertNotEqual(completed.returncode, 0)
                    self.assertIn(message, completed.stderr)
                    self.assertFalse(out_dir.exists())

    def test_cli_rejects_feedback_outside_frozen_order(self) -> None:
        with tempfile.TemporaryDirectory() as raw_temporary:
            temporary = Path(raw_temporary)
            records = [submission("global092"), submission("base")]
            feedback_path = temporary / "wrong_order.json"
            feedback_path.write_text(
                json.dumps({"schema_version": 1, "submissions": records}) + "\n",
                encoding="utf-8",
            )
            out_dir = temporary / "out"
            completed = subprocess.run(
                [sys.executable, str(SCRIPT), "--feedback", str(feedback_path), "--out-dir", str(out_dir)],
                cwd=PROJECT_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("submission order must be the frozen prefix", completed.stderr)
            self.assertFalse(out_dir.exists())

    def test_cli_manifest_hashes_and_source_artifacts_remain_unchanged(self) -> None:
        before = {
            path: (sha256(path), path.stat().st_size, path.stat().st_mtime_ns)
            for path in FILES.values()
        }
        with tempfile.TemporaryDirectory() as raw_temporary:
            completed, _, out_dir = self.invoke(
                Path(raw_temporary), ["base", "global092"]
            )
            self.assert_success(completed)
            manifest = json.loads(
                (out_dir / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertTrue(manifest["public_adaptive"])
            self.assertTrue(manifest["selection_unsafe"])
            self.assertFalse(manifest["private_champion"])
            self.assertFalse(manifest["submission_performed"])
            self.assertFalse(manifest["upload_capability"])
            for section in ("inputs", "outputs"):
                for record in manifest[section]:
                    path = Path(record["path"])
                    self.assertEqual(sha256(path), record["sha256"])
                    self.assertEqual(path.stat().st_size, record["size_bytes"])
        after = {
            path: (sha256(path), path.stat().st_size, path.stat().st_mtime_ns)
            for path in FILES.values()
        }
        self.assertEqual(before, after)

    def test_cli_refuses_overwrite_and_preserves_first_decision(self) -> None:
        with tempfile.TemporaryDirectory() as raw_temporary:
            temporary = Path(raw_temporary)
            first, feedback_path, out_dir = self.invoke(
                temporary, ["base", "global092"]
            )
            self.assert_success(first)
            before = {
                path.name: sha256(path) for path in out_dir.iterdir() if path.is_file()
            }
            second = subprocess.run(
                [sys.executable, str(SCRIPT), "--feedback", str(feedback_path), "--out-dir", str(out_dir)],
                cwd=PROJECT_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            after = {
                path.name: sha256(path) for path in out_dir.iterdir() if path.is_file()
            }
            self.assertNotEqual(second.returncode, 0)
            self.assertIn("refusing to overwrite", second.stderr)
            self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
