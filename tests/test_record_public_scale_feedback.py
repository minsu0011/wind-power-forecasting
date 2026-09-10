from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts/record_public_scale_feedback.py"
CONFIG = PROJECT_ROOT / "configs/public_scale_feedback_20260808_1630_preregister.json"
ADDENDUM = PROJECT_ROOT / (
    "configs/public_scale_feedback_20260808_1630_quadratic_addendum_preregister.json"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class RecordPublicScaleFeedbackTests(unittest.TestCase):
    def invoke(self, out_dir: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--config",
                str(CONFIG),
                "--quadratic-addendum",
                str(ADDENDUM),
                "--out-dir",
                str(out_dir),
            ],
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_forensics_exact_deltas_risk_flags_and_manifest(self) -> None:
        source_paths = [
            CONFIG,
            ADDENDUM,
            PROJECT_ROOT / "artifacts/final_cf_fix/corrected_recent_v4.csv",
            PROJECT_ROOT
            / "artifacts/postgate/public_scale_probe/corrected_recent_v4_scale_092_2025.csv",
            PROJECT_ROOT
            / "artifacts/postgate/public_scale_probe/corrected_recent_v4_scale_095_2025.csv",
        ]
        before = {path: (sha256(path), path.stat().st_size) for path in source_paths}
        with tempfile.TemporaryDirectory() as raw_temporary:
            out_dir = Path(raw_temporary) / "feedback_pack"
            completed = self.invoke(out_dir)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertFalse(list(out_dir.glob("*.csv")))
            analysis = json.loads((out_dir / "analysis.json").read_text(encoding="utf-8"))
            self.assertAlmostEqual(
                analysis["component_deltas"]["global_scale_095_minus_base_v4"]["score"],
                0.0006668153,
                places=13,
            )
            quadratic = analysis["descriptive_quadratic_only"]
            self.assertAlmostEqual(
                quadratic["interpolated_vertex_scale"], 0.9722074553716762, places=14
            )
            self.assertEqual(quadratic["residual_degrees_of_freedom"], 0)
            self.assertFalse(quadratic["recommendation_allowed"])
            self.assertFalse(quadratic["statistical_confidence_interval_estimable"])
            self.assertEqual(analysis["daily_submission_state"]["used"], 5)
            self.assertEqual(analysis["daily_submission_state"]["remaining"], 0)
            self.assertTrue(analysis["proxy_refutation"]["candidate_order_reversed"])
            self.assertFalse(
                analysis["identifiability"]["justified_public_based_new_correction"]
            )
            manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
            for section in ("inputs", "outputs"):
                for item in manifest[section]:
                    path = Path(item["path"])
                    self.assertTrue(path.is_file(), path)
                    self.assertEqual(sha256(path), item["sha256"])
                    self.assertEqual(path.stat().st_size, item["size_bytes"])
            self.assertEqual((out_dir / "preregister.json").read_bytes(), CONFIG.read_bytes())
            self.assertEqual(
                (out_dir / "quadratic_addendum_preregister.json").read_bytes(),
                ADDENDUM.read_bytes(),
            )
        after = {path: (sha256(path), path.stat().st_size) for path in source_paths}
        self.assertEqual(before, after)

    def test_refuses_to_overwrite_first_pack(self) -> None:
        with tempfile.TemporaryDirectory() as raw_temporary:
            out_dir = Path(raw_temporary) / "feedback_pack"
            first = self.invoke(out_dir)
            self.assertEqual(first.returncode, 0, first.stderr)
            before = {path.name: sha256(path) for path in out_dir.iterdir()}
            second = self.invoke(out_dir)
            after = {path.name: sha256(path) for path in out_dir.iterdir()}
            self.assertNotEqual(second.returncode, 0)
            self.assertIn("refusing to overwrite", second.stderr)
            self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
