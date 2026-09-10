from __future__ import annotations

import ctypes
import inspect
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

import pandas as pd

from scripts import run_raw_spatiotemporal_smooth_ficr as runner
from src.metric import TARGET_COLS


ROOT = Path(__file__).resolve().parents[1]


class RawSpatiotemporalSmoothFICRRunnerTests(unittest.TestCase):
    def test_09_verify_config_binds_single_factor_contract(self) -> None:
        config, raw_contract = runner.verify_config(runner.CONFIG_PATH)
        self.assertEqual(config["execution_protocol_changes"]["scientific_change_count"], 0)
        self.assertEqual(config["execution_protocol_changes"]["protocol_change_count"], 1)
        self.assertEqual(
            config["complete_scientific_contract_inheritance"]["blend_weights_unchanged"],
            [0.025, 0.05],
        )
        self.assertIn("physical_stage1_inputs", raw_contract)

    def test_10_recursive_source_closure_contains_new_and_inherited_logic(self) -> None:
        config, _ = runner.verify_config(runner.CONFIG_PATH)
        closure = runner.source_closure(config, runner.DEFAULT_RAW_DIR)
        paths = set(closure["resolved_relative_paths"])
        explicit = {row["relative_path"] for row in closure["explicit_sources"]}
        self.assertIn("scripts/run_raw_spatiotemporal_smooth_ficr.py", paths)
        self.assertIn("src/raw_spatiotemporal_smooth_ficr.py", paths)
        self.assertIn("src/raw_spatiotemporal_attention.py", paths)
        self.assertIn("src/smooth_ficr_objective.py", paths)
        self.assertEqual(
            explicit,
            {
                "scripts/launch_raw_spatiotemporal_smooth_ficr_v2.ps1",
                "tests/test_raw_spatiotemporal_smooth_ficr.py",
                "tests/test_raw_spatiotemporal_smooth_ficr_runner.py",
            },
        )

    def test_11_registered_group_slice_count_is_exactly_17(self) -> None:
        self.assertEqual(sum(len(value) for value in runner.STAGE1_REQUIRED.values()), 17)
        self.assertEqual(tuple(runner._segments(TARGET_COLS[0])), runner.STAGE1_REQUIRED[TARGET_COLS[0]])
        self.assertEqual(tuple(runner._segments(TARGET_COLS[2])), runner.STAGE1_REQUIRED[TARGET_COLS[2]])

    def test_12_g3_reader_fails_before_call_without_g12_candidate_lock(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.json"
            with mock.patch.object(runner.raw_protocol, "_read_label_prefix") as reader:
                with self.assertRaisesRegex(RuntimeError, "lock is absent"):
                    runner._read_g3_fit_labels_after_g12_lock(
                        Path(directory), {}, missing
                    )
                reader.assert_not_called()

    def test_13_score_reader_fails_before_call_with_invalid_global_lock(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            invalid = Path(directory) / "invalid.json"
            invalid.write_text(
                json.dumps(
                    {
                        "preregister_sha256": runner.CONFIG_SHA256,
                        "all_models_preprocessing_predictions_candidates_frozen": False,
                    }
                ),
                encoding="utf-8",
            )
            with mock.patch.object(runner.raw_protocol, "_read_label_prefix") as reader:
                with self.assertRaisesRegex(RuntimeError, "lock is not valid"):
                    runner._read_score_labels_after_global_lock(
                        Path(directory), {}, invalid
                    )
                reader.assert_not_called()

    def test_14_stage1_source_order_locks_each_candidate_before_label_reader(self) -> None:
        source = inspect.getsource(runner.run_stage1)
        g12_lock = source.index('lock_g12 = out_dir / "stage1_g12_candidate_lock_before_application_labels.json"')
        g3_reader = source.index("_read_g3_fit_labels_after_g12_lock")
        global_lock = source.index('global_lock = out_dir / "stage1_global_candidate_lock_before_score_labels.json"')
        score_reader = source.index("_read_score_labels_after_global_lock")
        self.assertLess(g12_lock, g3_reader)
        self.assertLess(global_lock, score_reader)
        self.assertIn("comparison_count != 34", source)

    def test_15_hidden_redirected_process_outlives_invoking_powershell(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stdout = root / "stdout.log"
            stderr = root / "stderr.log"

            def quote(path: Path) -> str:
                return str(path.resolve()).replace("'", "''")

            command = (
                "$p=Start-Process -FilePath 'powershell.exe' "
                "-ArgumentList @('-NoProfile','-Command','Start-Sleep','-Seconds','3') "
                f"-WorkingDirectory '{quote(root)}' "
                f"-WindowStyle Hidden -RedirectStandardOutput '{quote(stdout)}' "
                f"-RedirectStandardError '{quote(stderr)}' -PassThru; $p.Id"
            )
            completed = subprocess.run(
                ["powershell", "-NoProfile", "-Command", command],
                cwd=ROOT,
                text=True,
                capture_output=True,
                timeout=10,
                check=True,
            )
            pid = int(completed.stdout.strip().splitlines()[-1])
            kernel = ctypes.windll.kernel32
            handle = kernel.OpenProcess(0x00101000, False, pid)
            self.assertTrue(handle, completed.stderr)
            try:
                self.assertEqual(kernel.WaitForSingleObject(handle, 0), 0x00000102)
                self.assertEqual(kernel.WaitForSingleObject(handle, 10000), 0x00000000)
            finally:
                kernel.CloseHandle(handle)
            self.assertTrue(stdout.is_file())
            self.assertTrue(stderr.is_file())


if __name__ == "__main__":
    unittest.main()
