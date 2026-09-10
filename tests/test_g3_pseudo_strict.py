import json
from pathlib import Path
import tempfile
import unittest

from scripts.run_g3_pseudo_transfer import (
    DIRECT_BLEND_PREREGISTER,
    _assert_lock_snapshot,
    _create_or_verify_lock,
    _validate_direct_preregister,
)
from src.manifest import sha256_file


class StrictPseudoLockTests(unittest.TestCase):
    def test_lock_is_create_new_or_exact_verify_only(self):
        payload = {"schema_version": 2, "selection": None}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pre2024_lock.json"
            first = _create_or_verify_lock(path, payload)
            first_bytes = path.read_bytes()
            first_mtime = path.stat().st_mtime_ns
            second = _create_or_verify_lock(path, payload)
            self.assertEqual(first, second)
            self.assertEqual(path.read_bytes(), first_bytes)
            self.assertEqual(path.stat().st_mtime_ns, first_mtime)
            with self.assertRaisesRegex(FileExistsError, "immutable"):
                _create_or_verify_lock(
                    path, {"schema_version": 2, "selection": "changed"}
                )
            self.assertEqual(sha256_file(path), first)

    def test_lock_snapshot_detects_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pre2024_lock.json"
            snapshot = _create_or_verify_lock(path, {"schema_version": 2})
            path.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "changed"):
                _assert_lock_snapshot(path, snapshot, phase="unit test")

    def test_direct_preregister_requires_exact_schema_and_rules(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "preregister.json"
            path.write_text(
                json.dumps(DIRECT_BLEND_PREREGISTER, ensure_ascii=False),
                encoding="utf-8",
            )
            self.assertEqual(
                _validate_direct_preregister(path), DIRECT_BLEND_PREREGISTER
            )
            changed = json.loads(path.read_text(encoding="utf-8"))
            changed["selection_rule"]["primary"] = "posthoc_best_2024"
            path.write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exact v2 schema"):
                _validate_direct_preregister(path)


if __name__ == "__main__":
    unittest.main()
