import ast
import hashlib
import importlib.util
import json
import sys
import unittest
from datetime import date, timedelta
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs" / "noaa_gefs_operational_spread_00z_census_protocol_v2.json"
RUNNER = ROOT / "scripts" / "census_noaa_gefs_operational_spread_00z_v2.py"
CANONICAL = ROOT / "artifacts" / "external" / "noaa_gefs_operational_spread_00z_census_v2"
PROTOCOL_SHA256 = "4860050db45c512d9f5417b5cbcb35aa3153420f03aa2356692e4a48de386bd6"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_runner():
    spec = importlib.util.spec_from_file_location("gefs_00z_v2", RUNNER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestNoaaGefsOperationalSpread00zCensusV2(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
        cls.runner = load_runner()

    def test_01_protocol_frozen_before_network_values_or_labels(self) -> None:
        self.assertEqual(sha256(PROTOCOL), PROTOCOL_SHA256)
        self.assertEqual(self.protocol["status"], "FROZEN_BEFORE_FULL_CENSUS_NETWORK_REQUESTS")
        self.assertTrue(all(value == 0 for value in self.protocol["hard_prohibitions"].values()))
        self.assertEqual(self.protocol["predeclared_source_gate"]["expected_objects"], 39456)
        self.assertEqual(self.protocol["predeclared_source_gate"]["maximum_unavailable_operating_days"], 5)
        self.assertEqual(self.protocol["predeclared_source_gate"]["maximum_failed_objects"], 180)

    def test_02_runner_ast_and_fixed_universe(self) -> None:
        source = RUNNER.read_text(encoding="utf-8")
        ast.parse(source)
        self.assertEqual(self.runner.LEADS, tuple(range(39, 64, 3)))
        self.assertEqual(self.runner.EXPECTED_DAY_COUNT, 1096)
        self.assertEqual(self.runner.EXPECTED_OBJECT_COUNT, 39456)
        self.assertEqual(self.runner.MAX_UNAVAILABLE_DAYS, 5)
        self.assertEqual(self.runner.MAX_FAILED_OBJECTS, 180)
        self.assertNotIn("pandas", source)
        self.assertNotIn("sklearn", source)
        self.assertNotIn("lightgbm", source)
        self.assertNotIn("read_parquet", source)

    def test_03_exact_00z_keys_and_no_2025_initialization(self) -> None:
        days = list(self.runner.operating_days())
        self.assertEqual(len(days), 1096)
        self.assertEqual(days[0], date(2022, 1, 1))
        self.assertEqual(days[-1], date(2024, 12, 31))
        keys = [
            self.runner.data_key(day - timedelta(days=2), product, lead)
            for day in days
            for product in self.runner.PRODUCTS
            for lead in self.runner.LEADS
        ]
        self.assertEqual(len(keys), 39456)
        self.assertTrue(all("/00/atmos/" in key and ".t00z." in key for key in keys))
        self.assertTrue(all("gefs.2025" not in key for key in keys))
        self.assertIn("gefs.20211230", keys[0])
        self.assertTrue(any("gefs.20241229" in key for key in keys))

    def test_04_exact_products_exclude_unbound_80m_and_100m(self) -> None:
        self.assertEqual(
            {(item.level, item.statistic) for item in self.runner.PRODUCTS},
            {
                ("10 m above ground", "ens mean"),
                ("10 m above ground", "ens std dev"),
                ("850 mb", "ens mean"),
                ("850 mb", "ens std dev"),
            },
        )
        frozen = json.dumps(self.protocol["fixed_products"], sort_keys=True)
        self.assertNotIn("80 m above ground", frozen)
        self.assertNotIn("100 m above ground", frozen)

    def test_05_index_parser_accepts_only_exact_00z_semantics(self) -> None:
        product = self.runner.PRODUCTS[0]
        payload = (
            "1:0:d=2022010100:TMP:2 m above ground:39 hour fcst:ens mean\n"
            "2:100:d=2022010100:UGRD:10 m above ground:39 hour fcst:ens mean\n"
            "3:200:d=2022010100:VGRD:10 m above ground:39 hour fcst:ens mean\n"
            "4:300:d=2022010100:TMP:surface:39 hour fcst:ens mean\n"
        ).encode("utf-8")
        parsed = self.runner.parse_index(payload, date(2022, 1, 1), product, 39, 400)
        self.assertTrue(parsed["pass"])
        self.assertEqual(parsed["selected"]["UGRD"]["byte_count"], 100)
        self.assertEqual(parsed["selected"]["VGRD"]["byte_count"], 100)
        self.assertFalse(
            self.runner.parse_index(
                payload.replace(b"d=2022010100", b"d=2022010112"),
                date(2022, 1, 1),
                product,
                39,
                400,
            )["pass"]
        )

    def test_06_canonical_is_absent_prerun_or_complete_postrun(self) -> None:
        if CANONICAL.exists():
            self.assertTrue((CANONICAL / "manifest.json").is_file())
            self.assertTrue((CANONICAL / "census_summary.json").is_file())
        else:
            self.assertFalse(CANONICAL.exists())


if __name__ == "__main__":
    unittest.main()
