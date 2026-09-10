import csv
import gzip
import hashlib
import json
import unittest
import zipfile
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CANONICAL = ROOT / "artifacts" / "external" / "noaa_gefs_operational_spread_00z_census_v2"
PROTOCOL = ROOT / "configs" / "noaa_gefs_operational_spread_00z_census_protocol_v2.json"
RUNNER = ROOT / "scripts" / "census_noaa_gefs_operational_spread_00z_v2.py"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@unittest.skipUnless(CANONICAL.is_dir(), "postrun canonical not present")
class TestNoaaGefsOperationalSpread00zCensusV2Postrun(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.summary = json.loads((CANONICAL / "census_summary.json").read_text(encoding="utf-8"))
        cls.mask = json.loads((CANONICAL / "operating_day_availability.json").read_text(encoding="utf-8"))
        cls.manifest = json.loads((CANONICAL / "manifest.json").read_text(encoding="utf-8"))
        with gzip.open(CANONICAL / "object_metadata.csv.gz", "rt", encoding="utf-8", newline="") as handle:
            cls.rows = list(csv.DictReader(handle))

    def test_01_exact_output_and_source_closure(self) -> None:
        expected_names = {
            "census_summary.json",
            "manifest.json",
            "object_metadata.csv.gz",
            "operating_day_availability.json",
            "official_source_snapshots.zip",
            "required_index_bytes.zip",
        }
        self.assertEqual({path.name for path in CANONICAL.iterdir()}, expected_names)
        identities = {item["path"]: item for item in self.manifest["source_closure"]}
        for path in (PROTOCOL, RUNNER):
            rel = path.relative_to(ROOT).as_posix()
            self.assertEqual(identities[rel]["bytes"], path.stat().st_size)
            self.assertEqual(identities[rel]["sha256"], sha256(path))
        self.assertEqual(
            sha256(PROTOCOL),
            "4860050db45c512d9f5417b5cbcb35aa3153420f03aa2356692e4a48de386bd6",
        )
        for item in self.manifest["artifacts"]:
            path = ROOT / item["path"]
            self.assertEqual(item["bytes"], path.stat().st_size)
            self.assertEqual(item["sha256"], sha256(path))

    def test_02_complete_fixed_universe_and_no_2025(self) -> None:
        self.assertEqual(len(self.rows), 39456)
        self.assertEqual({int(row["lead"]) for row in self.rows}, set(range(39, 64, 3)))
        self.assertEqual(
            {row["product_id"] for row in self.rows},
            {
                "surface_0p25_ens_mean",
                "surface_0p25_ens_spread",
                "pressure_0p50_ens_mean",
                "pressure_0p50_ens_spread",
            },
        )
        self.assertEqual(min(row["source_date"] for row in self.rows), "2021-12-30")
        self.assertEqual(max(row["source_date"] for row in self.rows), "2024-12-29")
        self.assertTrue(all("/00/atmos/" in row["data_url"] for row in self.rows))
        self.assertTrue(all(".t00z." in row["data_key"] for row in self.rows))
        self.assertTrue(all("gefs.2025" not in row["data_key"] for row in self.rows))

    def test_03_row_logic_and_exact_index_binding(self) -> None:
        expected_indices = {}
        for row in self.rows:
            expected = all(
                (
                    row["data_status"] == "200",
                    row["idx_status"] == "200",
                    int(row["data_bytes"]) > 0,
                    bool(row["data_etag"]),
                    bool(row["idx_etag"]),
                    row["idx_content_length_match"] == "True",
                    row["data_cutoff_pass"] == "True",
                    row["idx_cutoff_pass"] == "True",
                    row["semantic_and_range_pass"] == "True",
                )
            )
            self.assertEqual(row["overall_pass"] == "True", expected)
            if row["idx_sha256"]:
                expected_indices[row["idx_key"]] = row["idx_sha256"]
        with zipfile.ZipFile(CANONICAL / "required_index_bytes.zip") as archive:
            self.assertIsNone(archive.testzip())
            self.assertEqual(set(archive.namelist()), set(expected_indices))
            for name in archive.namelist():
                self.assertEqual(hashlib.sha256(archive.read(name)).hexdigest(), expected_indices[name])

    def test_04_whole_day_mask_and_predeclared_gate(self) -> None:
        by_day = defaultdict(list)
        for row in self.rows:
            by_day[row["operating_date"]].append(row)
        self.assertEqual(len(by_day), 1096)
        recomputed = {}
        for day, rows in by_day.items():
            self.assertEqual(len(rows), 36)
            recomputed[day] = all(row["overall_pass"] == "True" for row in rows)
        self.assertEqual(len(self.mask["days"]), 1096)
        for item in self.mask["days"]:
            self.assertEqual(item["available"], recomputed[item["operating_date"]])
            self.assertEqual(item["missing_gefs_00z"], 0 if item["available"] else 1)
            self.assertEqual(
                item["later_paired_increment_rule"],
                "model_candidate" if item["available"] else "exact_zero_identity",
            )
        failed = sum(row["overall_pass"] != "True" for row in self.rows)
        unavailable = sum(not value for value in recomputed.values())
        gate = failed <= 180 and unavailable <= 5
        self.assertEqual(self.summary["coverage"]["source_gate_pass"], gate)
        self.assertEqual(self.summary["coverage"]["failed_objects"], failed)
        self.assertEqual(self.summary["coverage"]["unavailable_operating_days"], unavailable)

    def test_05_safety_and_provenance_closure(self) -> None:
        accounting = self.summary["request_accounting"]
        self.assertEqual(accounting["grib_head_requests"], 39456)
        self.assertEqual(accounting["index_get_requests"], 39456)
        for key in (
            "grib_get_requests",
            "2025_initialization_requests",
            "meteorological_value_bytes_read",
            "label_reads",
            "fits",
            "predictions",
            "scores",
            "csvs",
            "submissions",
        ):
            self.assertEqual(accounting[key], 0)
        self.assertTrue(self.summary["official_provenance"]["pass"])
        self.assertFalse(self.summary["official_provenance"]["nodd_is_official_archive"])
        with zipfile.ZipFile(CANONICAL / "official_source_snapshots.zip") as archive:
            self.assertIsNone(archive.testzip())
            self.assertEqual(len(archive.namelist()), 11)
        self.assertEqual(self.manifest["nonmutation"]["candidate_csvs_modified"], 0)
        self.assertEqual(
            self.manifest["nonmutation"]["labels_models_predictions_metrics_modified_or_created"], 0,
        )


if __name__ == "__main__":
    unittest.main()
