import ast
import csv
import gzip
import hashlib
import json
import re
import unittest
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs" / "noaa_gefs_operational_spread_upperair_census_protocol_v1.json"
AUDIT = ROOT / "artifacts" / "audits" / "noaa_gefs_operational_spread_upperair_census_v1_no_go.json"
INCIDENT = ROOT / "artifacts" / "incidents" / "noaa_gefs_operational_spread_upperair_census_v1_export_literal.json"
RUNNER = ROOT / "scripts" / "census_noaa_gefs_operational_spread_upperair_v1.py"
CANONICAL = ROOT / "artifacts" / "external" / "noaa_gefs_operational_spread_upperair_census_v1"
QUARANTINE = ROOT / "artifacts" / "external" / "noaa_gefs_operational_spread_upperair_census_v1_failed_attempt1_export_literal"
METADATA = QUARANTINE / "object_metadata.csv.gz"
INDICES = QUARANTINE / "required_index_bytes.zip"
SNAPSHOTS = QUARANTINE / "official_source_snapshots.zip"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class TestNoaaGefsOperationalCensusV1(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
        cls.audit = json.loads(AUDIT.read_text(encoding="utf-8"))
        cls.incident = json.loads(INCIDENT.read_text(encoding="utf-8"))
        with gzip.open(METADATA, "rt", encoding="utf-8", newline="") as handle:
            cls.rows = list(csv.DictReader(handle))

    def test_01_frozen_protocol_and_no_candidate(self) -> None:
        self.assertEqual(sha256(PROTOCOL), "8bc6f637bd081dd05699d89f9829144e4d3527aae47091d6462403276142d1c0")
        self.assertEqual(self.protocol["status"], "FROZEN_BEFORE_FULL_CENSUS_NETWORK_REQUESTS")
        self.assertEqual(self.audit["verdict"], "NO_GO_CUTOFF_AND_IMMUTABLE_AS_ISSUED_PROVENANCE_FAILURE")
        self.assertFalse(self.audit["decision"]["model_preregister_created"])
        self.assertFalse(self.audit["decision"]["label_fit_prediction_score_or_csv_allowed"])
        self.assertEqual(self.audit["final_closure"]["stage1"], "NOT_RUN")
        self.assertIsNone(self.audit["final_closure"]["model_preregister_sha256"])

    def test_02_physical_identities_and_quarantine(self) -> None:
        expected = {
            METADATA: (6856714, "e494ab148592e643eb003361242e16c56b5856c6fa2ee22d9c99868f371a84c2"),
            INDICES: (47146873, "1f0c61a1ca2941650ff656644544424110728a7b324c309cf7129ccee63002d9"),
            SNAPSHOTS: (485233, "5584d820f18036f56274062c8b72d2427954fac7d87a7721b33d9cb3f71ef004"),
            INCIDENT: (3914, "2b25fcc760816615810e498ba92008a0274336d25e92013280a9fef3051ca9c3"),
            AUDIT: (12152, "526be37245078ac230391695f0327a488b6808911ea0ebd06b383cc0e3c1e73e"),
        }
        self.assertFalse(CANONICAL.exists())
        self.assertTrue(QUARANTINE.is_dir())
        for path, (size, expected_sha) in expected.items():
            self.assertEqual(path.stat().st_size, size)
            self.assertEqual(sha256(path), expected_sha)

    def test_03_complete_metadata_and_no_2025_initialization(self) -> None:
        self.assertEqual(len(self.rows), 43840)
        self.assertTrue(all(row["data_status"] == "200" and row["idx_status"] == "200" for row in self.rows))
        self.assertTrue(all(row["semantic_and_range_pass"] == "True" for row in self.rows))
        self.assertTrue(all(row["idx_content_length_match"] == "True" for row in self.rows))
        self.assertTrue(all(row["index_offsets_strict"] == "True" for row in self.rows))
        self.assertTrue(all(row["index_offsets_in_bounds"] == "True" for row in self.rows))
        self.assertTrue(all("gefs.2025" not in row["data_key"] and "gefs.2025" not in row["idx_key"] for row in self.rows))
        self.assertEqual(min(row["source_date"] for row in self.rows), "2021-12-30")
        self.assertEqual(max(row["source_date"] for row in self.rows), "2024-12-29")
        self.assertTrue(all(not row["data_version_id"] for row in self.rows))
        self.assertTrue(all(not row["data_checksum_sha256_header"] for row in self.rows))
        self.assertEqual(sum(int(row["u_byte_count"]) + int(row["v_byte_count"]) for row in self.rows), 39726331347)

    def test_04_exact_cutoff_failure_set(self) -> None:
        self.assertEqual(sum(row["data_cutoff_pass"] == "True" for row in self.rows), 43820)
        self.assertEqual(sum(row["idx_cutoff_pass"] == "True" for row in self.rows), 43780)
        failures = [row for row in self.rows if row["overall_pass"] != "True"]
        self.assertEqual(len(failures), 60)
        first = [row for row in failures if row["operating_date"] == "2022-10-20"]
        second = [row for row in failures if row["operating_date"] == "2024-04-18"]
        self.assertEqual(len(first), 20)
        self.assertEqual({row["product_id"] for row in first}, {"surface_0p25_ens_mean", "surface_0p25_ens_spread"})
        self.assertTrue(all(row["data_cutoff_pass"] == "False" and row["idx_cutoff_pass"] == "False" for row in first))
        self.assertEqual(len(second), 40)
        self.assertEqual({row["product_id"] for row in second}, {
            "surface_0p25_ens_mean", "surface_0p25_ens_spread",
            "pressure_0p50_ens_mean", "pressure_0p50_ens_spread",
        })
        self.assertTrue(all(row["data_cutoff_pass"] == "True" and row["idx_cutoff_pass"] == "False" for row in second))
        self.assertEqual({int(row["lead"]) for row in first}, set(range(27, 55, 3)))
        self.assertEqual({int(row["lead"]) for row in second}, set(range(27, 55, 3)))

    def test_05_every_exact_index_payload_bound(self) -> None:
        expected = {row["idx_key"]: row["idx_sha256"] for row in self.rows}
        self.assertEqual(len(expected), 43840)
        with zipfile.ZipFile(INDICES) as archive:
            self.assertIsNone(archive.testzip())
            names = archive.namelist()
            self.assertEqual(len(names), 43840)
            self.assertEqual(set(names), set(expected))
            for name in names:
                self.assertEqual(hashlib.sha256(archive.read(name)).hexdigest(), expected[name])

    def test_06_official_snapshot_bytes_and_markers(self) -> None:
        expected = {
            "ncei_gefs.html": (38928, "6f703773711a068e71b2023e9f378c392fb2b0798a98164665ca24ce9032dc8f", ["not officially archived", "1/1/2017"]),
            "ncei_nodd.html": (39093, "d86429eba47a13983464d9ab3b0ae5327e7fa9c2d5850ed7c074200dbbe6cc29", ["Global Ensemble Forecast System"]),
            "emc_gefs.html": (103890, "915bbad82e0e0ff69ba2a7bae2c0d8ce30e7b0c8c297bc102e0c04e207d7e04d", ["31 members", "September 23, 2020"]),
            "aws_registry_gefs.html": (8861, "c1e2ebf1f673b599ca026bde45c8c1f9e1d907bde189b0e74c50fef57b34f4b9", ["open to the public", "used as desired", "noaa-gefs-pds"]),
            "nco_geavg_0p50_inventory.html": (28288, "fab4d0016129068982055e08a61933298d135dffbd63be6e52bb5d861bbb978f", ["850 mb", "UGRD", "10 m above ground", "ens mean"]),
            "nco_gespr_0p50_inventory.html": (28543, "bdd986af4cd2858ac0367815c96d1c4fbe7efc833343970629c1e983e5b602ba", ["850 mb", "UGRD", "10 m above ground", "ens std dev"]),
            "nco_gespr_0p25_inventory.html": (12665, "9f5361c2433bbf44a6f81d97f35b1e5c5e609006b50e2b00b88624b1a333f7cc", ["10 m above ground", "UGRD", "ens std dev"]),
            "dacon_rules.html": (933602, "3f5593ca31b98d05d07b8c85015745e2ed9cb6fe2b9d5ffcaa8f890269c45fc8", ["14:00"]),
            "dacon_evaluation.html": (1004308, "61cd86888c7fd202f55a146557a345ff8921162b2d981daef9d72f4dc16e7b33", ["Public Score", "40%"]),
        }
        with zipfile.ZipFile(SNAPSHOTS) as archive:
            self.assertIsNone(archive.testzip())
            self.assertEqual(set(archive.namelist()), set(expected))
            for name, (size, expected_sha, markers) in expected.items():
                payload = archive.read(name)
                text = payload.decode("utf-8", errors="replace").lower()
                self.assertEqual(len(payload), size)
                self.assertEqual(hashlib.sha256(payload).hexdigest(), expected_sha)
                for marker in markers:
                    self.assertIn(marker.lower(), text)

    def test_07_runner_amendment_is_source_only_and_parseable(self) -> None:
        source = RUNNER.read_text(encoding="utf-8")
        ast.parse(source)
        self.assertIsNone(re.search(r"\b(?:false|true|null)\b", source))
        self.assertEqual(sha256(RUNNER), "76990430dadbcfaf5e512d8ac3708ef365879113246f7ab35267a1f74d511c4f")
        self.assertEqual(self.incident["executed_source"]["sha256"], "dad949e5a97bbb9ef81df0d34db3624fcecaf0febd5e4a9bab79626a554f0267")
        self.assertFalse(self.incident["source_only_amendment"]["network_rerun"])


if __name__ == "__main__":
    unittest.main()
