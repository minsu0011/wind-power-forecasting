"""Materialize hourly KMA D-1 11 KST WSD from frozen nine-anchor raw records.

This is deliberately offline: it has no network or label-data code.  It validates
the acquisition chain, summaries, raw gzip bodies, and per-request metadata before
performing the exact V4 same-day float64 interpolation.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


PROJECT = Path(__file__).resolve().parents[1]
CONFIG = PROJECT / "configs" / "kma_d1_1100_wsd_sprint_execution_v2_recovery.json"
MATERIALIZER_CONFIG = PROJECT / "configs" / "kma_d1_1100_wsd_materializer_execution_v2_recovery.json"
DOWNLOADER = PROJECT / "scripts" / "download_kma_d1_1100_wsd_grid_sprint_recovery.py"
ROOT = PROJECT / "artifacts" / "final_submission_sprint_20260812" / "kma_d1_1100" / "wsd_typ01"
RAW_ROOT = ROOT / "raw_gzip"
OUTPUT_ROOT = ROOT / "materialized"

ANCHOR_HOURS = np.asarray([1, 4, 7, 10, 13, 16, 19, 22, 24], dtype=np.float64)
ALL_HOURS = np.arange(1, 25, dtype=np.float64)
CELLS = ((94, 121), (94, 122))
NX = 149
NY = 253
TOKEN_COUNT = NX * NY
ENDPOINT_ID = "KMA_APIHUB_TYP01_NPH_DFS_SHRT_GRD"

MODE = {
    "pregate": {
        "years": (2022, 2023, 2024),
        "records": 9864,
        "rows": 26304,
        "sequence_min": 1,
        "sequence_max": 9864,
        "summary": "PREGATE_2022_2024_SUMMARY.json",
        "summary_mode": "pregate",
        "output": "KMA_D1_1100_WSD_HOURLY_2022_2024.parquet",
        "manifest": "MANIFEST.json",
    },
    "2025": {
        "years": (2025,),
        "records": 3285,
        "rows": 8760,
        "sequence_min": 9865,
        "sequence_max": 13149,
        "summary": "YEAR2025_SUMMARY.json",
        "summary_mode": "year2025",
        "output": "KMA_D1_1100_WSD_HOURLY_2025.parquet",
        "manifest": "MANIFEST_2025.json",
    },
}

SCHEMA = pa.schema(
    [
        pa.field("forecast_kst_dtm", pa.timestamp("ns"), nullable=False),
        pa.field("operating_year", pa.int16(), nullable=False),
        pa.field("operating_day", pa.date32(), nullable=False),
        pa.field("issue_base_kst", pa.timestamp("ns"), nullable=False),
        pa.field("lead_hour", pa.int8(), nullable=False),
        pa.field("wsd_cell_94_121", pa.float64(), nullable=False),
        pa.field("wsd_cell_94_122", pa.float64(), nullable=False),
        pa.field("wsd_kpx_group_1", pa.float64(), nullable=False),
        pa.field("wsd_kpx_group_2", pa.float64(), nullable=False),
        pa.field("wsd_kpx_group_3", pa.float64(), nullable=False),
    ]
)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(path: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(PROJECT).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path.name}")
    return value


def canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True, indent=2) + "\n"
    ).encode("ascii")


def expected_days(year: int) -> int:
    return (date(year + 1, 1, 1) - date(year, 1, 1)).days


def expected_specs(years: Iterable[int]) -> list[dict[str, Any]]:
    sequence = 1
    for prior in (2022, 2023, 2024, 2025):
        if prior in years:
            break
        sequence += expected_days(prior) * 9
    result: list[dict[str, Any]] = []
    for year in years:
        day = date(year, 1, 1)
        stop = date(year + 1, 1, 1)
        while day < stop:
            issue = datetime.combine(day - timedelta(days=1), datetime.min.time()).replace(hour=11)
            for lead in ANCHOR_HOURS.astype(np.int64).tolist():
                forecast = datetime.combine(day, datetime.min.time()) + timedelta(hours=int(lead))
                tmfc = issue.strftime("%Y%m%d%H")
                tmef = forecast.strftime("%Y%m%d%H")
                stem = f"{sequence:05d}__tmfc_{tmfc}__tmef_{tmef}__WSD"
                result.append(
                    {
                        "sequence": sequence,
                        "operating_year": year,
                        "operating_day": day,
                        "lead_hour": int(lead),
                        "forecast": forecast,
                        "issue": issue,
                        "tmfc": tmfc,
                        "tmef": tmef,
                        "stem": stem,
                    }
                )
                sequence += 1
            day += timedelta(days=1)
    return result


def validate_chain() -> list[dict[str, Any]]:
    config = read_json(CONFIG)
    if config.get("schema_version") != 3:
        raise RuntimeError("recovery acquisition execution config schema mismatch")
    downloader_identity = identity(DOWNLOADER)
    if config.get("script_identity") != downloader_identity:
        raise RuntimeError("execution config does not bind current downloader")
    chain = config.get("preregistration_chain")
    if not isinstance(chain, list) or len(chain) != 7:
        raise RuntimeError("exact V1-V7 preregistration chain required")
    verified: list[dict[str, Any]] = []
    for declared in chain:
        path = PROJECT / str(declared["path"])
        actual = identity(path)
        if actual != declared:
            raise RuntimeError(f"preregistration identity mismatch: {path.name}")
        verified.append(actual)
    materializer_config = read_json(MATERIALIZER_CONFIG)
    if materializer_config.get("schema_version") != 2:
        raise RuntimeError("recovery materializer execution config schema mismatch")
    if materializer_config.get("acquisition_execution_config") != identity(CONFIG):
        raise RuntimeError("materializer config does not bind recovery acquisition config")
    if materializer_config.get("materializer") != identity(Path(__file__).resolve()):
        raise RuntimeError("materializer config does not bind current materializer")
    return [identity(MATERIALIZER_CONFIG), identity(CONFIG), downloader_identity, *verified]


def validate_summaries(
    mode: str, spec: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    required = [ROOT / "FIRST100_SUMMARY.json", ROOT / str(spec["summary"])]
    if mode == "2025":
        required.insert(1, ROOT / "PREGATE_2022_2024_SUMMARY.json")
        gate_path = ROOT / "MODEL_GATE_PASS.json"
        gate = read_json(gate_path)
        if gate.get("all_gates_pass") is not True:
            raise RuntimeError("2025 materialization requires all model gates to pass")
        required.append(gate_path)
    first = read_json(required[0])
    if first.get("mode") != "first100" or first.get("records") != 100 or first.get("byte_gate_pass") is not True:
        raise RuntimeError("valid first-100 byte/integrity gate required")
    summary_path = ROOT / str(spec["summary"])
    summary = read_json(summary_path)
    expected = {
        "mode": spec["summary_mode"],
        "records": spec["records"],
        "sequence_min": spec["sequence_min"],
        "sequence_max": spec["sequence_max"],
    }
    for key, value in expected.items():
        if summary.get(key) != value:
            raise RuntimeError(f"acquisition summary mismatch: {key}")
    if summary.get("byte_gate_pass") is not True:
        raise RuntimeError("acquisition summary byte gate did not pass")
    if int(summary.get("projected_grand_calls", -1)) != 13149:
        raise RuntimeError("acquisition grand-call projection mismatch")
    projected = int(summary.get("response_bytes_max", -1)) * 13149
    if int(summary.get("projected_grand_bytes_by_max", -1)) != projected or projected > 4_700_000_000:
        raise RuntimeError("acquisition grand-byte projection mismatch")
    if int(summary.get("recovery_unpersisted_transfer_count", -1)) != 2:
        raise RuntimeError("acquisition recovery-transfer count mismatch")
    if int(summary.get("recovery_unpersisted_response_bytes_known", -1)) != 682594:
        raise RuntimeError("acquisition recovery-byte accounting mismatch")
    adjusted = int(summary.get("response_bytes_max", -1)) * 13151
    if int(summary.get("recovery_adjusted_projected_bytes_by_max", -1)) != adjusted or adjusted > 4_700_000_000:
        raise RuntimeError("acquisition recovery-adjusted projection mismatch")
    if int(summary.get("quota_response_bytes_accounted_current_mode", -1)) != int(summary.get("response_bytes_sum", -1)) + 682594:
        raise RuntimeError("acquisition recovery-adjusted byte census mismatch")
    invalid_sequences = summary.get("invalid_selected_anchor_sequences")
    if not isinstance(invalid_sequences, list) or int(summary.get("invalid_selected_anchor_count", -1)) != len(invalid_sequences):
        raise RuntimeError("acquisition invalid-anchor summary mismatch")
    return [identity(path) for path in required], summary


def float_bits(value: float) -> int:
    return int(np.asarray([value], dtype=np.float64).view(np.uint64)[0])


def parse_selected(payload: bytes) -> tuple[float, float, list[bool], int, bool, int]:
    text = payload.decode("utf-8", errors="strict").strip(" \t\r\n")
    tokens = text.split(",")
    trailing = bool(tokens and tokens[-1] == "")
    if trailing:
        tokens.pop()
    if len(tokens) != TOKEN_COUNT or any(token.strip() == "" for token in tokens):
        raise RuntimeError("invalid national WSD grid tokenization")
    values = np.fromstring(",".join(token.strip() for token in tokens), dtype=np.float64, sep=",")
    if values.shape != (TOKEN_COUNT,) or not np.isfinite(values).all():
        raise RuntimeError("national WSD grid contains unparsable or non-finite values")
    selected: list[float] = []
    valid: list[bool] = []
    for x, y in CELLS:
        value = float(values[(y - 1) * NX + (x - 1)])
        selected.append(value)
        valid.append(bool(np.isfinite(value) and 0.0 <= value <= 75.0))
    return selected[0], selected[1], valid, len(tokens), trailing, int(np.sum(values <= -99.0))


def validate_record(item: dict[str, Any]) -> tuple[float, float, dict[str, Any]]:
    folder = RAW_ROOT / str(item["operating_year"])
    raw_path = folder / f"{item['stem']}.txt.gz"
    meta_path = folder / f"{item['stem']}.json"
    compressed = raw_path.read_bytes()
    payload = gzip.decompress(compressed)
    metadata_bytes = meta_path.read_bytes()
    lowered_metadata = metadata_bytes.lower()
    if b"authkey" in lowered_metadata or b"http://" in lowered_metadata or b"https://" in lowered_metadata:
        raise RuntimeError(f"record {item['sequence']} metadata contains forbidden secret/URL material")
    metadata = json.loads(metadata_bytes.decode("ascii"))
    expected = {
        "sequence": item["sequence"],
        "operating_year": item["operating_year"],
        "forecast_kst": item["forecast"].strftime("%Y-%m-%dT%H:%M:%S+09:00"),
        "tmfc_kst": item["tmfc"],
        "tmef_kst": item["tmef"],
        "variable": "WSD",
        "endpoint_id": ENDPOINT_ID,
        "response_http_status": 200,
        "response_content_type": "text/plain",
        "response_bytes": len(payload),
        "response_sha256": sha256_bytes(payload),
        "gzip_bytes": len(compressed),
        "gzip_sha256": sha256_bytes(compressed),
        "numeric_token_count": TOKEN_COUNT,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise RuntimeError(f"record {item['sequence']} binding mismatch: {key}")
    cell121, cell122, selected_valid, token_count, trailing, sentinel_count = parse_selected(payload)
    if int(item["sequence"]) == 4861:
        if len(payload) != 341297 or sha256_bytes(payload) != "2ff9e1ab475105d61e135c518a90a6f1493c94c8f83abcb297c0c8eb64f8aea9":
            raise RuntimeError("sequence 4861 frozen diagnostic identity mismatch")
        if sentinel_count != 37697 or cell121 != -99.0 or cell122 != -99.0 or selected_valid != [False, False]:
            raise RuntimeError("sequence 4861 frozen diagnostic semantics mismatch")
    if token_count != metadata.get("numeric_token_count"):
        raise RuntimeError(f"record {item['sequence']} token count mismatch")
    if trailing != metadata.get("one_terminal_comma_present"):
        raise RuntimeError(f"record {item['sequence']} terminal-comma mismatch")
    if sentinel_count != metadata.get("sentinel_le_minus_99_count"):
        raise RuntimeError(f"record {item['sequence']} sentinel-count mismatch")
    if float_bits(cell121) != float_bits(float(metadata.get("wsd_cell_94_121"))):
        raise RuntimeError(f"record {item['sequence']} cell 94,121 mismatch")
    if float_bits(cell122) != float_bits(float(metadata.get("wsd_cell_94_122"))):
        raise RuntimeError(f"record {item['sequence']} cell 94,122 mismatch")
    if "selected_cells_valid_0_75" in metadata and metadata["selected_cells_valid_0_75"] != selected_valid:
        raise RuntimeError(f"record {item['sequence']} selected-cell validity mismatch")
    if int(item["sequence"]) >= 4861:
        if metadata.get("selected_cells_valid_0_75") != selected_valid:
            raise RuntimeError(f"record {item['sequence']} recovery validity list missing or mismatched")
        if metadata.get("selected_cells_all_valid_0_75") != bool(all(selected_valid)):
            raise RuntimeError(f"record {item['sequence']} recovery validity aggregate missing or mismatched")
    lineage = {
        "sequence": item["sequence"],
        "tmfc": item["tmfc"],
        "tmef": item["tmef"],
        "response_bytes": len(payload),
        "response_sha256": expected["response_sha256"],
        "gzip_bytes": len(compressed),
        "gzip_sha256": expected["gzip_sha256"],
        "metadata_bytes": len(metadata_bytes),
        "metadata_sha256": sha256_bytes(metadata_bytes),
    }
    lineage["selected_cells_valid_0_75"] = selected_valid
    return cell121, cell122, selected_valid, lineage


def validate_census(years: Iterable[int], expected: list[dict[str, Any]]) -> None:
    expected_names: dict[int, set[str]] = {year: set() for year in years}
    for item in expected:
        expected_names[item["operating_year"]].update(
            {f"{item['stem']}.txt.gz", f"{item['stem']}.json"}
        )
    for year, names in expected_names.items():
        folder = RAW_ROOT / str(year)
        if not folder.is_dir():
            raise RuntimeError(f"raw year folder missing: {year}")
        actual = {path.name for path in folder.iterdir() if path.is_file()}
        if actual != names:
            missing = len(names - actual)
            extra = len(actual - names)
            raise RuntimeError(f"raw census mismatch for {year}: missing={missing}, extra={extra}")


def interpolate_day(anchors: np.ndarray, validity: np.ndarray) -> np.ndarray:
    values = np.asarray(anchors, dtype=np.float64)
    valid = np.asarray(validity, dtype=bool)
    if values.shape != (9, 2) or valid.shape != (9, 2):
        raise RuntimeError("nine-anchor shape mismatch")
    hourly = np.full((24, 2), np.nan, dtype=np.float64)
    anchor_positions = ANCHOR_HOURS.astype(np.int64) - 1
    for cell_index in range(2):
        if not valid[:, cell_index].all():
            continue
        if not np.isfinite(values[:, cell_index]).all() or np.any(values[:, cell_index] < 0.0) or np.any(values[:, cell_index] > 75.0):
            raise RuntimeError("valid anchor classification disagrees with decoded values")
        hourly[:, cell_index] = np.interp(ALL_HOURS, ANCHOR_HOURS, values[:, cell_index])
        hourly[anchor_positions, cell_index] = values[:, cell_index]
        if not np.array_equal(
            hourly[anchor_positions, cell_index].view(np.uint64),
            values[:, cell_index].view(np.uint64),
        ):
            raise RuntimeError("anchor bitwise-preservation check failed")
    return hourly


def materialize(mode: str) -> tuple[pa.Table, dict[str, Any]]:
    spec = MODE[mode]
    chain = validate_chain()
    summaries, acquisition_summary = validate_summaries(mode, spec)
    expected = expected_specs(spec["years"])
    if len(expected) != spec["records"]:
        raise RuntimeError("internal request census mismatch")
    validate_census(spec["years"], expected)

    columns: dict[str, list[Any]] = {field.name: [] for field in SCHEMA}
    lineage_digest = hashlib.sha256()
    response_bytes_sum = 0
    response_bytes_max = 0
    metadata_bytes_sum = 0
    invalid_operating_day_cells: list[dict[str, Any]] = []
    invalid_anchor_sequences: list[int] = []
    for offset in range(0, len(expected), 9):
        day_specs = expected[offset : offset + 9]
        if len(day_specs) != 9 or [x["lead_hour"] for x in day_specs] != ANCHOR_HOURS.astype(int).tolist():
            raise RuntimeError("same-day nine-anchor grouping failure")
        if len({x["operating_day"] for x in day_specs}) != 1:
            raise RuntimeError("interpolation may not cross operating-day boundaries")
        anchors = np.empty((9, 2), dtype=np.float64)
        validity = np.empty((9, 2), dtype=bool)
        for anchor_index, item in enumerate(day_specs):
            anchors[anchor_index, 0], anchors[anchor_index, 1], selected_valid, lineage = validate_record(item)
            validity[anchor_index, :] = selected_valid
            if not all(selected_valid):
                invalid_anchor_sequences.append(int(item["sequence"]))
            lineage_digest.update(canonical_json(lineage))
            response_bytes_sum += int(lineage["response_bytes"])
            response_bytes_max = max(response_bytes_max, int(lineage["response_bytes"]))
            metadata_bytes_sum += int(lineage["metadata_bytes"])
        for cell_index, cell_name in enumerate(("94_121", "94_122")):
            invalid_anchor_leads = [
                int(day_specs[index]["lead_hour"])
                for index in range(9)
                if not bool(validity[index, cell_index])
            ]
            if invalid_anchor_leads:
                invalid_operating_day_cells.append({
                    "operating_day": day_specs[0]["operating_day"].isoformat(),
                    "cell": cell_name,
                    "invalid_anchor_leads": invalid_anchor_leads,
                    "hourly_values_set_to_nan": 24,
                })
        hourly = interpolate_day(anchors, validity)
        operating_day = day_specs[0]["operating_day"]
        issue = day_specs[0]["issue"]
        for lead in range(1, 25):
            forecast = datetime.combine(operating_day, datetime.min.time()) + timedelta(hours=lead)
            cell121 = float(hourly[lead - 1, 0])
            cell122 = float(hourly[lead - 1, 1])
            columns["forecast_kst_dtm"].append(forecast)
            columns["operating_year"].append(operating_day.year)
            columns["operating_day"].append(operating_day)
            columns["issue_base_kst"].append(issue)
            columns["lead_hour"].append(lead)
            columns["wsd_cell_94_121"].append(cell121)
            columns["wsd_cell_94_122"].append(cell122)
            columns["wsd_kpx_group_1"].append(0.5 * cell121 + 0.5 * cell122)
            columns["wsd_kpx_group_2"].append((5.0 / 6.0) * cell121 + (1.0 / 6.0) * cell122)
            columns["wsd_kpx_group_3"].append(cell121)

    table = pa.Table.from_pydict(columns, schema=SCHEMA)
    if response_bytes_sum != int(acquisition_summary.get("response_bytes_sum", -1)):
        raise RuntimeError("raw response-byte census does not match acquisition summary")
    if response_bytes_max != int(acquisition_summary.get("response_bytes_max", -1)):
        raise RuntimeError("raw response-byte maximum does not match acquisition summary")
    observed_invalid_sequences = sorted(set(invalid_anchor_sequences))
    if observed_invalid_sequences != sorted(int(value) for value in acquisition_summary.get("invalid_selected_anchor_sequences", [])):
        raise RuntimeError("raw invalid-anchor sequences do not match acquisition summary")
    if table.num_rows != spec["rows"] or any(column.null_count for column in table.columns):
        raise RuntimeError("hourly row count or Arrow-null contract failed")
    coverage: dict[str, Any] = {}
    operating_years = np.asarray(columns["operating_year"], dtype=np.int16)
    leads = np.asarray(columns["lead_hour"], dtype=np.int8)
    for year in spec["years"]:
        year_mask = operating_years == year
        cell121 = np.asarray(columns["wsd_cell_94_121"], dtype=np.float64)
        cell122 = np.asarray(columns["wsd_cell_94_122"], dtype=np.float64)
        annual_121 = float(np.mean(np.isfinite(cell121[year_mask])))
        annual_122 = float(np.mean(np.isfinite(cell122[year_mask])))
        per_lead_121 = {
            str(lead): float(np.mean(np.isfinite(cell121[year_mask & (leads == lead)]))) for lead in range(1, 25)
        }
        per_lead_122 = {
            str(lead): float(np.mean(np.isfinite(cell122[year_mask & (leads == lead)]))) for lead in range(1, 25)
        }
        gate_pass = min(annual_121, annual_122, *per_lead_121.values(), *per_lead_122.values()) >= 0.995
        coverage[str(year)] = {
            "rows": int(year_mask.sum()),
            "cell_94_121_annual_finite_fraction": annual_121,
            "cell_94_122_annual_finite_fraction": annual_122,
            "per_lead_rows": {str(lead): int(np.sum(year_mask & (leads == lead))) for lead in range(1, 25)},
            "cell_94_121_per_lead_finite_fraction": per_lead_121,
            "cell_94_122_per_lead_finite_fraction": per_lead_122,
            "coverage_gate_minimum": 0.995,
            "coverage_gate_pass": gate_pass,
        }
        if not gate_pass:
            raise RuntimeError(f"annual or per-lead KMA WSD coverage gate failed: {year}")
    evidence = {
        "chain_identities": chain,
        "summary_identities": summaries,
        "raw_record_count": len(expected),
        "raw_response_bytes_sum": response_bytes_sum,
        "raw_response_bytes_max": response_bytes_max,
        "metadata_bytes_sum": metadata_bytes_sum,
        "ordered_raw_metadata_lineage_sha256": lineage_digest.hexdigest(),
        "invalid_operating_day_cell_count": len(invalid_operating_day_cells),
        "invalid_operating_day_cells": invalid_operating_day_cells,
        "coverage": coverage,
    }
    return table, evidence


def publish(mode: str, table: pa.Table, evidence: dict[str, Any]) -> tuple[Path, Path]:
    spec = MODE[mode]
    output_path = OUTPUT_ROOT / str(spec["output"])
    manifest_path = OUTPUT_ROOT / str(spec["manifest"])
    output_temp = output_path.with_name(output_path.name + ".partial")
    manifest_temp = manifest_path.with_name(manifest_path.name + ".partial")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for path in (output_path, manifest_path, output_temp, manifest_temp):
        if path.exists():
            raise RuntimeError(f"refusing to overwrite materialization artifact: {path.name}")
    with output_temp.open("xb") as handle:
        pq.write_table(
            table,
            handle,
            version="2.6",
            compression="zstd",
            compression_level=3,
            use_dictionary=False,
            write_statistics=True,
            data_page_version="1.0",
            row_group_size=65536,
        )
        handle.flush()
        os.fsync(handle.fileno())
    os.rename(output_temp, output_path)
    parquet_identity = identity(output_path)
    manifest = {
        "schema_version": 1,
        "artifact_type": "KMA_D1_1100_WSD_NINE_ANCHOR_HOURLY_MATERIALIZATION",
        "mode": mode,
        "network_access": False,
        "label_access": False,
        "interpolation": {
            "dtype": "float64",
            "implementation": "numpy.interp",
            "anchor_lead_hours": ANCHOR_HOURS.astype(int).tolist(),
            "hourly_lead_hours": ALL_HOURS.astype(int).tolist(),
            "same_operating_day_and_cell_only": True,
            "anchor_outputs_bitwise_identical": True,
            "extrapolation": False,
        },
        "rows": table.num_rows,
        "columns_exact_order": table.column_names,
        "arrow_schema": str(table.schema),
        "parquet": parquet_identity,
        "materializer": identity(Path(__file__).resolve()),
        "source_evidence": evidence,
        "secret_persisted": False,
    }
    manifest_payload = canonical_json(manifest)
    with manifest_temp.open("xb") as handle:
        handle.write(manifest_payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.rename(manifest_temp, manifest_path)
    return output_path, manifest_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", choices=("pregate", "2025"), required=True)
    args = parser.parse_args()
    table, evidence = materialize(args.years)
    output_path, manifest_path = publish(args.years, table, evidence)
    print(
        json.dumps(
            {
                "mode": args.years,
                "rows": table.num_rows,
                "parquet": output_path.relative_to(PROJECT).as_posix(),
                "manifest": manifest_path.relative_to(PROJECT).as_posix(),
                "network_access": False,
                "label_access": False,
            },
            ensure_ascii=True,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
