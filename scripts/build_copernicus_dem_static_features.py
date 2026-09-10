"""Build and lock the preregistered Copernicus DEM static terrain table."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.copernicus_dem_exposure import (  # noqa: E402
    EXTENDED_COLUMNS,
    RADII_M,
    SECTOR_CENTRES,
    build_static_terrain_table,
    parse_turbines,
    read_dem_tile,
)
from src.manifest import describe_file, sha256_file, utc_now  # noqa: E402


CONFIG_SHA = "21e2a6aaee02d11c00d2b8a266e30917efa4f5a505417ceadb9f618fe7d2443b"


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temp.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_parquet(temp, index=False)
    os.replace(temp, path)


def _verify(path: Path, *, size: int, sha256: str) -> None:
    if not path.is_file() or path.stat().st_size != size or sha256_file(path) != sha256:
        raise AssertionError(f"source identity differs: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_DIR / "configs/copernicus_dem_directional_exposure_paired_increment_preregister_v1.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_DIR / "artifacts/external/copernicus_dem_glo30_2021_terrain_v1/features_v1",
    )
    args = parser.parse_args()
    config_path = args.config.resolve()
    if sha256_file(config_path) != CONFIG_SHA:
        raise AssertionError("preregister identity differs")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["status"] != "frozen_before_any_raster_value_read_feature_materialization_label_read_candidate_fit_prediction_or_metric":
        raise AssertionError("preregister status differs")
    if args.out_dir.exists():
        raise FileExistsError(args.out_dir)
    for source in (
        config["official_rules_and_source"]["source_feasibility_audit"],
        config["official_rules_and_source"]["source_manifest"],
        config["official_rules_and_source"]["licence"],
        config["official_rules_and_source"]["product_handbook"],
        config["official_rules_and_source"]["public_mirror_readme"],
    ):
        _verify(PROJECT_DIR / source["path"], size=int(source["bytes"]), sha256=str(source["sha256"]))
    tile_specs = config["official_rules_and_source"]["tiles_exactly"]
    if len(tile_specs) != 2:
        raise AssertionError("exactly two tiles are required")
    for spec in tile_specs:
        _verify(PROJECT_DIR / spec["path"], size=int(spec["bytes"]), sha256=str(spec["sha256"]))

    tiles = (
        read_dem_tile(PROJECT_DIR / tile_specs[0]["path"], west=128, south=37),
        read_dem_tile(PROJECT_DIR / tile_specs[1]["path"], west=129, south=37),
    )
    turbines = parse_turbines(config["turbines"]["rows"])
    table = build_static_terrain_table(turbines, tiles)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    table_path = args.out_dir / "static_directional_terrain_17x16.parquet"
    _atomic_parquet(table, table_path)
    table_bytes = table.to_csv(index=False, lineterminator="\n", float_format="%.17g").encode("utf-8")
    semantic_sha = hashlib.sha256(table_bytes).hexdigest()
    audit_path = args.out_dir / "static_feature_audit.json"
    _atomic_json(
        audit_path,
        {
            "schema_version": 1,
            "artifact_type": "copernicus_dem_static_directional_terrain_feature_audit",
            "created_utc": utc_now(),
            "status": "LOCKED_BEFORE_ANY_EXPERIMENT_LABEL_READ_OR_CANDIDATE_FIT",
            "preregister": describe_file(config_path),
            "source_manifest": describe_file(PROJECT_DIR / config["official_rules_and_source"]["source_manifest"]["path"]),
            "tiles": [describe_file(PROJECT_DIR / spec["path"]) for spec in tile_specs],
            "static_table": describe_file(table_path),
            "static_table_semantic_csv_float17_sha256": semantic_sha,
            "rows": len(table),
            "turbines": int(table["turbine_id"].nunique()),
            "sectors": int(table["sector_index"].nunique()),
            "sector_centres": SECTOR_CENTRES.tolist(),
            "radii_m": list(RADII_M),
            "dynamic_feature_names": list(EXTENDED_COLUMNS),
            "finite_numeric_cells": int(table.select_dtypes("number").notna().sum().sum()),
            "nonfinite_registered_terrain_cells": int(table[["signed_slope_alignment", "horizon_angle_r0500m", "horizon_angle_r1000m", "horizon_angle_r2000m", "horizon_angle_r5000m"]].isna().sum().sum()),
            "labels_read": 0,
            "candidate_models_fit": 0,
            "candidate_predictions_materialized": 0,
            "metric_calls": 0,
            "contest_2025_values_read": 0,
        },
    )
    print(json.dumps({"table": describe_file(table_path), "audit": describe_file(audit_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
