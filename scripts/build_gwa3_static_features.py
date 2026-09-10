"""Build a label-free 17-turbine Global Wind Atlas v3 static lookup.

Only the immutable Version 1 (published 2019-10-09) objects from DTU's
official data.dtu.dk/Figshare record are accepted.  Rasterio performs HTTP
byte-range reads, so this extracts the 17 provided turbine pixels without a
bulk download of the global rasters.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import urllib.request

import pandas as pd
import rasterio


PROJECT_DIR = Path(__file__).resolve().parents[1]
ARTICLE_ID = 9420803
VERSION = 1
API_URL = f"https://api.figshare.com/v2/articles/{ARTICLE_ID}/versions/{VERSION}"
LINEAGE = PROJECT_DIR / "configs/copernicus_dem_directional_exposure_paired_increment_preregister_v1.json"
OUT_DIR = PROJECT_DIR / "artifacts/external/global_wind_atlas_v3_2019_static_fasttrack"

LAYERS = {
    "gwa3_ws100_ms": 17247017,
    "gwa3_weibull_k100": 17602748,
    "gwa3_weibull_a100_ms": 17281466,
    "gwa3_power_density100_wm2": 17263265,
    "gwa3_rix": 17248211,
    "gwa3_cf_iec1": 17281760,
    "gwa3_cf_iec2": 17281778,
    "gwa3_cf_iec3": 17281805,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if OUT_DIR.exists():
        raise FileExistsError(OUT_DIR)
    OUT_DIR.mkdir(parents=True)
    request = urllib.request.Request(API_URL, headers={"User-Agent": "baram2026-research/1.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        metadata = json.load(response)
    if metadata["published_date"] != "2019-10-09T05:31:43Z":
        raise AssertionError("GWA v3 Version 1 publication date differs")
    if metadata["license"]["name"] != "CC BY 4.0":
        raise AssertionError("GWA v3 license differs")
    files = {int(item["id"]): item for item in metadata["files"]}
    if set(LAYERS.values()).difference(files):
        raise AssertionError("registered GWA v3 file ID missing")

    lineage = json.loads(LINEAGE.read_text(encoding="utf-8"))
    rows = pd.DataFrame(lineage["turbines"]["rows"]).rename(columns={"id": "turbine_id"})
    points = list(zip(rows["longitude"].astype(float), rows["latitude"].astype(float)))

    os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
    os.environ.setdefault("VSI_CACHE", "TRUE")
    os.environ.setdefault("VSI_CACHE_SIZE", str(16 << 20))
    source_files = []
    for column, file_id in LAYERS.items():
        item = files[file_id]
        url = str(item["download_url"])
        with rasterio.open(f"/vsicurl/{url}") as dataset:
            if dataset.crs.to_epsg() != 4326 or dataset.count != 1:
                raise AssertionError(f"unexpected raster contract for {column}")
            values = [float(value[0]) for value in dataset.sample(points)]
            raster_contract = {
                "width": int(dataset.width),
                "height": int(dataset.height),
                "dtype": str(dataset.dtypes[0]),
                "nodata": float(dataset.nodata),
                "crs": str(dataset.crs),
                "transform": list(dataset.transform)[:6],
                "block_shape": list(dataset.block_shapes[0]),
            }
        series = pd.Series(values, dtype="float64")
        if series.isna().any() or (series == raster_contract["nodata"]).any():
            raise AssertionError(f"missing GWA values for {column}")
        rows[column] = series
        source_files.append(
            {
                "column": column,
                "file_id": file_id,
                "name": item["name"],
                "download_url": url,
                "bytes": int(item["size"]),
                "official_computed_md5": item["computed_md5"],
                "raster_contract": raster_contract,
            }
        )

    table_path = OUT_DIR / "static_gwa3_17_turbines.parquet"
    rows.to_parquet(table_path, index=False)
    manifest = {
        "schema_version": 1,
        "dataset": "Global Wind Atlas v3",
        "official_record": "https://data.dtu.dk/articles/dataset/Global_Wind_Atlas_v3/9420803",
        "official_version_api": API_URL,
        "article_id": ARTICLE_ID,
        "record_version": VERSION,
        "published_utc": metadata["published_date"],
        "license": metadata["license"],
        "methodology": "ERA5 2008-2017 dynamically downscaled with WRF then WAsP to approximately 250 m",
        "attribution": "Global Wind Atlas version 3, Technical University of Denmark (DTU) and World Bank Group/ESMAP",
        "access_method": "Public official Figshare download URLs; rasterio /vsicurl HTTP byte-range reads at the 17 provided coordinates; no GWA web-app automation.",
        "source_files": source_files,
        "turbine_lineage": {
            "path": str(LINEAGE.relative_to(PROJECT_DIR)).replace("\\", "/"),
            "sha256": sha256_file(LINEAGE),
            "rows": int(len(rows)),
        },
        "static_table": {
            "path": str(table_path.relative_to(PROJECT_DIR)).replace("\\", "/"),
            "bytes": table_path.stat().st_size,
            "sha256": sha256_file(table_path),
            "rows": int(len(rows)),
            "columns": list(rows.columns),
            "missing_cells": int(rows.isna().sum().sum()),
        },
        "label_values_read": 0,
        "prediction_values_read": 0,
        "metric_calls": 0,
    }
    manifest_path = OUT_DIR / "source_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(manifest["static_table"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
