"""Download the two frozen Copernicus DEM GLO-30 2021 terrain tiles.

This is a data-acquisition utility only.  It never opens raster values and it
does not read labels, weather values, predictions, metrics, Public feedback, or
2025 contest data.  The two data objects are the public AWS COG mirror of the
official Copernicus DEM 2021 release.  Official licence and product-handbook
PDFs are archived alongside them for second-stage reproducibility.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "external" / "copernicus_dem_glo30_2021_terrain_v1"
WORK = OUT.with_name(OUT.name + "_staging")

OFFICIAL = {
    "competition_rules": "https://dacon.io/competitions/official/236727/overview/rules",
    "collection_release_and_licensing": "https://dataspace.copernicus.eu/explore-data/data-collections/copernicus-contributing-missions/collections-description/COP-DEM",
    "public_release_announcement": "https://sentinels.copernicus.eu/-/copernicus-dem-30-metre-dataset-now-freely-available",
    "doi": "https://doi.org/10.5270/ESA-c5d3d65",
}

OBJECTS = (
    {
        "role": "terrain_tile",
        "release": "Copernicus DEM GLO-30 2021",
        "relative_path": "tiles/Copernicus_DSM_COG_10_N37_00_E128_00_DEM.tif",
        "url": "https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_N37_00_E128_00_DEM/Copernicus_DSM_COG_10_N37_00_E128_00_DEM.tif",
        "expected_bytes": 45_141_829,
        "expected_etag": "70b5a10cfcfa45d800f3d7d8b31b026f",
    },
    {
        "role": "terrain_tile",
        "release": "Copernicus DEM GLO-30 2021",
        "relative_path": "tiles/Copernicus_DSM_COG_10_N37_00_E129_00_DEM.tif",
        "url": "https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_N37_00_E129_00_DEM/Copernicus_DSM_COG_10_N37_00_E129_00_DEM.tif",
        "expected_bytes": 8_758_373,
        "expected_etag": "3e9c81d0543ad35e91fb045557d9675d",
    },
    {
        "role": "official_licence",
        "relative_path": "docs/copernicus_dem_licences.pdf",
        "url": "https://dataspace.copernicus.eu/sites/default/files/media/files/2025-06/copernicus_contributing_mission_data_access_v2_cop_dem_licenses.pdf",
    },
    {
        "role": "official_product_handbook",
        "relative_path": "docs/copernicus_dem_product_handbook_i5.0.pdf",
        "url": "https://dataspace.copernicus.eu/sites/default/files/media/files/2024-06/geo1988-copernicusdem-spe-002_producthandbook_i5.0.pdf",
    },
    {
        "role": "public_mirror_readme",
        "relative_path": "docs/aws_public_mirror_readme.html",
        "url": "https://copernicus-dem-30m.s3.amazonaws.com/readme.html",
    },
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_one(record: dict[str, object]) -> dict[str, object]:
    destination = WORK / str(record["relative_path"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected_bytes = record.get("expected_bytes")
    if destination.exists():
        if expected_bytes is not None and destination.stat().st_size != expected_bytes:
            raise RuntimeError(f"existing size mismatch: {destination}")
    else:
        request = urllib.request.Request(
            str(record["url"]), headers={"User-Agent": "baram2026-copdem-freeze/1"}
        )
        fd, temporary_name = tempfile.mkstemp(prefix=destination.name + ".", dir=destination.parent)
        os.close(fd)
        temporary = Path(temporary_name)
        try:
            with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as out:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
            if expected_bytes is not None and temporary.stat().st_size != expected_bytes:
                raise RuntimeError(f"downloaded size mismatch: {destination}")
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                temporary.unlink()
    result = dict(record)
    result["bytes"] = destination.stat().st_size
    result["sha256"] = sha256_file(destination)
    return result


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    WORK.mkdir(parents=True, exist_ok=True)
    files = [download_one(record) for record in OBJECTS]
    tile_count = sum(record["role"] == "terrain_tile" for record in files)
    if tile_count != 2:
        raise AssertionError("exactly two terrain tiles are required")
    manifest = {
        "schema_version": 1,
        "artifact_type": "copernicus_dem_glo30_2021_static_terrain_source",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_contract": {
            "product": "COP-DEM_GLO-30-DGED",
            "frozen_release": "2021_1",
            "release_date": "2021-07",
            "public_GLO30_announcement_date": "2020-12-01",
            "static_acquisition_basis": "TanDEM-X base acquired 2010-2015; no time-varying, reanalysis, observed-power, SCADA, or forecast content",
            "coverage": "all 17 registered turbine coordinates plus fixed 5 km terrain neighborhoods",
            "crs": "horizontal EPSG:4326; vertical EGM2008 EPSG:3855",
            "resolution": "1 arc second latitude, approximately 30 m",
            "format": "Cloud Optimized GeoTIFF mirror of DGED",
            "licence": "worldwide, free of charge, reproduction/distribution/adaptation allowed with the prescribed attribution and liability notice",
            "required_attribution": "produced using Copernicus WorldDEM-30 © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA; all rights reserved",
        },
        "official_sources": OFFICIAL,
        "files": files,
        "access_and_leakage": {
            "data_tile_count": tile_count,
            "raster_value_cells_read_by_downloader": 0,
            "label_prediction_metric_Public_or_2025_value_cells_read": 0,
            "remote_model_API_used": False,
            "later_release_or_reanalysis_used": False,
        },
    }
    manifest_path = WORK / "source_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest_sha256 = sha256_file(manifest_path)
    os.replace(WORK, OUT)
    print(json.dumps({
        "output": str(OUT),
        "manifest_sha256": manifest_sha256,
        "files": [{"path": item["relative_path"], "bytes": item["bytes"], "sha256": item["sha256"]} for item in files],
    }, indent=2))


if __name__ == "__main__":
    main()
