"""Create the append-only SHA-256 manifest for breakthrough-v1 outputs/code."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    artifact_root = root / "artifacts/breakthrough_v1"
    output = artifact_root / "MANIFEST_SHA256.csv"
    if output.exists():
        raise FileExistsError(output)

    paths = [path for path in artifact_root.rglob("*") if path.is_file()]
    paths.extend(sorted((root / "src/baram_breakthrough").glob("*.py")))
    paths.extend(
        root / relative
        for relative in (
            "scripts/audit_breakthrough_workspace.py",
            "scripts/run_breakthrough_headroom.py",
            "scripts/audit_breakthrough_final.py",
            "scripts/freeze_breakthrough_manifest.py",
            "tests/test_breakthrough_metric_official_parity.py",
            "tests/test_breakthrough_cutoff.py",
            "tests/test_breakthrough_headroom.py",
            "experiments/scada_wind_teacher_distributional_v1/README.md",
        )
    )
    unique = sorted(set(paths), key=lambda path: path.relative_to(root).as_posix())
    records = [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in unique
    ]
    with output.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("path", "bytes", "sha256"), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    print(f"wrote {len(records)} records to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

