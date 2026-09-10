"""Read-only integrity audit for the completed breakthrough-v1 NO-GO study."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


EXPECTED: dict[str, str] = {
    "00_inventory/integrity_audit.json": "",
    "01_novelty/DUPLICATE_AND_NOVELTY_MATRIX.md": "12487fb0fa3de9a93a76bf50bfc96a7b7fea2be7e47e1401d7d403580feb095c",
    "02_preregister/PREREGISTRATION_v2.json": "6442149cd40eaf45920aeb610b2e4a9089f55ae2aacb42677a89ce1b9580c5b1",
    "03_headroom/ORACLE_HEADROOM.json": "6a04829eeeb69857c018c5e28e2468bd806f64200e8f9a433303da1227265a85",
    "03_headroom/BOUNDARY_ENERGY_CENSUS.parquet": "b0e6591115c5a3cee12f71be946e7a6add96850b487a36a3efc9b635bbc08ce3",
    "03_headroom/OBSERVABLE_PREDICTABILITY.json": "4b61563b2e7c2565451fbf9a7938b79155d55b1e4250a1acc944c8950b879511",
    "03_headroom/INDEPENDENT_STAGE3_AUDIT.json": "74ce0b883ecff4f22c6a9ac0bcd35173616a9738f7297b177c8f701511d00909",
}


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
    records: list[dict[str, Any]] = []
    failures: list[str] = []
    for relative, expected_hash in EXPECTED.items():
        path = artifact_root / relative
        if not path.is_file():
            failures.append(f"missing:{relative}")
            continue
        actual_hash = sha256(path)
        if expected_hash and actual_hash != expected_hash:
            failures.append(f"sha256:{relative}")
        records.append({"path": relative, "bytes": path.stat().st_size, "sha256": actual_hash})

    integrity = json.loads((artifact_root / "00_inventory/integrity_audit.json").read_text(encoding="utf-8"))
    oracle = json.loads((artifact_root / "03_headroom/ORACLE_HEADROOM.json").read_text(encoding="utf-8"))
    observable = json.loads((artifact_root / "03_headroom/OBSERVABLE_PREDICTABILITY.json").read_text(encoding="utf-8"))
    independent = json.loads((artifact_root / "03_headroom/INDEPENDENT_STAGE3_AUDIT.json").read_text(encoding="utf-8"))
    if not integrity.get("pass"):
        failures.append("stage0_integrity")
    if not oracle.get("action_oracle_gate_pass"):
        failures.append("unexpected_action_oracle_gate")
    if observable.get("gate_pass"):
        failures.append("unexpected_observable_gate")
    verdict_text = json.dumps(independent, sort_keys=True)
    if "NO-GO" not in verdict_text and "NO_GO" not in verdict_text:
        failures.append("independent_no_go_missing")

    csvs = sorted(artifact_root.rglob("*.csv"))
    submission_csvs = [path for path in csvs if "MANIFEST" not in path.name and "LEDGER" not in path.name and "MATRIX" not in path.name]
    if submission_csvs:
        failures.append("unexpected_submission_csv")
    report = {
        "pass": not failures,
        "failures": failures,
        "records": records,
        "action_oracle_delta": oracle["macro"]["action_set_oracle_delta"],
        "observable_gate_pass": observable["gate_pass"],
        "submission_csv_count": len(submission_csvs),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())

