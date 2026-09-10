from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = PROJECT / "artifacts" / "baram2026_ncei_scada_research_20260810_210756"
V2 = PROJECT / "artifacts" / "baram2026_evidence_first_v2_20260810_160144" / "phase_a"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_text_new(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    path.write_text(text, encoding="utf-8", newline="\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    out = args.root.resolve() / "00_evidence"
    out.mkdir(parents=True, exist_ok=True)

    registry_path = V2 / "PUBLIC_SHA_REGISTRY_VERIFIED.csv"
    nodes_path = V2 / "SUBMISSION_LINEAGE_NODES.parquet"
    audit_path = V2 / "PHASE_A_COMPLETION_AUDIT_TIMESTAMP_AMENDMENT.json"
    for path in (registry_path, nodes_path, audit_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    registry = pd.read_csv(registry_path)
    nodes = pd.read_parquet(nodes_path)
    public = registry.loc[registry["claim_class"].eq("PUBLIC_SHA_LINKED")].copy()
    if len(public) != 9 or public["arithmetic_error"].abs().max() > 1e-9:
        raise AssertionError("sealed Public registry is not exact")
    if "score_verified" in public.columns and not public["score_verified"].all():
        raise AssertionError("a Public registry score is not verified")
    public.to_csv(out / "PUBLIC_SHA_LEDGER.csv", index=False, lineterminator="\n")

    cluster_cols = [
        "prediction_hash_full", "csv_name", "csv_sha256", "csv_path",
        "mechanism_family", "independent_axis", "claim_class", "score",
    ]
    clusters = (
        nodes.sort_values(["prediction_hash_full", "csv_path"])
        .groupby("prediction_hash_full", sort=True, dropna=False)
        .agg(
            file_count=("csv_path", "size"),
            representative_csv=("csv_name", "first"),
            representative_sha256=("csv_sha256", "first"),
            representative_path=("csv_path", "first"),
            mechanism_family=("mechanism_family", "first"),
            independent_axis=("independent_axis", "first"),
            public_score=("score", "max"),
        )
        .reset_index()
    )
    clusters.to_csv(out / "UNIQUE_PREDICTION_CLUSTERS.csv", index=False, lineterminator="\n")

    axes = (
        nodes.groupby("independent_axis", dropna=False)
        .agg(
            csv_path_count=("csv_path", "size"),
            unique_csv_sha=("csv_sha256", "nunique"),
            unique_prediction_hash=("prediction_hash_full", "nunique"),
            public_linked_count=("claim_class", lambda x: int((x == "PUBLIC_SHA_LINKED").sum())),
            best_linked_score=("score", "max"),
        )
        .reset_index()
        .sort_values("independent_axis")
    )
    axes.to_csv(out / "EXPERIMENT_AXIS_LEDGER.csv", index=False, lineterminator="\n")

    champion = public.sort_values("score", ascending=False).iloc[0]
    matching = nodes.loc[nodes["csv_sha256"].eq(champion["sha256"])]
    if matching.empty:
        raise AssertionError("champion SHA is absent from prediction lineage")
    node = matching.iloc[0]
    reconstruction = {
        "schema_version": 1,
        "claim_class": "PUBLIC_SHA_LINKED",
        "champion_file": champion["csv_name"],
        "champion_sha256": champion["sha256"],
        "champion_public_score": float(champion["score"]),
        "champion_N": float(champion["one_minus_nmae"]),
        "champion_FICR": float(champion["ficr"]),
        "prediction_hash": node["prediction_hash_full"],
        "unique_prediction_cluster": int(
            clusters.index[clusters["prediction_hash_full"].eq(node["prediction_hash_full"])][0]
        ),
        "experiment_axis": node["independent_axis"],
        "platform_receipt_verified": False,
        "selector_eligible": False,
        "source_inputs": [
            {"path": str(registry_path), "bytes": registry_path.stat().st_size, "sha256": sha256(registry_path)},
            {"path": str(nodes_path), "bytes": nodes_path.stat().st_size, "sha256": sha256(nodes_path)},
            {"path": str(audit_path), "bytes": audit_path.stat().st_size, "sha256": sha256(audit_path)},
        ],
    }
    write_text_new(out / "CHAMPION_RECONSTRUCTION.json", json.dumps(reconstruction, indent=2) + "\n")
    write_text_new(
        out / "CHAMPION_RECONSTRUCTION.md",
        "# Champion reconstruction\n\n"
        f"- Claim: `PUBLIC_SHA_LINKED`\n"
        f"- CSV: `{reconstruction['champion_file']}`\n"
        f"- SHA-256: `{reconstruction['champion_sha256']}`\n"
        f"- Score / N / FICR: `{reconstruction['champion_public_score']:.10f}` / "
        f"`{reconstruction['champion_N']:.10f}` / `{reconstruction['champion_FICR']:.10f}`\n"
        f"- Prediction hash: `{reconstruction['prediction_hash']}`\n"
        "- Platform receipt: not independently verified.\n"
        "- Public score is evidence only and is never a model selector.\n",
    )
    print(json.dumps({"status": "PASS", "public": len(public), "clusters": len(clusters), "axes": len(axes), "champion": reconstruction["champion_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
