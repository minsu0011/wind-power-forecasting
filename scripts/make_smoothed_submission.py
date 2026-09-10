"""Create the cross-year-stable within-run smoothed BARAM submission."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.metric import CAPACITY_KWH, TARGET_COLS  # noqa: E402
from src.temporal import smooth_within_runs  # noqa: E402


STRENGTHS = {
    "kpx_group_1": 0.15,
    "kpx_group_2": 0.05,
    "kpx_group_3": 0.0,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    for destination in (args.output, args.manifest):
        if destination.exists() and not args.overwrite:
            raise FileExistsError(f"refusing to overwrite {destination}")

    source = pd.read_csv(args.input, encoding="utf-8-sig")
    required = ("forecast_id", "forecast_kst_dtm", *TARGET_COLS)
    if tuple(source.columns) != required:
        raise ValueError(f"submission columns must be {required}")
    times = pd.DatetimeIndex(pd.to_datetime(source["forecast_kst_dtm"]), name="forecast_kst_dtm")
    prediction = source.loc[:, list(TARGET_COLS)].copy()
    prediction.index = times
    smoothed = smooth_within_runs(prediction, STRENGTHS)
    for group in TARGET_COLS:
        smoothed[group] = np.clip(
            smoothed[group].to_numpy(dtype=float),
            0.0,
            1.02 * CAPACITY_KWH[group],
        )
    output = source.copy()
    output.loc[:, list(TARGET_COLS)] = smoothed.to_numpy(dtype=float)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False, encoding="utf-8-sig", float_format="%.6f")

    observed = pd.read_csv(args.output, encoding="utf-8-sig")
    if not observed[["forecast_id", "forecast_kst_dtm"]].equals(
        source[["forecast_id", "forecast_kst_dtm"]]
    ):
        raise AssertionError("written identifiers or timestamps changed")
    numeric = observed.loc[:, list(TARGET_COLS)].to_numpy(dtype=float)
    if numeric.shape != (8760, 3) or not np.isfinite(numeric).all():
        raise AssertionError("written numeric submission is incomplete")
    if args.output.read_bytes()[:3] != b"\xef\xbb\xbf":
        raise AssertionError("submission is not UTF-8-SIG")

    manifest = {
        "artifact_type": "baram_within_run_smoothed_submission",
        "input": {"path": str(args.input.resolve()), "sha256": _sha256(args.input)},
        "output": {"path": str(args.output.resolve()), "sha256": _sha256(args.output)},
        "strengths": STRENGTHS,
        "formula": "(1-s)*current + s/2*(previous+next), replicated run edges",
        "run_definition": "operating hours 01:00 through next-day 00:00",
        "selection_evidence": {
            "dev2023_g1_g2_score_before": 0.6420910548034591,
            "dev2023_g1_g2_score_after": 0.642659,
            "corrected_gate_score_before": 0.6402501943254041,
            "corrected_gate_score_after": 0.641148,
            "note": "rounded diagnostics; exact source OOF artifacts remain authoritative",
        },
        "leaderboard_score_claimed": False,
        "rows": int(len(observed)),
        "utf8_sig": True,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
