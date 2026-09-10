"""Reproduce the locked dev-2023 ensemble without reading any 2024 gate labels.

Only six explicitly named dev2023 prediction artifacts and the existing locked
prediction reference are opened. No label file is read and no parameter search
is performed. The script exits non-zero if index alignment, group names, the
time boundary, or the locked numerical recipe changes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.postprocess import (  # noqa: E402
    FICRBoundedPostprocessor,
    GroupAffineCalibrator,
    SimplexEnsembler,
)


GROUPS = ("kpx_group_1", "kpx_group_2")
CAPACITIES = {"kpx_group_1": 21_600.0, "kpx_group_2": 21_600.0}
MODEL_FILES = {
    "lgb_l1_n1500": "dev2023_lgb_l1_eligible_n1500.parquet",
    "q06": "dev2023_lgb_q06_eligible.parquet",
    "q07": "dev2023_lgb_q07_eligible.parquet",
    "xgb_l1": "dev2023_xgb_l1_eligible.parquet",
    "cat_mae": "dev2023_cat_mae_eligible.parquet",
    "extra": "dev2023_extra_eligible.parquet",
}
LOCKED_WEIGHTS = {
    "kpx_group_1": {
        "lgb_l1_n1500": 0.460,
        "q06": 0.045,
        "q07": 0.220,
        "xgb_l1": 0.145,
        "cat_mae": 0.125,
        "extra": 0.005,
    },
    "kpx_group_2": {
        "lgb_l1_n1500": 0.175,
        "q06": 0.110,
        "q07": 0.485,
        "xgb_l1": 0.090,
        "cat_mae": 0.085,
        "extra": 0.055,
    },
}
LOCKED_AFFINE = {
    "kpx_group_1": (1.16, -270.0),
    "kpx_group_2": (1.04, -300.0),
}
# The existing reference proves that the locked development recipe used 102%
# of capacity, not 100%, as its upper physical guardrail.
LOCKED_BOUNDS = {
    "kpx_group_1": (1.0, 0.0, 0.0, 1.02),
    "kpx_group_2": (1.0, 0.0, 0.0, 1.02),
}
REFERENCE_FILE = "dev2023_locked_ensemble.parquet"
DEV_START = pd.Timestamp("2023-01-01 01:00:00")
DEV_END_INCLUSIVE = pd.Timestamp("2024-01-01 00:00:00")


def _read_dev_prediction(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"missing locked dev artifact: {path}")
    frame = pd.read_parquet(path)
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError(f"{path.name} must use a DatetimeIndex")
    if frame.index.min() < DEV_START or frame.index.max() > DEV_END_INCLUSIVE:
        raise ValueError(
            f"{path.name} is outside the locked dev-2023 boundary; refusing to read it"
        )
    if tuple(frame.columns) != GROUPS:
        raise ValueError(
            f"{path.name} groups/order changed: {tuple(frame.columns)!r}"
        )
    return frame


def reproduce(artifact_dir: Path) -> pd.DataFrame:
    """Apply the fixed recipe to the six exact dev prediction artifacts."""

    predictions = {
        model: _read_dev_prediction(artifact_dir / filename)
        for model, filename in MODEL_FILES.items()
    }
    ensembler = SimplexEnsembler.from_weights(
        LOCKED_WEIGHTS,
        capacities=CAPACITIES,
    )
    affine = GroupAffineCalibrator.from_parameters(
        LOCKED_AFFINE,
        capacities=CAPACITIES,
    )
    bounded = FICRBoundedPostprocessor.from_parameters(
        LOCKED_BOUNDS,
        capacities=CAPACITIES,
    )
    blended = ensembler.predict(predictions)
    calibrated = affine.transform(blended)
    return bounded.transform(calibrated)


def locked_recipe_dict() -> dict[str, object]:
    """Return the complete versionable recipe with no fitted/gate data."""

    return {
        "scope": "dev2023_only",
        "model_files": MODEL_FILES,
        "groups": list(GROUPS),
        "capacities_kwh": CAPACITIES,
        "weights": LOCKED_WEIGHTS,
        "affine": {
            group: {"scale": value[0], "bias_kwh": value[1]}
            for group, value in LOCKED_AFFINE.items()
        },
        "bounds": {
            group: {
                "scale": value[0],
                "bias_fraction": value[1],
                "lower_capacity_fraction": value[2],
                "upper_capacity_fraction": value[3],
            }
            for group, value in LOCKED_BOUNDS.items()
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "oof",
        help="directory containing the exact dev2023 parquet artifacts",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="optional path for the reproduced parquet; omitted means verify only",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="allow replacement of an explicitly requested output path",
    )
    parser.add_argument("--atol", type=float, default=1e-10)
    args = parser.parse_args()

    artifact_dir = args.artifact_dir.resolve()
    reproduced = reproduce(artifact_dir)
    reference = _read_dev_prediction(artifact_dir / REFERENCE_FILE)
    if not reproduced.index.equals(reference.index):
        raise AssertionError("reproduced and reference indexes differ")
    difference = np.abs(
        reproduced.to_numpy(dtype=float) - reference.to_numpy(dtype=float)
    )
    max_abs_difference = float(difference.max(initial=0.0))
    if not np.allclose(
        reproduced.to_numpy(dtype=float),
        reference.to_numpy(dtype=float),
        rtol=0.0,
        atol=float(args.atol),
    ):
        raise AssertionError(
            "locked recipe no longer reproduces the reference; "
            f"max_abs_difference={max_abs_difference:.12g}"
        )

    if args.output is not None:
        output = args.output.resolve()
        if output.exists() and not args.overwrite:
            raise FileExistsError(
                f"output exists: {output}; pass --overwrite to replace it"
            )
        output.parent.mkdir(parents=True, exist_ok=True)
        reproduced.to_parquet(output)

    report = locked_recipe_dict()
    report["reference_file"] = REFERENCE_FILE
    report["rows"] = len(reproduced)
    report["max_abs_difference"] = max_abs_difference
    report["status"] = "verified"
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
