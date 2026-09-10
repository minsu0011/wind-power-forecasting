"""Frozen low-dimensional inverse calibration feasibility diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution


PROJECT_DIR = Path(__file__).resolve().parents[1]
PROTOCOL = PROJECT_DIR / "configs/public_inverse_scale_feasibility_protocol_v1.json"
ADDENDUM = PROJECT_DIR / "configs/public_inverse_scale_feasibility_protocol_v1_addendum.json"
PROTOCOL_SHA = "bec381226ddcab1ec76cf273fe176fb47eb0425aa5af4b6c9077823326e0bb66"
ADDENDUM_SHA = "e29582ecec11c828a73e08ef43c5624f05ce6408f380ba7e8b54bda421880c49"
TARGETS = ("kpx_group_1", "kpx_group_2", "kpx_group_3")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/audits/public_inverse_scale_lowdim_feasibility_v1.json"),
    )
    return parser.parse_args(argv)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify(spec: Mapping[str, Any]) -> Path:
    path = Path(str(spec["path"]))
    path = path if path.is_absolute() else PROJECT_DIR / path
    assert path.is_file()
    expected_size = spec.get("bytes", spec.get("size_bytes"))
    if expected_size is not None:
        assert path.stat().st_size == int(expected_size)
    assert sha256(path) == str(spec["sha256"])
    return path


def write_atomic(path: Path, payload: Mapping[str, Any]) -> Path:
    destination = path if path.is_absolute() else PROJECT_DIR / path
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


class SyntheticMetricModel:
    def __init__(
        self,
        baseline: pd.DataFrame,
        actual: pd.DataFrame,
        capacities: Mapping[str, float],
    ) -> None:
        assert baseline.index.equals(actual.index)
        self.baseline = {
            group: baseline[group].to_numpy(dtype=np.float64) / float(capacities[group])
            for group in TARGETS
        }
        self.residual = {
            group: actual[group].to_numpy(dtype=np.float64) / float(capacities[group])
            - self.baseline[group]
            for group in TARGETS
        }

    def predict_components(self, theta: np.ndarray, scales: Sequence[float]) -> np.ndarray:
        inflation = float(theta[0])
        synthetic: dict[str, np.ndarray] = {}
        valid: dict[str, np.ndarray] = {}
        for position, group in enumerate(TARGETS):
            synthetic[group] = np.clip(
                float(theta[position + 1]) * self.baseline[group]
                + inflation * self.residual[group],
                0.0,
                1.02,
            )
            valid[group] = synthetic[group] >= 0.10
        result: list[float] = []
        for scale in scales:
            n_rows: list[float] = []
            f_rows: list[float] = []
            for group in TARGETS:
                y = synthetic[group][valid[group]]
                p = np.clip(float(scale) * self.baseline[group][valid[group]], 0.0, 1.02)
                error = np.abs(p - y)
                n_rows.append(1.0 - float(error.mean()))
                price = np.where(error <= 0.06, 4.0, np.where(error <= 0.08, 3.0, 0.0))
                f_rows.append(float(np.sum(y * price) / np.sum(y * 4.0)))
            result.extend((float(np.mean(n_rows)), float(np.mean(f_rows))))
        return np.asarray(result, dtype=np.float64)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    assert sha256(PROTOCOL) == PROTOCOL_SHA
    assert sha256(ADDENDUM) == ADDENDUM_SHA
    assert PROTOCOL.with_suffix(".sha256").read_text(encoding="utf-8") == f"{PROTOCOL_SHA}  {PROTOCOL.name}\n"
    assert ADDENDUM.with_suffix(".sha256").read_text(encoding="utf-8") == f"{ADDENDUM_SHA}  {ADDENDUM.name}\n"
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    addendum = json.loads(ADDENDUM.read_text(encoding="utf-8"))
    input_before = {
        name: {"path": str(verify(spec).resolve()), "bytes": verify(spec).stat().st_size, "sha256": sha256(verify(spec))}
        for name, spec in protocol["input_identities"].items()
    }

    baseline_path = verify(protocol["input_identities"]["recent_v4_2024_analog"])
    baseline = pd.read_parquet(baseline_path).astype(np.float64)
    baseline.index = pd.DatetimeIndex(baseline.index, name="forecast_kst_dtm")
    expected_index = pd.date_range(
        "2024-01-01 01:00", "2025-01-01 00:00", freq="h", name="forecast_kst_dtm"
    )
    assert tuple(baseline.columns) == TARGETS and baseline.index.equals(expected_index)
    labels_path = verify(protocol["input_identities"]["labels"])
    actual = pd.read_csv(labels_path)
    actual.index = pd.DatetimeIndex(pd.to_datetime(actual.pop("kst_dtm")), name="forecast_kst_dtm")
    actual = actual.loc[expected_index, list(TARGETS)].astype(np.float64)

    capacities = protocol["metric_contract"]["capacities_kwh"]
    model = SyntheticMetricModel(baseline, actual, capacities)
    observations = protocol["public_observations"]
    scales = tuple(float(item["global_scale"]) for item in observations)
    observed = np.asarray(
        [value for item in observations for value in (item["one_minus_nmae"], item["ficr"])],
        dtype=np.float64,
    )
    contract = protocol["four_parameter_synthetic_shift_model"]
    centers = np.asarray(contract["prior_centers"], dtype=np.float64)
    prior_scales = np.asarray(contract["prior_scales"], dtype=np.float64)
    component_scales = np.asarray(
        [contract["metric_residual_scales"]["one_minus_nmae"], contract["metric_residual_scales"]["ficr"]]
        * len(scales),
        dtype=np.float64,
    )
    bounds = tuple(tuple(map(float, contract["bounds"][name])) for name in contract["parameter_order"])
    optimizer = protocol["continuous_optimizer"]

    def fit(included: tuple[int, ...], ridge: float, seed: int) -> dict[str, Any]:
        fit_scales = tuple(scales[index] for index in included)
        positions = np.asarray([2 * index + offset for index in included for offset in (0, 1)], dtype=np.int64)

        def objective(theta: np.ndarray) -> float:
            predicted = model.predict_components(theta, fit_scales)
            data = (predicted - observed[positions]) / component_scales[positions]
            prior = np.sqrt(float(ridge)) * (theta - centers) / prior_scales
            return float(np.dot(data, data) + np.dot(prior, prior))

        result = differential_evolution(
            objective,
            bounds=bounds,
            seed=int(seed),
            maxiter=int(optimizer["maxiter"]),
            popsize=int(optimizer["popsize"]),
            tol=float(optimizer["tol"]),
            polish=bool(optimizer["polish"]),
            workers=int(optimizer["workers"]),
            init="latinhypercube",
            updating="immediate",
        )
        theta = np.asarray(result.x, dtype=np.float64)
        all_predicted = model.predict_components(theta, scales)
        return {
            "included_scale_indices": list(included),
            "included_scales": list(fit_scales),
            "ridge_multiplier": float(ridge),
            "seed": int(seed),
            "theta": theta.tolist(),
            "center_scales": theta[1:].tolist(),
            "objective": float(result.fun),
            "iterations": int(result.nit),
            "evaluations": int(result.nfev),
            "optimizer_success": bool(result.success),
            "optimizer_message": str(result.message),
            "predicted_all_components": all_predicted.tolist(),
            "residual_all_components": (all_predicted - observed).tolist(),
        }

    all_indices = tuple(range(len(scales)))
    primary = fit(all_indices, 1.0, int(optimizer["primary_seed"]))
    seed_stress = [fit(all_indices, 1.0, int(seed)) for seed in optimizer["optimizer_seed_stress"]]
    ridge_stress = [
        fit(all_indices, float(ridge), int(optimizer["primary_seed"]))
        for ridge in protocol["stability_diagnostics"]["ridge_multiplier_stress"]
    ]
    leave_one_out: list[dict[str, Any]] = []
    heldout_errors: list[float] = []
    for heldout in all_indices:
        included = tuple(index for index in all_indices if index != heldout)
        record = fit(included, 1.0, int(optimizer["primary_seed"]))
        positions = (2 * heldout, 2 * heldout + 1)
        record["heldout_scale"] = scales[heldout]
        record["heldout_observed"] = observed[list(positions)].tolist()
        record["heldout_predicted"] = [record["predicted_all_components"][position] for position in positions]
        record["heldout_residual"] = [record["residual_all_components"][position] for position in positions]
        heldout_errors.extend(abs(value) for value in record["heldout_residual"])
        leave_one_out.append(record)

    theta_primary = np.asarray(primary["theta"], dtype=np.float64)
    z_step = float(addendum["numerical_details"]["jacobian_central_step_in_z_units"])
    jacobian = np.empty((2 * len(scales), 4), dtype=np.float64)
    for column in range(4):
        shift = np.zeros(4, dtype=np.float64)
        shift[column] = z_step * prior_scales[column]
        plus = model.predict_components(theta_primary + shift, scales)
        minus = model.predict_components(theta_primary - shift, scales)
        jacobian[:, column] = (plus - minus) / (2.0 * z_step)
    singular = np.linalg.svd(jacobian, compute_uv=False)
    rank = int(np.linalg.matrix_rank(jacobian))
    condition = float(singular[0] / singular[-1]) if singular[-1] > 0.0 else float("inf")

    populations = [primary, *seed_stress, *ridge_stress, *leave_one_out]
    center_matrix = np.asarray([record["center_scales"] for record in populations], dtype=np.float64)
    factor_spans = np.ptp(center_matrix, axis=0)
    primary_centers = np.asarray(primary["center_scales"], dtype=np.float64)
    seed_difference = max(
        float(np.max(np.abs(np.asarray(record["center_scales"]) - primary_centers)))
        for record in seed_stress
    )
    primary_residual = np.asarray(primary["residual_all_components"], dtype=np.float64)
    primary_rmse = float(np.sqrt(np.mean(np.square(primary_residual))))
    loo_max = float(max(heldout_errors))
    primary_inside = bool(
        all(lower < value < upper for value, (lower, upper) in zip(theta_primary, bounds, strict=True))
    )
    thresholds = protocol["GO_all_required"]
    checks = {
        "primary_component_rmse": {"value": primary_rmse, "limit": float(thresholds["primary_component_rmse_max"]), "pass": primary_rmse <= float(thresholds["primary_component_rmse_max"])},
        "LOO_component_max_absolute_error": {"value": loo_max, "limit": float(thresholds["leave_one_scale_out_component_absolute_error_max"]), "pass": loo_max <= float(thresholds["leave_one_scale_out_component_absolute_error_max"])},
        "center_scale_max_span": {"value": float(factor_spans.max()), "per_group": factor_spans.tolist(), "limit": float(thresholds["center_scale_max_span_across_primary_LOO_ridge_and_seed_stress"]), "pass": float(factor_spans.max()) <= float(thresholds["center_scale_max_span_across_primary_LOO_ridge_and_seed_stress"])},
        "optimizer_seed_center_scale_difference": {"value": seed_difference, "limit": float(thresholds["primary_vs_optimizer_seed_center_scale_max_absolute_difference"]), "pass": seed_difference <= float(thresholds["primary_vs_optimizer_seed_center_scale_max_absolute_difference"])},
        "jacobian_rank": {"value": rank, "required": int(thresholds["jacobian_numerical_rank"]), "pass": rank == int(thresholds["jacobian_numerical_rank"])},
        "jacobian_condition": {"value": condition, "limit": float(thresholds["jacobian_condition_number_max"]), "pass": condition <= float(thresholds["jacobian_condition_number_max"])},
        "primary_parameters_inside_bounds": {"value": primary_inside, "pass": primary_inside},
    }
    go = bool(all(record["pass"] for record in checks.values()))
    input_after = {
        name: {"path": str(verify(spec).resolve()), "bytes": verify(spec).stat().st_size, "sha256": sha256(verify(spec))}
        for name, spec in protocol["input_identities"].items()
    }
    assert input_before == input_after
    payload = {
        "schema_version": 1,
        "diagnostic_type": "public_inverse_scale_lowdim_feasibility_v1",
        "risk": protocol["risk_classification"],
        "protocol": {"path": str(PROTOCOL), "sha256": PROTOCOL_SHA},
        "addendum": {"path": str(ADDENDUM), "sha256": ADDENDUM_SHA},
        "script": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__))},
        "inputs_before_after_exact": True,
        "inputs": input_before,
        "public_observed_components": observed.tolist(),
        "primary": primary,
        "optimizer_seed_stress": seed_stress,
        "ridge_stress": ridge_stress,
        "leave_one_scale_out": leave_one_out,
        "jacobian": {"matrix": jacobian.tolist(), "singular_values": singular.tolist(), "rank": rank, "condition_number": condition},
        "stability": {"factor_population_count": len(populations), "factor_spans": factor_spans.tolist(), "seed_max_difference": seed_difference},
        "GO_checks": checks,
        "diagnostic_GO": go,
        "disposition": "eligible_for_separate_candidate_preregister" if go else "NO_GO_underidentified_or_unstable_no_candidate",
        "candidate_prediction_cells_created": 0,
        "candidate_parquet_created": False,
        "candidate_CSV_created": False,
        "no_discrete_factor_or_row_bin_grid": True,
        "leaderboard_score_claim": False,
    }
    output = write_atomic(args.out, payload)
    print(f"diagnostic_GO={go} disposition={payload['disposition']}")
    print(f"primary_theta={primary['theta']}")
    print(f"primary_rmse={primary_rmse:.12g} loo_max={loo_max:.12g} factor_span={factor_spans.max():.12g} rank={rank} condition={condition:.12g}")
    print(f"audit_sha256={sha256(output)}")


if __name__ == "__main__":
    main()
