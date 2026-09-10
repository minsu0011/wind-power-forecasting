"""CUDA-only smooth-FICR training for the frozen raw attention architecture.

The representation, preprocessing, optimizer, schedule, seeds and batching are
imported from :mod:`src.raw_spatiotemporal_attention`.  This module changes only
the registered elementwise training loss and its official eligibility mask.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from src.metric import TARGET_COLS
from src.raw_grid_wind import canonical_sha256
from src.raw_spatiotemporal_attention import (
    BATCH_SIZE,
    BLEND_WEIGHTS,
    DROPOUT,
    EPOCHS,
    EXPECTED_HOURS,
    GRADIENT_CLIP_NORM,
    LEARNING_RATE,
    MODEL_ID as ARCHITECTURE_ID,
    NODE_DIM,
    RawSpatiotemporalAttentionRegressor,
    RawSpatiotemporalTransformer,
    SEEDS,
    SharedNodeGeoAttentionConvNet,
    TEMPORAL_DILATIONS,
    TEMPORAL_DIM,
    WEIGHT_DECAY,
    _array_sha256,
    _configure_determinism,
    _numpy_state_dict,
    _run_keys,
)
from src.smooth_ficr_objective import SmoothFICRLossConfig


MODEL_ID = f"{ARCHITECTURE_ID}_sficr"
CANDIDATE_KEYS: tuple[str, ...] = tuple(
    f"{MODEL_ID}_w{int(round(weight * 1000)):03d}" for weight in BLEND_WEIGHTS
)
_REGISTERED_LOSS = SmoothFICRLossConfig()
_REGISTERED_LOSS.validate()
OFFICIAL_ELIGIBILITY_CF = 0.10
OFFICIAL_THRESHOLDS_CF = _REGISTERED_LOSS.thresholds_cf
SETTLEMENT_WEIGHTS = _REGISTERED_LOSS.settlement_weights
MAE_COEFFICIENT = _REGISTERED_LOSS.mae_coefficient
TEMPERATURE_CF = _REGISTERED_LOSS.temperature_cf
SMOOTH_ABSOLUTE_EPSILON_CF = _REGISTERED_LOSS.smooth_absolute_epsilon_cf
SIGMOID_CLIP = _REGISTERED_LOSS.sigmoid_clip


def torch_smooth_ficr_elementwise(
    prediction_cf: torch.Tensor,
    actual_cf: torch.Tensor,
    mean_fit_eligible_actual_cf: torch.Tensor,
) -> torch.Tensor:
    """Return the preregistered differentiable per-cell loss.

    Eligibility is deliberately handled by the caller so this function remains
    directly comparable with the registered NumPy analytic objective.
    """

    if prediction_cf.shape != actual_cf.shape:
        raise ValueError("prediction and actual tensor shapes differ")
    if mean_fit_eligible_actual_cf.ndim != 3:
        raise ValueError("mean actual tensor must have shape [1, groups, 1]")
    if mean_fit_eligible_actual_cf.shape[0] != 1 or mean_fit_eligible_actual_cf.shape[2] != 1:
        raise ValueError("mean actual tensor must have shape [1, groups, 1]")
    if mean_fit_eligible_actual_cf.shape[1] != prediction_cf.shape[1]:
        raise ValueError("mean actual tensor group dimension changed")
    if not torch.isfinite(mean_fit_eligible_actual_cf).all() or not torch.all(
        mean_fit_eligible_actual_cf > 0.0
    ):
        raise ValueError("mean fit eligible actual must be finite and positive")
    error = prediction_cf - actual_cf
    radius = torch.sqrt(error.square() + SMOOTH_ABSOLUTE_EPSILON_CF**2)
    actual_scale = actual_cf / (8.0 * mean_fit_eligible_actual_cf)
    settlement = torch.zeros_like(radius)
    for threshold, weight in zip(OFFICIAL_THRESHOLDS_CF, SETTLEMENT_WEIGHTS):
        argument = torch.clamp(
            (threshold - radius) / TEMPERATURE_CF,
            -SIGMOID_CLIP,
            SIGMOID_CLIP,
        )
        settlement = settlement + weight * torch.sigmoid(argument)
    result = MAE_COEFFICIENT * radius - actual_scale * settlement
    if not torch.isfinite(result).all():
        raise FloatingPointError("smooth-FICR elementwise loss is non-finite")
    return result


class RawSpatiotemporalSmoothFICRRegressor(RawSpatiotemporalAttentionRegressor):
    """Frozen raw attention architecture trained by exact smooth-FICR autograd."""

    @staticmethod
    def _device() -> torch.device:
        if not torch.cuda.is_available() or torch.version.cuda != "12.8":
            raise RuntimeError("registered CUDA 12.8 device is required; fallback is forbidden")
        if torch.cuda.get_device_name(0) != "NVIDIA GeForce RTX 5060 Ti":
            raise RuntimeError("registered CUDA device identity changed")
        return torch.device("cuda:0")

    def fit(
        self, features: pd.DataFrame, target_cf: pd.DataFrame
    ) -> "RawSpatiotemporalSmoothFICRRegressor":
        if not isinstance(target_cf, pd.DataFrame) or not features.index.equals(target_cf.index):
            raise ValueError("feature and target indexes differ")
        target_columns = tuple(map(str, target_cf.columns))
        if not target_columns or any(group not in TARGET_COLS for group in target_columns):
            raise ValueError("target columns must be a non-empty ordered subset of TARGET_COLS")
        if len(set(target_columns)) != len(target_columns):
            raise ValueError("target columns contain duplicates")

        requested_keys = _run_keys(features.index)
        values = np.full((len(features), len(TARGET_COLS)), np.nan, dtype=np.float32)
        for column in target_columns:
            values[:, TARGET_COLS.index(column)] = target_cf[column].to_numpy(
                dtype=np.float32, copy=False
            )
        target_by_run = values.reshape(
            -1, EXPECTED_HOURS, len(TARGET_COLS)
        ).transpose(0, 2, 1)
        complete_by_group = np.isfinite(target_by_run).all(axis=2)
        eligible_run = complete_by_group.any(axis=1)
        if not eligible_run.any():
            raise ValueError("no run has a complete 24-hour target for any registered group")

        eligible_features = features.iloc[np.repeat(eligible_run, EXPECTED_HOURS)]
        target = target_by_run[eligible_run]
        complete_targets = complete_by_group[eligible_run]
        target_filled = np.where(complete_targets[:, :, None], target, 0.0).astype(np.float32)
        eligible_cells = complete_targets[:, :, None] & (
            target_filled >= OFFICIAL_ELIGIBILITY_CF
        )
        means = np.ones(len(TARGET_COLS), dtype=np.float32)
        counts: dict[str, int] = {}
        for position, group in enumerate(TARGET_COLS):
            count = int(eligible_cells[:, position, :].sum())
            counts[group] = count
            if group in target_columns:
                if count <= 0:
                    raise ValueError(f"{group} has no officially eligible fit cell")
                means[position] = np.mean(
                    target_filled[:, position, :][eligible_cells[:, position, :]],
                    dtype=np.float64,
                )
            elif count != 0:
                raise AssertionError("unmaterialized target head became eligible")
        if not np.isfinite(means).all() or np.any(means <= 0.0):
            raise ValueError("per-head causal mean actual is invalid")

        transformer = RawSpatiotemporalTransformer(self.catalogs)
        tensors, fit_keys = transformer.fit_transform(eligible_features)
        device = self._device()
        coords = {
            source: torch.from_numpy(transformer.coords_[source]).to(device)
            for source in ("ldaps", "gfs")
        }
        mean_tensor = torch.from_numpy(means.reshape(1, len(TARGET_COLS), 1)).to(device)
        states: list[dict[str, np.ndarray]] = []
        histories: list[dict[str, float | int]] = []
        parameter_count = None
        for seed in SEEDS:
            _configure_determinism(seed)
            model = SharedNodeGeoAttentionConvNet().to(device)
            if parameter_count is None:
                parameter_count = sum(parameter.numel() for parameter in model.parameters())
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=EPOCHS
            )
            generator = torch.Generator(device="cpu").manual_seed(seed)
            losses: list[float] = []
            for _epoch in range(EPOCHS):
                order = torch.randperm(len(target_filled), generator=generator).numpy()
                epoch_loss = 0.0
                epoch_weight = 0
                model.train()
                for start in range(0, len(order), BATCH_SIZE):
                    rows = order[start : start + BATCH_SIZE]
                    ldaps = torch.from_numpy(tensors["ldaps"][rows]).to(device)
                    gfs = torch.from_numpy(tensors["gfs"][rows]).to(device)
                    truth = torch.from_numpy(target_filled[rows]).to(device)
                    mask = torch.from_numpy(eligible_cells[rows]).to(device)
                    if not torch.any(mask):
                        raise ValueError("minibatch has no officially eligible target cell")
                    optimizer.zero_grad(set_to_none=True)
                    prediction = model(ldaps, gfs, coords["ldaps"], coords["gfs"])
                    element = torch_smooth_ficr_elementwise(
                        prediction, truth, mean_tensor
                    )
                    loss = torch.sum(element * mask) / torch.sum(mask)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("smooth-FICR batch loss is non-finite")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), GRADIENT_CLIP_NORM
                    )
                    optimizer.step()
                    epoch_loss += float(loss.detach().cpu()) * len(rows)
                    epoch_weight += len(rows)
                scheduler.step()
                losses.append(epoch_loss / epoch_weight)
            model.eval()
            states.append(_numpy_state_dict(model))
            histories.append(
                {
                    "seed": seed,
                    "first_epoch_loss": losses[0],
                    "final_epoch_loss": losses[-1],
                    "minimum_epoch_loss": min(losses),
                }
            )

        self.transformer_ = transformer
        self.state_dicts_ = states
        self.training_histories_ = histories
        self.parameter_count_ = int(parameter_count or 0)
        self.device_used_ = str(device)
        self.requested_fit_index_ = features.index.copy()
        self.fit_index_ = eligible_features.index.copy()
        self.requested_fit_run_keys_ = requested_keys
        self.fit_run_keys_ = fit_keys
        self.dropped_run_keys_ = requested_keys[~eligible_run]
        self.target_columns_ = target_columns
        self.eligible_by_group_ = complete_by_group
        self.eligible_run_mask_sha256_ = _array_sha256(eligible_run.astype(np.uint8))
        self.eligible_group_mask_sha256_ = _array_sha256(complete_by_group.astype(np.uint8))
        self.fit_target_sha256_ = _array_sha256(target_filled)
        self.smooth_ficr_eligible_cell_mask_sha256_ = _array_sha256(
            eligible_cells.astype(np.uint8)
        )
        self.mean_fit_eligible_actual_cf_ = means
        self.mean_fit_eligible_actual_cf_sha256_ = _array_sha256(means)
        self.eligible_cells_by_group_ = counts
        return self

    def metadata(self) -> dict[str, Any]:
        result = super().metadata()
        result["model_id"] = MODEL_ID
        result["architecture"]["base_model_id"] = ARCHITECTURE_ID
        result["architecture"]["dropout"] = DROPOUT
        result["training"].update(
            {
                "loss": "fixed eligible-only exact-autograd official smooth-FICR",
                "smooth_l1_beta": None,
                "low_cf_loss_weight": 0.0,
                "official_eligibility_cf": OFFICIAL_ELIGIBILITY_CF,
                "official_thresholds_cf": list(OFFICIAL_THRESHOLDS_CF),
                "settlement_weights": list(SETTLEMENT_WEIGHTS),
                "mae_coefficient": MAE_COEFFICIENT,
                "temperature_cf": TEMPERATURE_CF,
                "smooth_absolute_epsilon_cf": SMOOTH_ABSOLUTE_EPSILON_CF,
                "sigmoid_clip": SIGMOID_CLIP,
                "batch_reduction": "global arithmetic mean over eligible cells",
                "tree_hessian_surrogate_used": False,
            }
        )
        result["smooth_ficr"] = {
            "mean_fit_eligible_actual_cf": {
                group: float(self.mean_fit_eligible_actual_cf_[position])
                if group in self.target_columns_
                else None
                for position, group in enumerate(TARGET_COLS)
            },
            "mean_fit_eligible_actual_cf_array_sha256": self.mean_fit_eligible_actual_cf_sha256_,
            "eligible_cells_by_group": dict(self.eligible_cells_by_group_),
            "eligible_cell_mask_sha256": self.smooth_ficr_eligible_cell_mask_sha256_,
            "per_head_causal_mean": True,
            "eligible_only": True,
        }
        result["state_dict_contract_sha256"] = canonical_sha256(
            result["state_dict_hashes"]
        )
        return result


def blend_smooth_ficr_prediction(
    baseline_kwh: pd.Series,
    model_cf: pd.Series,
    *,
    capacity_kwh: float,
    weight: float,
) -> pd.Series:
    if weight not in (0.0, *BLEND_WEIGHTS):
        raise ValueError("blend weight is not preregistered")
    if not baseline_kwh.index.equals(model_cf.index):
        raise ValueError("baseline and model prediction indexes differ")
    if weight == 0.0:
        return baseline_kwh.copy()
    values = (1.0 - weight) * baseline_kwh.to_numpy(dtype=np.float64) + weight * (
        np.clip(model_cf.to_numpy(dtype=np.float64), 0.0, 1.02) * capacity_kwh
    )
    return pd.Series(
        np.clip(values, 0.0, 1.02 * capacity_kwh),
        index=baseline_kwh.index,
        name=baseline_kwh.name,
    )


def candidate_frame(
    baseline_kwh: pd.Series, model_cf: pd.Series, *, capacity_kwh: float
) -> pd.DataFrame:
    result = pd.DataFrame(index=baseline_kwh.index)
    for key, weight in zip(CANDIDATE_KEYS, BLEND_WEIGHTS):
        result[key] = blend_smooth_ficr_prediction(
            baseline_kwh, model_cf, capacity_kwh=capacity_kwh, weight=weight
        )
    if tuple(result.columns) != CANDIDATE_KEYS:
        raise AssertionError("smooth-FICR candidate order changed")
    identity = blend_smooth_ficr_prediction(
        baseline_kwh, model_cf, capacity_kwh=capacity_kwh, weight=0.0
    )
    if identity.to_numpy().tobytes() != baseline_kwh.to_numpy().tobytes():
        raise AssertionError("weight-zero identity is not value-bit exact")
    return result


def parse_candidate_key(key: str) -> float:
    for candidate, weight in zip(CANDIDATE_KEYS, BLEND_WEIGHTS):
        if key == candidate:
            return weight
    raise ValueError(f"candidate is not preregistered: {key}")


def select_group_candidate(
    comparisons: Mapping[str, Mapping[str, Mapping[str, Any]]],
    required_slices: Sequence[str],
) -> tuple[str | None, dict[str, Any]]:
    """Apply the frozen all-slice total plus full-component Stage1 gate."""

    if set(comparisons) != set(CANDIDATE_KEYS) or len(comparisons) != len(CANDIDATE_KEYS):
        raise ValueError("candidate key set changed")
    required = tuple(required_slices)
    if "full" not in required:
        raise ValueError("full slice is required for the component gate")
    audit: dict[str, Any] = {}
    passing: list[str] = []
    for key in CANDIDATE_KEYS:
        records = comparisons[key]
        if set(records) != set(required) or len(records) != len(required):
            raise ValueError(f"slice key set changed for {key}")
        deltas: dict[str, float] = {}
        for name in required:
            exact = float(records[name]["candidate"]["score"]) - float(
                records[name]["baseline"]["score"]
            )
            if float(records[name]["delta"]) != exact:
                raise ValueError("candidate delta arithmetic changed")
            deltas[name] = exact
        full = records["full"]
        one_minus_nmae_delta = float(full["candidate"]["one_minus_nmae"]) - float(
            full["baseline"]["one_minus_nmae"]
        )
        ficr_delta = float(full["candidate"]["ficr"]) - float(
            full["baseline"]["ficr"]
        )
        total_pass = all(value > 0.0 for value in deltas.values())
        component_pass = one_minus_nmae_delta >= 0.0 and ficr_delta >= 0.0
        row = {
            "weight": parse_candidate_key(key),
            "deltas": deltas,
            "minimum": min(deltas.values()),
            "mean": float(np.mean(list(deltas.values()))),
            "all_slices_strictly_positive": total_pass,
            "full_one_minus_nmae_delta": one_minus_nmae_delta,
            "full_ficr_delta": ficr_delta,
            "full_components_nonnegative": component_pass,
            "passed": total_pass and component_pass,
        }
        audit[key] = row
        if row["passed"]:
            passing.append(key)
    if not passing:
        return None, {"candidates": audit, "selected": "identity"}
    selected = max(
        passing,
        key=lambda key: (
            audit[key]["minimum"],
            audit[key]["mean"],
            -audit[key]["weight"],
            -CANDIDATE_KEYS.index(key),
        ),
    )
    return selected, {"candidates": audit, "selected": selected}


__all__ = [
    "ARCHITECTURE_ID",
    "CANDIDATE_KEYS",
    "MODEL_ID",
    "RawSpatiotemporalSmoothFICRRegressor",
    "blend_smooth_ficr_prediction",
    "candidate_frame",
    "parse_candidate_key",
    "select_group_candidate",
    "torch_smooth_ficr_elementwise",
]
