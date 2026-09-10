"""Training-only SCADA multitask regularization for the raw spatial CNN.

The inference graph consumes exactly the registered LDAPS/GFS weather tensors.
SCADA is accepted only by :meth:`fit` as a causally bounded auxiliary target;
it is neither persisted as an inference feature nor accepted by :meth:`predict`.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from src.metric import TARGET_COLS
from src.raw_grid_wind import canonical_sha256
from src.raw_spatial_grid_temporal_cnn import (
    BATCH_SIZE,
    BLEND_WEIGHT,
    DROPOUT,
    EPOCHS,
    EXPECTED_HOURS,
    GRADIENT_CLIP_NORM,
    LEARNING_RATE,
    LOW_CF_LOSS_WEIGHT,
    MaskedSpatialEncoder,
    SEEDS,
    SMOOTH_L1_BETA,
    SPATIAL_OUTPUT,
    TEMPORAL_DILATIONS,
    TEMPORAL_DIM,
    WEIGHT_DECAY,
    _array_sha256,
    _configure_determinism,
    _load_numpy_state_dict,
    _numpy_state_dict,
    _run_keys,
    rasterize_source,
    validate_raster_catalogs,
)
from src.raw_spatiotemporal_attention import RawSpatiotemporalTransformer, ResidualTemporalBlock


MODEL_ID = "spatial16x24_maskconv_temporal48_scadaaux4_e240_s2"
CANDIDATE_KEY = f"{MODEL_ID}_w025"
AUX_CHANNELS: tuple[str, ...] = (
    "ws_norm",
    "wd_sin",
    "wd_cos",
    "availability",
)
AUX_WS_WEIGHT = 0.10
AUX_DIRECTION_WEIGHT = 0.05
AUX_AVAILABILITY_WEIGHT = 0.10
EXPECTED_PARAMETER_COUNT = 41823


class SpatialGridTemporalConvNetSCADAMultitask(nn.Module):
    """Exact base inference backbone plus group-specific training heads."""

    def __init__(self) -> None:
        super().__init__()
        self.ldaps_spatial = MaskedSpatialEncoder()
        self.gfs_spatial = MaskedSpatialEncoder()
        self.fusion = nn.Linear(2 * SPATIAL_OUTPUT + 4, TEMPORAL_DIM, bias=True)
        self.temporal = nn.Sequential(
            *(
                ResidualTemporalBlock(TEMPORAL_DIM, dilation, DROPOUT)
                for dilation in TEMPORAL_DILATIONS
            )
        )
        self.group_heads = nn.ModuleDict(
            {
                group: nn.Conv1d(TEMPORAL_DIM, 1, kernel_size=1, bias=True)
                for group in TARGET_COLS
            }
        )
        self.auxiliary_heads = nn.ModuleDict(
            {
                group: nn.Conv1d(TEMPORAL_DIM, len(AUX_CHANNELS), kernel_size=1, bias=True)
                for group in TARGET_COLS
            }
        )

    @staticmethod
    def _horizon_features(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        hours = torch.arange(1, EXPECTED_HOURS + 1, device=device, dtype=dtype)
        angle = 2.0 * math.pi * hours / EXPECTED_HOURS
        return torch.stack(
            (
                torch.sin(angle),
                torch.cos(angle),
                torch.sin(2.0 * angle),
                torch.cos(2.0 * angle),
            ),
            dim=-1,
        )

    def _encode(self, ldaps: torch.Tensor, gfs: torch.Tensor) -> torch.Tensor:
        if ldaps.ndim != 5 or tuple(ldaps.shape[2:]) != (11, 4, 5):
            raise ValueError("LDAPS raster batch shape changed")
        if gfs.ndim != 5 or tuple(gfs.shape[2:]) != (11, 3, 3):
            raise ValueError("GFS raster batch shape changed")
        if ldaps.shape[:2] != gfs.shape[:2] or ldaps.shape[1] != EXPECTED_HOURS:
            raise ValueError("source run batches differ")
        batch, hours = ldaps.shape[:2]
        ldaps_summary = self.ldaps_spatial(ldaps.reshape(batch * hours, 11, 4, 5))
        gfs_summary = self.gfs_spatial(gfs.reshape(batch * hours, 11, 3, 3))
        ldaps_summary = ldaps_summary.reshape(batch, hours, SPATIAL_OUTPUT)
        gfs_summary = gfs_summary.reshape(batch, hours, SPATIAL_OUTPUT)
        position = self._horizon_features(ldaps.device, ldaps.dtype)
        position = position.unsqueeze(0).expand(batch, -1, -1)
        fused = F.silu(
            self.fusion(torch.cat((ldaps_summary, gfs_summary, position), dim=-1))
        )
        return self.temporal(fused.transpose(1, 2))

    def forward(
        self, ldaps: torch.Tensor, gfs: torch.Tensor, *, return_auxiliary: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        temporal = self._encode(ldaps, gfs)
        direct = torch.stack(
            [
                1.02 * torch.sigmoid(self.group_heads[group](temporal).squeeze(1))
                for group in TARGET_COLS
            ],
            dim=1,
        )
        if not return_auxiliary:
            return direct
        raw_aux = torch.stack(
            [self.auxiliary_heads[group](temporal) for group in TARGET_COLS], dim=1
        )
        auxiliary = torch.stack(
            (
                torch.sigmoid(raw_aux[:, :, 0]),
                torch.tanh(raw_aux[:, :, 1]),
                torch.tanh(raw_aux[:, :, 2]),
                torch.sigmoid(raw_aux[:, :, 3]),
            ),
            dim=2,
        )
        return direct, auxiliary


def _masked_mean(loss: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.to(dtype=loss.dtype)
    denominator = torch.sum(weight)
    if float(denominator.detach().cpu()) == 0.0:
        return torch.sum(loss) * 0.0
    return torch.sum(loss * weight) / denominator


def multitask_loss(
    direct_prediction: torch.Tensor,
    auxiliary_prediction: torch.Tensor,
    direct_truth: torch.Tensor,
    direct_group_mask: torch.Tensor,
    auxiliary_truth: torch.Tensor,
    auxiliary_mask: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the frozen independently-normalized loss components."""

    direct_element = F.smooth_l1_loss(
        direct_prediction, direct_truth, reduction="none", beta=SMOOTH_L1_BETA
    )
    direct_weight = torch.where(
        direct_truth >= 0.10,
        torch.ones_like(direct_truth),
        torch.full_like(direct_truth, LOW_CF_LOSS_WEIGHT),
    ) * direct_group_mask[:, :, None].to(direct_truth.dtype)
    direct = torch.sum(direct_element * direct_weight) / torch.sum(direct_weight)

    auxiliary_element = F.smooth_l1_loss(
        auxiliary_prediction, auxiliary_truth, reduction="none", beta=SMOOTH_L1_BETA
    )
    ws = _masked_mean(auxiliary_element[:, :, 0], auxiliary_mask[:, :, 0])
    direction_joint = auxiliary_mask[:, :, 1] & auxiliary_mask[:, :, 2]
    direction_sin = _masked_mean(auxiliary_element[:, :, 1], direction_joint)
    direction_cos = _masked_mean(auxiliary_element[:, :, 2], direction_joint)
    direction = 0.5 * (direction_sin + direction_cos)
    availability = _masked_mean(auxiliary_element[:, :, 3], auxiliary_mask[:, :, 3])
    total = (
        direct
        + AUX_WS_WEIGHT * ws
        + AUX_DIRECTION_WEIGHT * direction
        + AUX_AVAILABILITY_WEIGHT * availability
    )
    return total, {
        "direct": direct,
        "ws": ws,
        "direction": direction,
        "availability": availability,
        "total": total,
    }


def prepare_auxiliary_targets(
    index: pd.DatetimeIndex,
    targets: Mapping[str, pd.DataFrame],
    expected_masks: Mapping[str, pd.Series | np.ndarray],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Transform selected hourly SCADA targets and enforce frozen coverage."""

    if set(targets) != set(TARGET_COLS) or set(expected_masks) != set(TARGET_COLS):
        raise ValueError("auxiliary target group set changed")
    values = np.full((len(index), len(TARGET_COLS), len(AUX_CHANNELS)), np.nan, np.float32)
    coverage: dict[str, Any] = {}
    for group_position, group in enumerate(TARGET_COLS):
        frame = targets[group]
        if not isinstance(frame, pd.DataFrame) or not frame.index.equals(index):
            raise ValueError(f"{group} auxiliary index changed")
        if tuple(frame.columns) != AUX_CHANNELS:
            raise ValueError(f"{group} auxiliary columns changed")
        raw = frame.to_numpy(dtype=np.float32, copy=True)
        if np.isfinite(raw[:, 0]).any() and (
            np.nanmin(raw[:, 0]) < 0.0 or np.nanmax(raw[:, 0]) > 40.0
        ):
            raise ValueError(f"{group} wind target outside registered 0..40 transform")
        raw[:, 0] = np.clip(raw[:, 0], 0.0, 40.0) / 40.0
        for column in (1, 2):
            finite = np.isfinite(raw[:, column])
            if finite.any() and np.max(np.abs(raw[finite, column])) > 1.000001:
                raise ValueError(f"{group} circular target outside [-1,1]")
        finite_availability = np.isfinite(raw[:, 3])
        if finite_availability.any() and (
            np.min(raw[finite_availability, 3]) < 0.0
            or np.max(raw[finite_availability, 3]) > 1.0
        ):
            raise ValueError(f"{group} availability outside [0,1]")
        expected = np.asarray(expected_masks[group], dtype=bool)
        if expected.shape != (len(index),):
            raise ValueError(f"{group} expected auxiliary mask shape changed")
        if np.any(np.isfinite(raw[~expected])):
            raise ValueError(f"{group} auxiliary target exists outside causal expected mask")
        denominator = int(expected.sum())
        wind_finite = np.isfinite(raw[:, 0]) & expected
        direction_finite = np.isfinite(raw[:, 1]) & np.isfinite(raw[:, 2]) & expected
        availability_finite = np.isfinite(raw[:, 3]) & expected
        if denominator:
            fractions = {
                "wind": float(wind_finite.sum() / denominator),
                "direction": float(direction_finite.sum() / denominator),
                "availability": float(availability_finite.sum() / denominator),
            }
            if fractions["wind"] < 0.99 or fractions["direction"] < 0.99:
                raise ValueError(f"{group} wind/direction coverage below 99%")
            if fractions["availability"] < 0.90:
                raise ValueError(f"{group} availability coverage below 90%")
        else:
            fractions = {"wind": 0.0, "direction": 0.0, "availability": 0.0}
        values[:, group_position] = raw
        coverage[group] = {
            "expected_cells": denominator,
            "finite_cells": {
                "wind": int(wind_finite.sum()),
                "direction_joint": int(direction_finite.sum()),
                "availability": int(availability_finite.sum()),
            },
            "fractions": fractions,
        }
    mask = np.isfinite(values)
    direction_joint = mask[:, :, 1] & mask[:, :, 2]
    mask[:, :, 1] = direction_joint
    mask[:, :, 2] = direction_joint
    filled = np.where(mask, values, 0.0).astype(np.float32)
    coverage["filled_sha256"] = _array_sha256(filled)
    coverage["mask_sha256"] = _array_sha256(mask.astype(np.uint8))
    return filled, mask, coverage


class RawSpatialGridTemporalCNNScadaMultitaskRegressor:
    """Deterministic two-seed direct-CF model with fit-only SCADA losses."""

    def __init__(self, catalogs: Mapping[str, Sequence[Mapping[str, Any]]]) -> None:
        self.catalogs = {
            source: [dict(record) for record in catalogs[source]]
            for source in ("ldaps", "gfs")
        }
        self.raster_metadata_ = validate_raster_catalogs(self.catalogs)

    @staticmethod
    def _device() -> torch.device:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; CPU fallback is forbidden")
        return torch.device("cuda")

    def _rasterize(
        self, tensors: Mapping[str, np.ndarray], transformer: RawSpatiotemporalTransformer
    ) -> dict[str, np.ndarray]:
        return {
            source: rasterize_source(tensors[source], transformer.coords_[source], source)
            for source in ("ldaps", "gfs")
        }

    def fit(
        self,
        features: pd.DataFrame,
        target_cf: pd.DataFrame,
        auxiliary_targets: Mapping[str, pd.DataFrame],
        auxiliary_expected_masks: Mapping[str, pd.Series | np.ndarray],
    ) -> "RawSpatialGridTemporalCNNScadaMultitaskRegressor":
        if not isinstance(target_cf, pd.DataFrame) or not features.index.equals(target_cf.index):
            raise ValueError("feature and target indexes differ")
        target_columns = tuple(map(str, target_cf.columns))
        if not target_columns or any(group not in TARGET_COLS for group in target_columns):
            raise ValueError("target columns must be a non-empty ordered subset")
        requested_keys = _run_keys(features.index)
        direct_values = np.full((len(features), len(TARGET_COLS)), np.nan, np.float32)
        for group in target_columns:
            direct_values[:, TARGET_COLS.index(group)] = target_cf[group].to_numpy(
                dtype=np.float32, copy=False
            )
        direct_by_run = direct_values.reshape(
            -1, EXPECTED_HOURS, len(TARGET_COLS)
        ).transpose(0, 2, 1)
        eligible_by_group = np.isfinite(direct_by_run).all(axis=2)
        eligible_run = eligible_by_group.any(axis=1)
        if not eligible_run.any():
            raise ValueError("no complete 24-hour direct target run")

        aux_hourly, aux_mask_hourly, coverage = prepare_auxiliary_targets(
            features.index, auxiliary_targets, auxiliary_expected_masks
        )
        aux_by_run = aux_hourly.reshape(
            -1, EXPECTED_HOURS, len(TARGET_COLS), len(AUX_CHANNELS)
        ).transpose(0, 2, 3, 1)
        aux_mask_by_run = aux_mask_hourly.reshape(
            -1, EXPECTED_HOURS, len(TARGET_COLS), len(AUX_CHANNELS)
        ).transpose(0, 2, 3, 1)

        eligible_features = features.iloc[np.repeat(eligible_run, EXPECTED_HOURS)]
        direct = direct_by_run[eligible_run]
        direct_group_mask = eligible_by_group[eligible_run]
        direct_filled = np.where(
            direct_group_mask[:, :, None], direct, 0.0
        ).astype(np.float32)
        auxiliary = aux_by_run[eligible_run]
        auxiliary_mask = aux_mask_by_run[eligible_run]

        transformer = RawSpatiotemporalTransformer(self.catalogs)
        node_tensors, fit_keys = transformer.fit_transform(eligible_features)
        tensors = self._rasterize(node_tensors, transformer)
        device = self._device()
        states: list[dict[str, np.ndarray]] = []
        histories: list[dict[str, float | int]] = []
        for seed in SEEDS:
            _configure_determinism(seed)
            model = SpatialGridTemporalConvNetSCADAMultitask().to(device)
            parameter_count = sum(parameter.numel() for parameter in model.parameters())
            if parameter_count != EXPECTED_PARAMETER_COUNT:
                raise AssertionError(f"registered parameter count changed: {parameter_count}")
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
            generator = torch.Generator(device="cpu").manual_seed(seed)
            epoch_totals: list[float] = []
            last_components: dict[str, float] = {}
            for _epoch in range(EPOCHS):
                order = torch.randperm(len(direct_filled), generator=generator).numpy()
                epoch_loss = 0.0
                epoch_weight = 0
                model.train()
                for start in range(0, len(order), BATCH_SIZE):
                    rows = order[start : start + BATCH_SIZE]
                    ldaps = torch.from_numpy(tensors["ldaps"][rows]).to(device)
                    gfs = torch.from_numpy(tensors["gfs"][rows]).to(device)
                    direct_truth = torch.from_numpy(direct_filled[rows]).to(device)
                    direct_mask = torch.from_numpy(direct_group_mask[rows]).to(device)
                    aux_truth = torch.from_numpy(auxiliary[rows]).to(device)
                    aux_mask = torch.from_numpy(auxiliary_mask[rows]).to(device)
                    optimizer.zero_grad(set_to_none=True)
                    direct_prediction, aux_prediction = model(
                        ldaps, gfs, return_auxiliary=True
                    )
                    loss, components = multitask_loss(
                        direct_prediction,
                        aux_prediction,
                        direct_truth,
                        direct_mask,
                        aux_truth,
                        aux_mask,
                    )
                    if not torch.isfinite(loss):
                        raise FloatingPointError("training loss is non-finite")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
                    optimizer.step()
                    epoch_loss += float(loss.detach().cpu()) * len(rows)
                    epoch_weight += len(rows)
                    last_components = {
                        key: float(value.detach().cpu()) for key, value in components.items()
                    }
                scheduler.step()
                epoch_totals.append(epoch_loss / epoch_weight)
            model.eval()
            states.append(_numpy_state_dict(model))
            histories.append(
                {
                    "seed": seed,
                    "first_epoch_loss": epoch_totals[0],
                    "final_epoch_loss": epoch_totals[-1],
                    "minimum_epoch_loss": min(epoch_totals),
                    "final_batch_components": last_components,
                }
            )

        self.transformer_ = transformer
        self.state_dicts_ = states
        self.training_histories_ = histories
        self.parameter_count_ = EXPECTED_PARAMETER_COUNT
        self.device_used_ = str(device)
        self.requested_fit_index_ = features.index.copy()
        self.fit_index_ = eligible_features.index.copy()
        self.requested_fit_run_keys_ = requested_keys
        self.fit_run_keys_ = fit_keys
        self.dropped_run_keys_ = requested_keys[~eligible_run]
        self.target_columns_ = target_columns
        self.eligible_by_group_ = eligible_by_group
        self.auxiliary_coverage_ = coverage
        self.eligible_run_mask_sha256_ = _array_sha256(eligible_run.astype(np.uint8))
        self.eligible_group_mask_sha256_ = _array_sha256(eligible_by_group.astype(np.uint8))
        self.fit_target_sha256_ = _array_sha256(direct_filled)
        self.fit_auxiliary_sha256_ = _array_sha256(auxiliary)
        self.fit_auxiliary_mask_sha256_ = _array_sha256(auxiliary_mask.astype(np.uint8))
        return self

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "state_dicts_"):
            raise RuntimeError("regressor must be fitted first")
        node_tensors, _ = self.transformer_.transform(features)
        tensors = self._rasterize(node_tensors, self.transformer_)
        device = self._device()
        seed_predictions: list[np.ndarray] = []
        for seed, state in zip(SEEDS, self.state_dicts_):
            _configure_determinism(seed)
            model = SpatialGridTemporalConvNetSCADAMultitask().to(device)
            _load_numpy_state_dict(model, state)
            model.eval()
            batches: list[np.ndarray] = []
            with torch.no_grad():
                for start in range(0, len(tensors["ldaps"]), BATCH_SIZE):
                    stop = start + BATCH_SIZE
                    direct = model(
                        torch.from_numpy(tensors["ldaps"][start:stop]).to(device),
                        torch.from_numpy(tensors["gfs"][start:stop]).to(device),
                    )
                    batches.append(direct.cpu().numpy())
            seed_predictions.append(np.concatenate(batches, axis=0))
        prediction = np.mean(np.stack(seed_predictions), axis=0, dtype=np.float64)
        expected = (len(features) // EXPECTED_HOURS, len(TARGET_COLS), EXPECTED_HOURS)
        if prediction.shape != expected or not np.isfinite(prediction).all():
            raise AssertionError("joint prediction shape/finiteness changed")
        hourly = prediction.transpose(0, 2, 1).reshape(len(features), len(TARGET_COLS))
        return pd.DataFrame(
            np.clip(hourly, 0.0, 1.02), index=features.index, columns=TARGET_COLS
        )

    def metadata(self) -> dict[str, Any]:
        if not hasattr(self, "state_dicts_"):
            raise RuntimeError("regressor must be fitted first")
        state_hashes = [
            canonical_sha256({key: _array_sha256(value) for key, value in state.items()})
            for state in self.state_dicts_
        ]
        dropped = [value.isoformat() for value in self.dropped_run_keys_]
        return {
            "model_id": MODEL_ID,
            "architecture": {
                "parameter_count": self.parameter_count_,
                "unchanged_weather_backbone": True,
                "group_specific_direct_cf_heads": True,
                "group_specific_training_only_auxiliary_heads": list(AUX_CHANNELS),
                "auxiliary_input_at_inference": False,
                "raster_metadata": self.raster_metadata_,
            },
            "training": {
                "seeds": list(SEEDS),
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "optimizer": "AdamW",
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "scheduler": "CosineAnnealingLR",
                "smooth_l1_beta": SMOOTH_L1_BETA,
                "low_cf_loss_weight": LOW_CF_LOSS_WEIGHT,
                "auxiliary_loss_weights": {
                    "ws": AUX_WS_WEIGHT,
                    "direction": AUX_DIRECTION_WEIGHT,
                    "availability": AUX_AVAILABILITY_WEIGHT,
                },
                "gradient_clip_norm": GRADIENT_CLIP_NORM,
                "device_used": self.device_used_,
                "histories": self.training_histories_,
            },
            "sequence_alignment": "availability-run 01:00 through next-day 00:00",
            "target_columns_materialized": list(self.target_columns_),
            "fit_start": self.fit_index_.min(),
            "fit_end": self.fit_index_.max(),
            "fit_hourly_rows": len(self.fit_index_),
            "fit_runs": len(self.fit_run_keys_),
            "requested_fit_runs": len(self.requested_fit_run_keys_),
            "dropped_incomplete_target_runs": len(dropped),
            "dropped_incomplete_target_run_keys": dropped,
            "eligible_run_mask_sha256": self.eligible_run_mask_sha256_,
            "eligible_group_mask_sha256": self.eligible_group_mask_sha256_,
            "eligible_runs_by_group": {
                group: int(self.eligible_by_group_[:, position].sum())
                for position, group in enumerate(TARGET_COLS)
            },
            "auxiliary_coverage": self.auxiliary_coverage_,
            "fit_target_sha256": self.fit_target_sha256_,
            "fit_auxiliary_sha256": self.fit_auxiliary_sha256_,
            "fit_auxiliary_mask_sha256": self.fit_auxiliary_mask_sha256_,
            "state_dict_hashes": state_hashes,
            "transformer": self.transformer_.metadata(),
        }


__all__ = [
    "AUX_CHANNELS",
    "CANDIDATE_KEY",
    "EXPECTED_PARAMETER_COUNT",
    "MODEL_ID",
    "RawSpatialGridTemporalCNNScadaMultitaskRegressor",
    "SpatialGridTemporalConvNetSCADAMultitask",
    "multitask_loss",
    "prepare_auxiliary_targets",
]
