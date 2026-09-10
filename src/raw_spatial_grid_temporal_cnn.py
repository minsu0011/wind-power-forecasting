"""Local spatial-grid Conv2d followed by residual temporal Conv1d.

Only the registered raw LDAPS/GFS node tensors enter the model.  LDAPS is
rasterized losslessly onto its physical 4x5 sparse lattice and GFS onto its
dense 3x3 lattice.  Fit-prefix normalization and availability-run grouping are
inherited from the audited raw spatiotemporal transformer.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from src.metric import CAPACITY_KWH, TARGET_COLS
from src.raw_grid_wind import SOURCE_CHANNELS, SOURCE_GRID_COUNTS, canonical_sha256
from src.raw_spatiotemporal_attention import RawSpatiotemporalTransformer, ResidualTemporalBlock
from src.run_sequence_residual import validate_complete_runs
from src.temporal import operating_run_key


EXPECTED_HOURS = 24
MODEL_ID = "spatial16x24_maskconv_temporal48_e240_s2"
SPATIAL_HIDDEN = 16
SPATIAL_OUTPUT = 24
TEMPORAL_DIM = 48
TEMPORAL_DILATIONS: tuple[int, ...] = (1, 2)
SEEDS: tuple[int, ...] = (42, 2026)
EPOCHS = 240
BATCH_SIZE = 32
LEARNING_RATE = 2.0e-3
WEIGHT_DECAY = 1.0e-3
DROPOUT = 0.10
SMOOTH_L1_BETA = 0.05
LOW_CF_LOSS_WEIGHT = 0.25
GRADIENT_CLIP_NORM = 1.0
BLEND_WEIGHT = 0.025
CANDIDATE_KEY = f"{MODEL_ID}_w025"

LDAPS_LAYOUT = np.asarray(
    [[0, 1, 2, 3, 0], [4, 5, 6, 7, 8], [9, 10, 11, 12, 13], [0, 14, 15, 16, 0]],
    dtype=np.int64,
)
GFS_LAYOUT = np.arange(1, 10, dtype=np.int64).reshape(3, 3)
LAYOUTS: Mapping[str, np.ndarray] = {"ldaps": LDAPS_LAYOUT, "gfs": GFS_LAYOUT}


def _array_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(values)).tobytes()).hexdigest()


def _run_keys(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    validate_complete_runs(index, expected_hours=EXPECTED_HOURS)
    keys = pd.DatetimeIndex(
        operating_run_key(index)[::EXPECTED_HOURS], name="forecast_run_kst_date"
    )
    if not keys.is_unique or not keys.is_monotonic_increasing:
        raise AssertionError("availability-run keys must be unique and increasing")
    return keys


def validate_raster_catalogs(
    catalogs: Mapping[str, Sequence[Mapping[str, Any]]]
) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for source in ("ldaps", "gfs"):
        records = list(catalogs[source])
        ids = [int(record["grid_id"]) for record in records]
        expected = list(range(1, SOURCE_GRID_COUNTS[source] + 1))
        if ids != expected:
            raise ValueError(f"{source} catalog identity/order changed")
        layout = LAYOUTS[source]
        occupied = sorted(int(value) for value in layout.ravel() if value)
        if occupied != expected or len(occupied) != len(set(occupied)):
            raise AssertionError(f"{source} raster is not a lossless node permutation")
        coords = {
            int(record["grid_id"]): (
                float(record["latitude"]), float(record["longitude"])
            )
            for record in records
        }
        for row in layout:
            row_ids = [int(value) for value in row if value]
            row_lons = [coords[value][1] for value in row_ids]
            if any(right <= left for left, right in zip(row_lons, row_lons[1:])):
                raise ValueError(f"{source} raster longitude order changed")
        row_lats = [
            float(np.mean([coords[int(value)][0] for value in row if value]))
            for row in layout
        ]
        if any(south >= north for north, south in zip(row_lats, row_lats[1:])):
            raise ValueError(f"{source} raster latitude order changed")
        metadata[source] = {
            "shape": list(layout.shape),
            "layout": layout.tolist(),
            "occupied_cells": len(occupied),
            "missing_cells": int((layout == 0).sum()),
            "layout_sha256": _array_sha256(layout),
            "north_to_south_west_to_east": True,
            "lossless_node_permutation": True,
        }
    return metadata


def rasterize_source(
    values: np.ndarray,
    coords: np.ndarray,
    source: str,
) -> np.ndarray:
    """Losslessly place normalized node values and fixed geometry on a raster."""

    if source not in LAYOUTS:
        raise ValueError(f"unknown source: {source}")
    values = np.asarray(values, dtype=np.float32)
    coords = np.asarray(coords, dtype=np.float32)
    nodes = SOURCE_GRID_COUNTS[source]
    channels = len(SOURCE_CHANNELS[source])
    if values.ndim != 4 or values.shape[2:] != (nodes, channels):
        raise ValueError(f"{source} tensor shape changed: {values.shape}")
    if coords.shape != (nodes, 2) or not np.isfinite(coords).all():
        raise ValueError(f"{source} coordinate tensor changed")
    layout = LAYOUTS[source]
    result = np.zeros(
        (values.shape[0], values.shape[1], channels + 3, *layout.shape),
        dtype=np.float32,
    )
    for row in range(layout.shape[0]):
        for column in range(layout.shape[1]):
            grid_id = int(layout[row, column])
            if not grid_id:
                continue
            node = grid_id - 1
            result[:, :, :channels, row, column] = values[:, :, node, :]
            result[:, :, channels, row, column] = 1.0
            result[:, :, channels + 1 :, row, column] = coords[node]
    recovered = np.empty_like(values)
    for row in range(layout.shape[0]):
        for column in range(layout.shape[1]):
            grid_id = int(layout[row, column])
            if grid_id:
                recovered[:, :, grid_id - 1, :] = result[:, :, :channels, row, column]
    if not np.array_equal(recovered, values):
        raise AssertionError(f"{source} rasterization is not value-bit lossless")
    mask = result[:, :, channels]
    if not np.all(mask.sum(axis=(-2, -1)) == nodes):
        raise AssertionError(f"{source} raster occupancy changed")
    if not np.isfinite(result).all():
        raise ValueError(f"{source} raster contains non-finite values")
    return result


class MaskedSpatialEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(11, SPATIAL_HIDDEN, kernel_size=3, padding=1, bias=True)
        self.norm1 = nn.GroupNorm(4, SPATIAL_HIDDEN)
        self.conv2 = nn.Conv2d(
            SPATIAL_HIDDEN, SPATIAL_OUTPUT, kernel_size=3, padding=1, bias=True
        )
        self.norm2 = nn.GroupNorm(4, SPATIAL_OUTPUT)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 4 or values.shape[1] != 11:
            raise ValueError("spatial raster shape changed")
        mask = values[:, 8:9]
        encoded = F.silu(self.norm1(self.conv1(values)))
        encoded = F.silu(self.norm2(self.conv2(encoded))) * mask
        denominator = torch.sum(mask, dim=(-2, -1))
        if torch.any(denominator <= 0):
            raise ValueError("spatial raster has no occupied cell")
        return torch.sum(encoded, dim=(-2, -1)) / denominator


class SpatialGridTemporalConvNet(nn.Module):
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

    @staticmethod
    def _horizon_features(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        hours = torch.arange(1, EXPECTED_HOURS + 1, device=device, dtype=dtype)
        angle = 2.0 * math.pi * hours / EXPECTED_HOURS
        return torch.stack(
            (
                torch.sin(angle), torch.cos(angle),
                torch.sin(2.0 * angle), torch.cos(2.0 * angle),
            ),
            dim=-1,
        )

    def forward(self, ldaps: torch.Tensor, gfs: torch.Tensor) -> torch.Tensor:
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
        temporal = self.temporal(fused.transpose(1, 2))
        return torch.stack(
            [
                1.02 * torch.sigmoid(self.group_heads[group](temporal).squeeze(1))
                for group in TARGET_COLS
            ],
            dim=1,
        )


def _configure_determinism(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _numpy_state_dict(model: nn.Module) -> dict[str, np.ndarray]:
    return {
        key: value.detach().cpu().numpy().copy()
        for key, value in model.state_dict().items()
    }


def _load_numpy_state_dict(model: nn.Module, state: Mapping[str, np.ndarray]) -> None:
    model.load_state_dict(
        {key: torch.from_numpy(np.asarray(value)) for key, value in state.items()}
    )


class RawSpatialGridTemporalCNNRegressor:
    """Deterministic two-seed multitask direct-CF estimator."""

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
        self, features: pd.DataFrame, target_cf: pd.DataFrame
    ) -> "RawSpatialGridTemporalCNNRegressor":
        if not isinstance(target_cf, pd.DataFrame) or not features.index.equals(target_cf.index):
            raise ValueError("feature and target indexes differ")
        target_columns = tuple(map(str, target_cf.columns))
        if not target_columns or any(group not in TARGET_COLS for group in target_columns):
            raise ValueError("target columns must be a non-empty ordered subset")
        if len(set(target_columns)) != len(target_columns):
            raise ValueError("target columns contain duplicates")
        requested_keys = _run_keys(features.index)
        values = np.full((len(features), len(TARGET_COLS)), np.nan, dtype=np.float32)
        for column in target_columns:
            values[:, TARGET_COLS.index(column)] = target_cf[column].to_numpy(
                dtype=np.float32, copy=False
            )
        target_by_run = values.reshape(-1, EXPECTED_HOURS, len(TARGET_COLS)).transpose(0, 2, 1)
        eligible_by_group = np.isfinite(target_by_run).all(axis=2)
        eligible_run = eligible_by_group.any(axis=1)
        if not eligible_run.any():
            raise ValueError("no complete 24-hour target run")
        eligible_features = features.iloc[np.repeat(eligible_run, EXPECTED_HOURS)]
        target = target_by_run[eligible_run]
        eligible_targets = eligible_by_group[eligible_run]
        target_filled = np.where(eligible_targets[:, :, None], target, 0.0).astype(np.float32)

        transformer = RawSpatiotemporalTransformer(self.catalogs)
        node_tensors, fit_keys = transformer.fit_transform(eligible_features)
        tensors = self._rasterize(node_tensors, transformer)
        device = self._device()
        states: list[dict[str, np.ndarray]] = []
        histories: list[dict[str, float | int]] = []
        parameter_count: int | None = None
        for seed in SEEDS:
            _configure_determinism(seed)
            model = SpatialGridTemporalConvNet().to(device)
            count = sum(parameter.numel() for parameter in model.parameters())
            if parameter_count is None:
                parameter_count = count
            elif parameter_count != count:
                raise AssertionError("parameter count changed between seeds")
            if count != 41235:
                raise AssertionError(f"registered parameter count changed: {count}")
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
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
                    valid_group = torch.from_numpy(eligible_targets[rows]).to(device)
                    optimizer.zero_grad(set_to_none=True)
                    prediction = model(ldaps, gfs)
                    element = F.smooth_l1_loss(
                        prediction, truth, reduction="none", beta=SMOOTH_L1_BETA
                    )
                    weights = torch.where(
                        truth >= 0.10,
                        torch.ones_like(truth),
                        torch.full_like(truth, LOW_CF_LOSS_WEIGHT),
                    )
                    weights = weights * valid_group[:, :, None]
                    loss = torch.sum(element * weights) / torch.sum(weights)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("training loss is non-finite")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
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
        self.eligible_by_group_ = eligible_by_group
        self.eligible_run_mask_sha256_ = _array_sha256(eligible_run.astype(np.uint8))
        self.eligible_group_mask_sha256_ = _array_sha256(eligible_by_group.astype(np.uint8))
        self.fit_target_sha256_ = _array_sha256(target_filled)
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
            model = SpatialGridTemporalConvNet().to(device)
            _load_numpy_state_dict(model, state)
            model.eval()
            batches: list[np.ndarray] = []
            with torch.no_grad():
                for start in range(0, len(tensors["ldaps"]), BATCH_SIZE):
                    stop = start + BATCH_SIZE
                    prediction = model(
                        torch.from_numpy(tensors["ldaps"][start:stop]).to(device),
                        torch.from_numpy(tensors["gfs"][start:stop]).to(device),
                    )
                    batches.append(prediction.cpu().numpy())
            seed_predictions.append(np.concatenate(batches, axis=0))
        prediction = np.mean(np.stack(seed_predictions, axis=0), axis=0, dtype=np.float64)
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
                "spatial_encoder": "source-specific Conv2d 11->16->24 kernel3 masked mean",
                "raster_metadata": self.raster_metadata_,
                "fusion": "Linear52->48",
                "temporal_dimension": TEMPORAL_DIM,
                "residual_conv_dilations": list(TEMPORAL_DILATIONS),
                "group_specific_direct_cf_heads": True,
                "global_node_attention": False,
            },
            "training": {
                "seeds": list(SEEDS),
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "optimizer": "AdamW",
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "scheduler": "CosineAnnealingLR",
                "loss": "weighted SmoothL1 direct CF",
                "smooth_l1_beta": SMOOTH_L1_BETA,
                "low_cf_loss_weight": LOW_CF_LOSS_WEIGHT,
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
            "fit_target_sha256": self.fit_target_sha256_,
            "state_dict_hashes": state_hashes,
            "transformer": self.transformer_.metadata(),
        }


def blend_prediction(
    baseline_kwh: pd.Series,
    model_cf: pd.Series,
    *,
    capacity_kwh: float,
    weight: float = BLEND_WEIGHT,
) -> pd.Series:
    if weight not in (0.0, BLEND_WEIGHT):
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


def candidate_series(
    baseline_kwh: pd.Series, model_cf: pd.Series, *, capacity_kwh: float
) -> pd.Series:
    candidate = blend_prediction(
        baseline_kwh, model_cf, capacity_kwh=capacity_kwh, weight=BLEND_WEIGHT
    )
    identity = blend_prediction(
        baseline_kwh, model_cf, capacity_kwh=capacity_kwh, weight=0.0
    )
    if identity.to_numpy().tobytes() != baseline_kwh.to_numpy().tobytes():
        raise AssertionError("weight-zero identity is not value-bit exact")
    return candidate.rename(CANDIDATE_KEY)


def stage1_group_gate(
    comparison: Mapping[str, Mapping[str, Any]], required_slices: Sequence[str]
) -> dict[str, Any]:
    required = tuple(required_slices)
    if tuple(comparison) != required or "full" not in required:
        raise ValueError("registered Stage1 slice order changed")
    deltas: dict[str, float] = {}
    for name in required:
        row = comparison[name]
        exact = float(row["candidate"]["score"]) - float(row["baseline"]["score"])
        if float(row["delta"]) != exact:
            raise AssertionError("Stage1 delta arithmetic changed")
        deltas[name] = exact
    full = comparison["full"]
    components = {
        "one_minus_nmae": float(full["candidate"]["one_minus_nmae"])
        - float(full["baseline"]["one_minus_nmae"]),
        "ficr": float(full["candidate"]["ficr"])
        - float(full["baseline"]["ficr"]),
    }
    passed = all(value > 0.0 for value in deltas.values()) and all(
        value >= 0.0 for value in components.values()
    )
    return {
        "candidate": CANDIDATE_KEY,
        "deltas": deltas,
        "minimum": min(deltas.values()),
        "mean": float(np.mean(list(deltas.values()))),
        "full_component_deltas": components,
        "passed": passed,
    }


__all__ = [
    "BATCH_SIZE", "BLEND_WEIGHT", "CANDIDATE_KEY", "EPOCHS", "GFS_LAYOUT",
    "LDAPS_LAYOUT", "MODEL_ID", "RawSpatialGridTemporalCNNRegressor",
    "SpatialGridTemporalConvNet", "blend_prediction", "candidate_series",
    "rasterize_source", "stage1_group_gate", "validate_raster_catalogs",
]
