"""Compact raw-grid spatiotemporal neural model for complete forecast runs.

The model consumes only the 24-hour LDAPS/GFS wind-node tensors.  A single
node encoder is shared by both weather sources, while each source has its own
geometry-aware spatial attention head.  The two hourly spatial summaries are
fused and processed by a small residual temporal Conv1d stack before emitting
24 direct capacity-factor predictions.

All preprocessing statistics are fitted on the eligible fit prefix.  Baseline
predictions, targets, SCADA, leaderboard feedback, and scale probes are not
model inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from src.raw_grid_wind import SOURCE_CHANNELS, SOURCE_GRID_COUNTS, TIME_FEATURES, canonical_sha256
from src.raw_temporal_kernel import registered_raw_node_columns
from src.run_sequence_residual import validate_complete_runs
from src.temporal import operating_run_key
from src.metric import TARGET_COLS


EXPECTED_HOURS = 24
EXPECTED_RAW_CHANNELS = 200
MODEL_ID = "node32_geoattn_conv48_e240_s2"
NODE_DIM = 32
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
BLEND_WEIGHTS: tuple[float, ...] = (0.025, 0.05)
CANDIDATE_KEYS: tuple[str, ...] = tuple(
    f"{MODEL_ID}_w{int(round(weight * 1000)):03d}" for weight in BLEND_WEIGHTS
)


def _array_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(values)).tobytes()).hexdigest()


def _run_keys(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    validate_complete_runs(index, expected_hours=EXPECTED_HOURS)
    keys = pd.DatetimeIndex(
        operating_run_key(index)[::EXPECTED_HOURS], name="forecast_run_kst_date"
    )
    if not keys.is_unique or not keys.is_monotonic_increasing:
        raise AssertionError("run keys must be unique and increasing")
    return keys


def _ordered_source_columns(features: pd.DataFrame, source: str) -> tuple[str, ...]:
    registered_raw_node_columns(features)
    if source not in SOURCE_CHANNELS:
        raise ValueError(f"unknown source: {source}")
    expected = tuple(
        f"{source}__grid_{grid_id:02d}__{channel}"
        for grid_id in range(1, SOURCE_GRID_COUNTS[source] + 1)
        for channel in SOURCE_CHANNELS[source]
    )
    observed = set(map(str, features.columns[:EXPECTED_RAW_CHANNELS]))
    if set(expected) - observed:
        raise ValueError(f"{source} raw-node schema changed")
    return expected


@dataclass
class RawSpatiotemporalTransformer:
    """Fit-prefix-only source/channel normalization and run tensorization."""

    catalogs: Mapping[str, Sequence[Mapping[str, Any]]]
    scale_floor: float = 1.0e-6

    def _source_values(self, features: pd.DataFrame, source: str) -> np.ndarray:
        columns = _ordered_source_columns(features, source)
        values = features.loc[:, list(columns)].to_numpy(dtype=np.float32, copy=True)
        nodes = SOURCE_GRID_COUNTS[source]
        channels = len(SOURCE_CHANNELS[source])
        values = values.reshape(len(features), nodes, channels)
        if np.isinf(values).any():
            raise ValueError(f"{source} values contain infinity")
        return values

    def _catalog_coords(self, source: str) -> np.ndarray:
        records = list(self.catalogs[source])
        expected_ids = list(range(1, SOURCE_GRID_COUNTS[source] + 1))
        if [int(row["grid_id"]) for row in records] != expected_ids:
            raise ValueError(f"{source} catalog identity/order changed")
        coords = np.asarray(
            [[float(row["latitude"]), float(row["longitude"])] for row in records],
            dtype=np.float32,
        )
        center = coords.mean(axis=0, keepdims=True)
        scale = coords.std(axis=0, keepdims=True)
        if np.any(scale < self.scale_floor):
            raise ValueError(f"{source} coordinate scale is degenerate")
        return ((coords - center) / scale).astype(np.float32)

    def fit(self, features: pd.DataFrame) -> "RawSpatiotemporalTransformer":
        _run_keys(features.index)
        self.raw_columns_ = registered_raw_node_columns(features)
        self.statistics_: dict[str, dict[str, np.ndarray | int]] = {}
        self.coords_: dict[str, np.ndarray] = {}
        for source in ("ldaps", "gfs"):
            values = self._source_values(features, source)
            with np.errstate(all="ignore"):
                medians = np.nanmedian(values, axis=(0, 1))
            if not np.isfinite(medians).all():
                raise ValueError(f"{source} has an entirely missing fit channel")
            missing = np.isnan(values)
            if missing.any():
                _, _, channel_ids = np.where(missing)
                values[missing] = medians[channel_ids]
            means = values.mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
            scales = values.std(axis=(0, 1), dtype=np.float64).astype(np.float32)
            if np.any(scales < self.scale_floor) or not np.isfinite(scales).all():
                raise ValueError(f"{source} has a zero-variance fit channel")
            self.statistics_[source] = {
                "median": medians.astype(np.float32),
                "mean": means,
                "scale": scales,
                "missing_cells": int(missing.sum()),
            }
            self.coords_[source] = self._catalog_coords(source)
        self.fit_hourly_rows_ = len(features)
        self.fit_runs_ = len(features) // EXPECTED_HOURS
        return self

    def transform(
        self, features: pd.DataFrame
    ) -> tuple[dict[str, np.ndarray], pd.DatetimeIndex]:
        if not hasattr(self, "statistics_"):
            raise RuntimeError("transformer must be fitted first")
        if registered_raw_node_columns(features) != self.raw_columns_:
            raise ValueError("raw feature schema/order changed")
        keys = _run_keys(features.index)
        tensors: dict[str, np.ndarray] = {}
        for source in ("ldaps", "gfs"):
            values = self._source_values(features, source)
            stats = self.statistics_[source]
            missing = np.isnan(values)
            if missing.any():
                _, _, channel_ids = np.where(missing)
                values[missing] = np.asarray(stats["median"])[channel_ids]
            values = (values - np.asarray(stats["mean"])[None, None, :]) / np.asarray(
                stats["scale"]
            )[None, None, :]
            if not np.isfinite(values).all():
                raise ValueError(f"{source} normalization left non-finite values")
            tensors[source] = values.reshape(
                len(keys), EXPECTED_HOURS, SOURCE_GRID_COUNTS[source], len(SOURCE_CHANNELS[source])
            ).astype(np.float32, copy=False)
        return tensors, keys

    def fit_transform(
        self, features: pd.DataFrame
    ) -> tuple[dict[str, np.ndarray], pd.DatetimeIndex]:
        return self.fit(features).transform(features)

    def metadata(self) -> dict[str, Any]:
        if not hasattr(self, "statistics_"):
            raise RuntimeError("transformer must be fitted first")
        source_meta: dict[str, Any] = {}
        for source in ("ldaps", "gfs"):
            stats = self.statistics_[source]
            source_meta[source] = {
                "nodes": SOURCE_GRID_COUNTS[source],
                "channels": list(SOURCE_CHANNELS[source]),
                "median_sha256": _array_sha256(np.asarray(stats["median"])),
                "mean_sha256": _array_sha256(np.asarray(stats["mean"])),
                "scale_sha256": _array_sha256(np.asarray(stats["scale"])),
                "coords_sha256": _array_sha256(self.coords_[source]),
                "fit_missing_cells": int(stats["missing_cells"]),
            }
        return {
            "raw_channel_count": len(self.raw_columns_),
            "raw_columns_sha256": canonical_sha256(self.raw_columns_),
            "fit_hourly_rows": self.fit_hourly_rows_,
            "fit_runs": self.fit_runs_,
            "sources": source_meta,
            "normalization": "fit-prefix pooled source/channel mean and std",
            "baseline_calendar_issuance_target_scada_public_scale_features_used": False,
        }


class ResidualTemporalBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(
            channels, channels, kernel_size=3, padding=dilation, dilation=dilation
        )
        self.norm1 = nn.GroupNorm(4, channels)
        self.conv2 = nn.Conv1d(
            channels, channels, kernel_size=3, padding=dilation, dilation=dilation
        )
        self.norm2 = nn.GroupNorm(4, channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        residual = values
        values = F.silu(self.norm1(self.conv1(values)))
        values = self.dropout(values)
        values = self.norm2(self.conv2(values))
        return F.silu(values + residual)


class SharedNodeGeoAttentionConvNet(nn.Module):
    """Shared node encoder, source-specific spatial attention, temporal Conv1d."""

    def __init__(self) -> None:
        super().__init__()
        # 8 normalized source channels + 2 coordinates + 2 source indicators.
        self.node_encoder = nn.Sequential(
            nn.Linear(12, NODE_DIM),
            nn.SiLU(),
            nn.LayerNorm(NODE_DIM),
            nn.Linear(NODE_DIM, NODE_DIM),
            nn.SiLU(),
        )
        self.source_attention = nn.ModuleDict(
            {
                source: nn.Sequential(
                    nn.Linear(NODE_DIM + 2, 16), nn.Tanh(), nn.Linear(16, 1)
                )
                for source in ("ldaps", "gfs")
            }
        )
        self.fusion = nn.Linear(2 * NODE_DIM + 4, TEMPORAL_DIM)
        self.temporal = nn.Sequential(
            *(ResidualTemporalBlock(TEMPORAL_DIM, dilation, DROPOUT) for dilation in TEMPORAL_DILATIONS)
        )
        self.group_heads = nn.ModuleDict(
            {group: nn.Conv1d(TEMPORAL_DIM, 1, kernel_size=1) for group in TARGET_COLS}
        )

    @staticmethod
    def _horizon_features(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        hours = torch.arange(1, EXPECTED_HOURS + 1, device=device, dtype=dtype)
        angle = 2.0 * math.pi * hours / EXPECTED_HOURS
        return torch.stack(
            (torch.sin(angle), torch.cos(angle), torch.sin(2.0 * angle), torch.cos(2.0 * angle)),
            dim=-1,
        )

    def _spatial_summary(
        self, values: torch.Tensor, coords: torch.Tensor, source_index: int, source: str
    ) -> torch.Tensor:
        batch, hours, nodes, _ = values.shape
        geometry = coords.view(1, 1, nodes, 2).expand(batch, hours, nodes, 2)
        indicator = values.new_zeros((batch, hours, nodes, 2))
        indicator[..., source_index] = 1.0
        embedding = self.node_encoder(torch.cat((values, geometry, indicator), dim=-1))
        logits = self.source_attention[source](torch.cat((embedding, geometry), dim=-1))
        attention = torch.softmax(logits, dim=2)
        return torch.sum(attention * embedding, dim=2)

    def forward(
        self,
        ldaps: torch.Tensor,
        gfs: torch.Tensor,
        ldaps_coords: torch.Tensor,
        gfs_coords: torch.Tensor,
    ) -> torch.Tensor:
        ldaps_summary = self._spatial_summary(ldaps, ldaps_coords, 0, "ldaps")
        gfs_summary = self._spatial_summary(gfs, gfs_coords, 1, "gfs")
        position = self._horizon_features(ldaps.device, ldaps.dtype)
        position = position.unsqueeze(0).expand(ldaps.shape[0], -1, -1)
        fused = F.silu(self.fusion(torch.cat((ldaps_summary, gfs_summary, position), dim=-1)))
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
    model.load_state_dict({key: torch.from_numpy(np.asarray(value)) for key, value in state.items()})


class RawSpatiotemporalAttentionRegressor:
    """Fold-causal multitask direct-CF estimator with group-specific heads."""

    def __init__(self, catalogs: Mapping[str, Sequence[Mapping[str, Any]]]) -> None:
        self.catalogs = {
            source: [dict(record) for record in catalogs[source]]
            for source in ("ldaps", "gfs")
        }

    @staticmethod
    def _device() -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(
        self, features: pd.DataFrame, target_cf: pd.DataFrame
    ) -> "RawSpatiotemporalAttentionRegressor":
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
        target_by_run = values.reshape(-1, EXPECTED_HOURS, len(TARGET_COLS)).transpose(0, 2, 1)
        eligible_by_group = np.isfinite(target_by_run).all(axis=2)
        eligible_run = eligible_by_group.any(axis=1)
        if not eligible_run.any():
            raise ValueError("no run has a complete 24-hour target for any registered group")
        eligible_features = features.iloc[np.repeat(eligible_run, EXPECTED_HOURS)]
        target = target_by_run[eligible_run]
        eligible_targets = eligible_by_group[eligible_run]
        target_filled = np.where(eligible_targets[:, :, None], target, 0.0).astype(np.float32)
        transformer = RawSpatiotemporalTransformer(self.catalogs)
        tensors, fit_keys = transformer.fit_transform(eligible_features)
        device = self._device()
        coords = {
            source: torch.from_numpy(transformer.coords_[source]).to(device)
            for source in ("ldaps", "gfs")
        }
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
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
            generator = torch.Generator(device="cpu").manual_seed(seed)
            losses: list[float] = []
            for epoch in range(EPOCHS):
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
                    prediction = model(ldaps, gfs, coords["ldaps"], coords["gfs"])
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
        tensors, _ = self.transformer_.transform(features)
        device = self._device()
        coords = {
            source: torch.from_numpy(self.transformer_.coords_[source]).to(device)
            for source in ("ldaps", "gfs")
        }
        seed_predictions: list[np.ndarray] = []
        for seed, state in zip(SEEDS, self.state_dicts_):
            _configure_determinism(seed)
            model = SharedNodeGeoAttentionConvNet().to(device)
            _load_numpy_state_dict(model, state)
            model.eval()
            batches: list[np.ndarray] = []
            with torch.no_grad():
                for start in range(0, len(tensors["ldaps"]), BATCH_SIZE):
                    stop = start + BATCH_SIZE
                    prediction = model(
                        torch.from_numpy(tensors["ldaps"][start:stop]).to(device),
                        torch.from_numpy(tensors["gfs"][start:stop]).to(device),
                        coords["ldaps"],
                        coords["gfs"],
                    )
                    batches.append(prediction.cpu().numpy())
            seed_predictions.append(np.concatenate(batches, axis=0))
        prediction = np.mean(np.stack(seed_predictions, axis=0), axis=0, dtype=np.float64)
        if prediction.shape != (len(features) // EXPECTED_HOURS, len(TARGET_COLS), EXPECTED_HOURS):
            raise AssertionError("joint neural prediction shape changed")
        if not np.isfinite(prediction).all():
            raise ValueError("joint neural prediction contains non-finite values")
        hourly = prediction.transpose(0, 2, 1).reshape(len(features), len(TARGET_COLS))
        return pd.DataFrame(
            np.clip(hourly, 0.0, 1.02),
            index=features.index,
            columns=TARGET_COLS,
        )

    def metadata(self) -> dict[str, Any]:
        if not hasattr(self, "state_dicts_"):
            raise RuntimeError("regressor must be fitted first")
        dropped = [value.isoformat() for value in self.dropped_run_keys_]
        state_hashes = [
            canonical_sha256({key: _array_sha256(value) for key, value in state.items()})
            for state in self.state_dicts_
        ]
        return {
            "model_id": MODEL_ID,
            "architecture": {
                "shared_node_encoder_dimension": NODE_DIM,
                "source_specific_geometry_attention": True,
                "temporal_dimension": TEMPORAL_DIM,
                "residual_conv_dilations": list(TEMPORAL_DILATIONS),
                "parameter_count": self.parameter_count_,
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
            "target": "multitask direct joint capacity factor 3x24 with group heads",
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


def assert_fit_before_apply(
    fit_index: pd.DatetimeIndex, application_index: pd.DatetimeIndex
) -> dict[str, Any]:
    _run_keys(fit_index)
    _run_keys(application_index)
    if len(fit_index.intersection(application_index)) or fit_index.max() >= application_index.min():
        raise ValueError("fit period must be disjoint and strictly before application")
    return {
        "fit_start": fit_index.min(),
        "fit_end": fit_index.max(),
        "application_start": application_index.min(),
        "application_end": application_index.max(),
        "fit_end_before_application_start": True,
        "overlap_count": 0,
        "fit_runs_complete": True,
        "application_runs_complete": True,
    }


def blend_attention_prediction(
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
    for weight in BLEND_WEIGHTS:
        key = f"{MODEL_ID}_w{int(round(weight * 1000)):03d}"
        result[key] = blend_attention_prediction(
            baseline_kwh, model_cf, capacity_kwh=capacity_kwh, weight=weight
        )
    if tuple(result.columns) != CANDIDATE_KEYS:
        raise AssertionError("attention candidate order changed")
    identity = blend_attention_prediction(
        baseline_kwh, model_cf, capacity_kwh=capacity_kwh, weight=0.0
    )
    if identity.to_numpy().tobytes() != baseline_kwh.to_numpy().tobytes():
        raise AssertionError("weight-zero identity is not value-bit exact")
    return result


def parse_candidate_key(key: str) -> float:
    for weight in BLEND_WEIGHTS:
        if key == f"{MODEL_ID}_w{int(round(weight * 1000)):03d}":
            return weight
    raise ValueError(f"candidate is not preregistered: {key}")


def select_group_candidate(
    comparisons: Mapping[str, Mapping[str, Mapping[str, Any]]],
    required_slices: Sequence[str],
) -> tuple[str | None, dict[str, Any]]:
    if set(comparisons) != set(CANDIDATE_KEYS) or len(comparisons) != len(CANDIDATE_KEYS):
        raise ValueError("candidate key set changed")
    required = tuple(required_slices)
    audit: dict[str, Any] = {}
    eligible: list[str] = []
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
        row = {
            "weight": parse_candidate_key(key),
            "deltas": deltas,
            "minimum": min(deltas.values()),
            "mean": float(np.mean(list(deltas.values()))),
            "all_slices_strictly_positive": all(value > 0.0 for value in deltas.values()),
        }
        audit[key] = row
        if row["all_slices_strictly_positive"]:
            eligible.append(key)
    if not eligible:
        return None, {"candidates": audit, "selected": "identity"}
    selected = max(
        eligible,
        key=lambda key: (
            audit[key]["minimum"], audit[key]["mean"], -audit[key]["weight"]
        ),
    )
    return selected, {"candidates": audit, "selected": selected}


__all__ = [
    "BATCH_SIZE",
    "BLEND_WEIGHTS",
    "CANDIDATE_KEYS",
    "EPOCHS",
    "MODEL_ID",
    "RawSpatiotemporalAttentionRegressor",
    "RawSpatiotemporalTransformer",
    "SharedNodeGeoAttentionConvNet",
    "assert_fit_before_apply",
    "blend_attention_prediction",
    "candidate_frame",
    "parse_candidate_key",
    "select_group_candidate",
]
