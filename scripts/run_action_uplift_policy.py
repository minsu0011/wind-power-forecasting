"""Run the preregistered baseline-conditioned fixed-action uplift policy."""

from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from scripts import run_shared_q07_multiseed as shared  # noqa: E402
from src.action_uplift_policy import (  # noqa: E402
    ACTION_FACTORS,
    CLASSIFIER_PARAMETERS,
    PROBABILITY_THRESHOLD,
    REGRESSOR_PARAMETERS,
    ActionUpliftPolicy,
    compare_group_blocks,
    passes_all_blocks,
    transfer_delta,
)
from src.manifest import (  # noqa: E402
    describe_file,
    git_state,
    package_versions,
    sha256_file,
    utc_now,
    write_json_atomic,
)
from src.metric import CAPACITY_KWH, TARGET_COLS, score_details  # noqa: E402
from src.residual_histogram_bayes import (  # noqa: E402
    COMPONENT_NAMES,
    META_FEATURE_COLUMNS,
    WEATHER_COLUMNS,
    build_meta_features,
)


PREREGISTER_SHA256 = "294c80126005b7cf01a862cf008fa3f3b765c0a300af85d0a471b41d35f942aa"
BASE_PREREGISTER_SHA256 = "0a4562ff614d328ffbc77e51c28cb9fd710315e5cf69e989cd5fa05954c1e369"
YEAR_2022_TO_2023 = pd.date_range(
    "2022-01-01 01:00:00", "2024-01-01 00:00:00", freq="h", name="forecast_kst_dtm"
)
YEAR_2022_TO_2024 = pd.date_range(
    "2022-01-01 01:00:00", "2025-01-01 00:00:00", freq="h", name="forecast_kst_dtm"
)
YEAR_2023 = pd.date_range(
    "2023-01-01 01:00:00", "2024-01-01 00:00:00", freq="h", name="forecast_kst_dtm"
)
YEAR_2023_H1 = pd.date_range(
    "2023-01-01 01:00:00", "2023-07-01 00:00:00", freq="h", name="forecast_kst_dtm"
)
YEAR_2023_H2 = pd.date_range(
    "2023-07-01 01:00:00", "2024-01-01 00:00:00", freq="h", name="forecast_kst_dtm"
)
YEAR_2023_Q3 = pd.date_range(
    "2023-07-01 01:00:00", "2023-10-01 00:00:00", freq="h", name="forecast_kst_dtm"
)
YEAR_2023_Q4 = pd.date_range(
    "2023-10-01 01:00:00", "2024-01-01 00:00:00", freq="h", name="forecast_kst_dtm"
)
YEAR_2024 = pd.date_range(
    "2024-01-01 01:00:00", "2025-01-01 00:00:00", freq="h", name="forecast_kst_dtm"
)
YEAR_2025 = pd.date_range(
    "2025-01-01 01:00:00", "2026-01-01 00:00:00", freq="h", name="forecast_kst_dtm"
)
G3_COLUMN_MAP = {
    "lgb_l1": "l1",
    "lgb_q07": "q07",
    "shared_l1": "shared_l1",
    "shared_q07": "shared_q07",
    "top200_q07": "top200q07",
    "energy_q06": "ewq06",
}
STAGE2_BLOCKS = {
    "full": ("2024-01-01 01:00:00", "2025-01-01 00:00:00"),
    "H1": ("2024-01-01 01:00:00", "2024-07-01 00:00:00"),
    "H2": ("2024-07-01 01:00:00", "2025-01-01 00:00:00"),
    "Q1": ("2024-01-01 01:00:00", "2024-04-01 00:00:00"),
    "Q2": ("2024-04-01 01:00:00", "2024-07-01 00:00:00"),
    "Q3": ("2024-07-01 01:00:00", "2024-10-01 00:00:00"),
    "Q4": ("2024-10-01 01:00:00", "2025-01-01 00:00:00"),
}
EXPECTED_SOURCE_CLOSURE: tuple[str, ...] = (
    "scripts/run_action_uplift_policy.py",
    "scripts/run_shared_q07_multiseed.py",
    "src/action_uplift_policy.py",
    "src/features.py",
    "src/manifest.py",
    "src/metric.py",
    "src/residual_histogram_bayes.py",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("stage1", "stage2", "final", "all"), required=True)
    parser.add_argument("--raw-dir", type=Path, default=Path(r"data/local/open"))
    parser.add_argument("--cache-dir", type=Path, default=Path("artifacts/cache"))
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--out-dir", type=Path, default=Path("artifacts/postgate/action_uplift_policy_strict_v2")
    )
    parser.add_argument(
        "--preregister", type=Path, default=Path("configs/action_uplift_policy_preregister_v2.json")
    )
    return parser.parse_args(argv)


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        frame.to_parquet(temporary, engine="pyarrow")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_joblib(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        joblib.dump(payload, temporary, compress=3)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    write_json_atomic(path, json.loads(json.dumps(payload, default=str)))


def _load_preregister(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    if sha256_file(path) != PREREGISTER_SHA256:
        raise AssertionError("v2 action-uplift preregistration changed")
    sidecar = path.with_suffix(".sha256")
    expected_sidecar = f"{PREREGISTER_SHA256}  {path.name}"
    if sidecar.read_text(encoding="utf-8").strip() != expected_sidecar:
        raise AssertionError("v2 preregistration sidecar changed")
    v2 = json.loads(path.read_text(encoding="utf-8"))
    base_path = PROJECT_DIR / v2["supersedes"]["path"]
    if sha256_file(base_path) != BASE_PREREGISTER_SHA256:
        raise AssertionError("superseded v1 preregistration changed")
    base = json.loads(base_path.read_text(encoding="utf-8"))
    base["experiment_id"] = "action_uplift_policy_strict_v2"
    base["status"] = v2["status"]
    base["stage1_component_identities"]["top200_q07"]["sha256"] = v2[
        "corrected_input_identity"
    ]["v2_correct_sha256"]
    base["final"]["output_csv"] = v2["v2_overrides"]["final.output_csv"]
    base["output_contract"]["directory"] = v2["v2_overrides"]["output_contract.directory"]
    base["source_contract"]["expected_static_source_closure"] = v2["v2_overrides"][
        "source_contract.expected_static_source_closure"
    ]
    if tuple(base["fixed_actions"]["learned_order"]) != ACTION_FACTORS:
        raise AssertionError("fixed action order changed")
    if float(base["fixed_actions"]["classifier_probability_threshold"]) != PROBABILITY_THRESHOLD:
        raise AssertionError("policy probability threshold changed")
    classifier = dict(base["models"]["classifier"]["parameters"])
    classifier.update(
        objective=base["models"]["classifier"]["objective"],
        random_state=base["models"]["classifier"]["random_state"],
    )
    regressor = dict(base["models"]["magnitude_regressor"]["parameters"])
    regressor.update(
        objective=base["models"]["magnitude_regressor"]["objective"],
        random_state=base["models"]["magnitude_regressor"]["random_state"],
    )
    if classifier != CLASSIFIER_PARAMETERS or regressor != REGRESSOR_PARAMETERS:
        raise AssertionError("fixed model parameters changed")
    if tuple(base["features"]["component_order"]) != COMPONENT_NAMES:
        raise AssertionError("component order changed")
    if tuple(base["features"]["base_state_columns"]) != META_FEATURE_COLUMNS:
        raise AssertionError("base feature order changed")
    if tuple(base["features"]["weather_source_columns"]) != WEATHER_COLUMNS:
        raise AssertionError("weather feature order changed")
    if tuple(base["source_contract"]["expected_static_source_closure"]) != EXPECTED_SOURCE_CLOSURE:
        raise AssertionError("source closure changed")
    if v2["supersedes"]["v1_action_uplift_fit_count"] != 0:
        raise AssertionError("v1 no-fit incident changed")
    return base, {"v1": describe_file(base_path), "v2": describe_file(path), "v2_sidecar": describe_file(sidecar)}


def _source_paths(preregister_path: Path) -> dict[str, Path]:
    paths = {relative: PROJECT_DIR / relative for relative in EXPECTED_SOURCE_CLOSURE}
    paths["tests/test_action_uplift_policy.py"] = PROJECT_DIR / "tests/test_action_uplift_policy.py"
    paths["configs/action_uplift_policy_preregister_v1.json"] = PROJECT_DIR / "configs/action_uplift_policy_preregister_v1.json"
    paths["configs/action_uplift_policy_preregister_v1.sha256"] = PROJECT_DIR / "configs/action_uplift_policy_preregister_v1.sha256"
    paths["configs/action_uplift_policy_preregister_v2.json"] = preregister_path
    paths["configs/action_uplift_policy_preregister_v2.sha256"] = preregister_path.with_suffix(".sha256")
    return paths


def _write_source_lock(out_dir: Path, preregister_path: Path, preregister_evidence: Mapping[str, Any]) -> Path:
    source_files = {name: describe_file(path) for name, path in _source_paths(preregister_path).items()}
    path = out_dir / "source_lock_before_fit.json"
    _write_json(
        path,
        {
            "preregister_sha256": PREREGISTER_SHA256,
            "preregister_evidence": preregister_evidence,
            "source_files": source_files,
            "source_files_sha256": _canonical_sha256(source_files),
            "fit_count_before_lock": 0,
            "prediction_count_before_lock": 0,
            "candidate_score_count_before_lock": 0,
            "public_or_scale_artifact_bytes_read": 0,
        },
    )
    return path


def _verify_source_lock(out_dir: Path, preregister_path: Path) -> dict[str, Any]:
    path = out_dir / "source_lock_before_fit.json"
    lock = json.loads(path.read_text(encoding="utf-8"))
    current = {name: describe_file(value) for name, value in _source_paths(preregister_path).items()}
    if current != lock["source_files"] or _canonical_sha256(current) != lock["source_files_sha256"]:
        raise AssertionError("source closure changed after pre-fit lock")
    return lock


def _assert_identity(path: Path, spec: Mapping[str, Any]) -> dict[str, Any]:
    record = describe_file(path)
    if int(record["size_bytes"]) != int(spec["bytes"]) or record["sha256"] != spec["sha256"]:
        raise AssertionError(f"input identity changed: {path}")
    return record


def _read_label_prefix(
    path: Path, spec: Mapping[str, Any], groups: Sequence[str]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = int(spec["data_rows"])
    byte_limit = int(spec["bytes"])
    digest, observed_bytes = shared._csv_prefix_identity(path, data_rows=rows, byte_limit=byte_limit)
    if observed_bytes != byte_limit or digest != spec["sha256"]:
        raise AssertionError("label prefix identity changed")
    bounded_raw = shared._BoundedRawReader(path, byte_limit=byte_limit)
    try:
        with io.BufferedReader(bounded_raw, buffer_size=1024 * 1024) as bounded:
            frame = pd.read_csv(
                bounded,
                usecols=["kst_dtm", *groups],
                nrows=rows,
                encoding="utf-8-sig",
                memory_map=False,
            )
            bytes_returned = bounded_raw.bytes_returned
            position = bounded_raw.underlying_position
    finally:
        bounded_raw.close()
    if bytes_returned != byte_limit or position != byte_limit:
        raise AssertionError("label reader crossed physical prefix")
    if tuple(frame.columns) != ("kst_dtm", *groups) or len(frame) != rows:
        raise AssertionError("label prefix schema changed")
    index = pd.to_datetime(frame.pop("kst_dtm"), errors="raise")
    frame.index = pd.DatetimeIndex(index, name="forecast_kst_dtm")
    frame = frame.astype(np.float64)
    if frame.index.max() != pd.Timestamp(spec["end"]):
        raise AssertionError("label prefix end changed")
    next_timestamp = shared._next_csv_first_field(path, after_data_rows=rows, prefix_bytes=byte_limit)
    if next_timestamp != spec["next_row_first_field_only"]:
        raise AssertionError("label boundary timestamp changed")
    return frame, {
        "path": str(path.resolve()),
        "groups_materialized": list(groups),
        "rows": rows,
        "physical_byte_limit": byte_limit,
        "physical_prefix_sha256": digest,
        "physical_bytes_returned": bytes_returned,
        "underlying_position_after_parse": position,
        "suffix_bytes_exposed_to_parser": 0,
        "start": frame.index.min(),
        "end": frame.index.max(),
        "next_row_timestamp_only": next_timestamp,
        "next_row_target_values_read": False,
    }


def _read_full_labels(raw_dir: Path, preregister: Mapping[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    spec = preregister["full_label_identity_after_stage2_prescore_lock"]
    path = raw_dir / "train/train_labels.csv"
    evidence = _assert_identity(path, spec)
    frame = pd.read_csv(path, encoding="utf-8-sig")
    if tuple(frame.columns) != ("kst_dtm", *TARGET_COLS) or len(frame) != int(spec["rows"]):
        raise AssertionError("full label schema changed")
    index = pd.to_datetime(frame.pop("kst_dtm"), errors="raise")
    frame.index = pd.DatetimeIndex(index, name="forecast_kst_dtm")
    frame = frame.astype(np.float64)
    if not frame.index.equals(YEAR_2022_TO_2024) or np.isinf(frame.to_numpy()).any():
        raise AssertionError("full label values/index changed")
    return frame, evidence


def _read_prediction(
    path: Path, index: pd.DatetimeIndex, *, required_columns: Sequence[str]
) -> pd.DataFrame:
    frame = pd.read_parquet(path, engine="pyarrow")
    frame.index = pd.DatetimeIndex(frame.index, name="forecast_kst_dtm")
    if not frame.index.equals(index):
        raise AssertionError(f"prediction index changed: {path}")
    missing = [column for column in required_columns if column not in frame]
    if missing:
        raise AssertionError(f"prediction columns changed: {path}: {missing}")
    frame = frame.loc[:, list(required_columns)].astype(np.float64)
    if not np.isfinite(frame.to_numpy()).all():
        raise AssertionError(f"prediction contains non-finite values: {path}")
    return frame


def _load_recipe(preregister: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    spec = preregister["locked_v3_assembly"]
    path = PROJECT_DIR / spec["recipe_path"]
    evidence = _assert_identity(path, {"bytes": spec["recipe_bytes"], "sha256": spec["recipe_sha256"]})
    recipe = json.loads(path.read_text(encoding="utf-8"))
    for group in TARGET_COLS:
        if tuple(recipe["ensemble"]["weights"][group]) != COMPONENT_NAMES:
            raise AssertionError("recipe component order changed")
    return recipe, evidence


def _load_stage1_sources(
    preregister: Mapping[str, Any], recipe: Mapping[str, Any]
) -> tuple[dict[str, dict[str, pd.Series]], dict[str, pd.Series], dict[str, Any]]:
    identities = preregister["stage1_component_identities"]
    evidence: dict[str, Any] = {}
    dev: dict[str, pd.DataFrame] = {}
    for component in COMPONENT_NAMES:
        spec = identities[component]
        path = PROJECT_DIR / spec["path"]
        evidence[component] = _assert_identity(path, spec)
        dev[component] = _read_prediction(path, YEAR_2023, required_columns=TARGET_COLS[:2])
    g3_spec = identities["g3_components"]
    g3_path = PROJECT_DIR / g3_spec["path"]
    evidence["g3_components"] = _assert_identity(g3_path, g3_spec)
    g3 = _read_prediction(g3_path, YEAR_2023_H2, required_columns=tuple(G3_COLUMN_MAP.values()))
    components: dict[str, dict[str, pd.Series]] = {
        group: {component: dev[component][group].rename(group) for component in COMPONENT_NAMES}
        for group in TARGET_COLS[:2]
    }
    components["kpx_group_3"] = {
        component: g3[column].rename("kpx_group_3") for component, column in G3_COLUMN_MAP.items()
    }
    base_spec = identities["g12_corrected_v3"]
    base_path = PROJECT_DIR / base_spec["path"]
    evidence["g12_corrected_v3"] = _assert_identity(base_path, base_spec)
    g12 = _read_prediction(base_path, YEAR_2023, required_columns=TARGET_COLS[:2])
    weights = recipe["ensemble"]["weights"]["kpx_group_3"]
    weighted = sum(
        float(weights[name]) * components["kpx_group_3"][name].to_numpy(dtype=np.float64)
        for name in COMPONENT_NAMES
    )
    affine = recipe["ensemble"]["affine"]["kpx_group_3"]
    g3_base = np.clip(
        float(affine["scale"]) * weighted + float(affine["bias_kwh"]),
        0.0,
        1.02 * CAPACITY_KWH["kpx_group_3"],
    )
    baseline = {
        "kpx_group_1": g12["kpx_group_1"].rename("kpx_group_1"),
        "kpx_group_2": g12["kpx_group_2"].rename("kpx_group_2"),
        "kpx_group_3": pd.Series(g3_base, index=YEAR_2023_H2, name="kpx_group_3"),
    }
    return components, baseline, evidence


def _read_component_set(
    artifact_root: Path,
    mapping: Mapping[str, str],
    index: pd.DatetimeIndex,
) -> tuple[dict[str, dict[str, pd.Series]], dict[str, Any]]:
    frames: dict[str, pd.DataFrame] = {}
    evidence: dict[str, Any] = {}
    for component in COMPONENT_NAMES:
        path = PROJECT_DIR / mapping[component]
        evidence[component] = describe_file(path)
        frames[component] = _read_prediction(path, index, required_columns=TARGET_COLS)
    return {
        group: {component: frames[component][group].rename(group) for component in COMPONENT_NAMES}
        for group in TARGET_COLS
    }, evidence


def _dummy_index_frame(index: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(index=index)


def _read_stage1_weather(raw_dir: Path, preregister: Mapping[str, Any]) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    features, evidence = shared._read_stage1_raw_features(raw_dir, _dummy_index_frame(YEAR_2022_TO_2023))
    expected = preregister["stage1_weather_physical_prefixes"]
    for source in ("ldaps", "gfs"):
        record = evidence["raw_prefix"][source]
        if record["physical_byte_limit"] != int(expected[source]["bytes"]):
            raise AssertionError("weather prefix bytes changed")
        if record["physical_prefix_sha256"] != expected[source]["sha256"]:
            raise AssertionError("weather prefix hash changed")
        if record["suffix_bytes_exposed_to_parser"] != 0:
            raise AssertionError("weather reader crossed physical prefix")
    compact = {group: features[group].loc[:, list(WEATHER_COLUMNS)].copy() for group in TARGET_COLS}
    return compact, evidence


def _read_through_2024_weather(cache_dir: Path) -> dict[str, pd.DataFrame]:
    features = shared._read_features(cache_dir, _dummy_index_frame(YEAR_2022_TO_2024), expected_end=YEAR_2024.max())
    return {group: features[group].loc[:, list(WEATHER_COLUMNS)].copy() for group in TARGET_COLS}


def _meta(
    group: str,
    index: pd.DatetimeIndex,
    baseline: Mapping[str, pd.Series],
    components: Mapping[str, Mapping[str, pd.Series]],
    weather: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    return build_meta_features(
        baseline[group].loc[index],
        {name: components[group][name].loc[index] for name in COMPONENT_NAMES},
        weather[group].loc[index],
        capacity_kwh=CAPACITY_KWH[group],
    )


def _stage1_periods(preregister: Mapping[str, Any], group: str) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    spec = preregister["stage1"]["group_1_and_2" if group in TARGET_COLS[:2] else "group_3"]
    fit_start, fit_end = map(pd.Timestamp, spec["fit"])
    app_start, app_end = map(pd.Timestamp, spec["application"])
    if group in TARGET_COLS[:2]:
        universe = YEAR_2023
    else:
        universe = YEAR_2023_H2
    fit = universe[(universe >= fit_start) & (universe <= fit_end)]
    app = universe[(universe >= app_start) & (universe <= app_end)]
    return fit, app


def _stage1_blocks(preregister: Mapping[str, Any], group: str) -> tuple[Mapping[str, Sequence[str]], str]:
    spec = preregister["stage1"]["group_1_and_2" if group in TARGET_COLS[:2] else "group_3"]
    return spec["required_blocks"], str(spec["full_block"])


def _fit_predict_group(
    *,
    group: str,
    fit_index: pd.DatetimeIndex,
    application_index: pd.DatetimeIndex,
    fit_meta: pd.DataFrame,
    application_meta: pd.DataFrame,
    fit_baseline: pd.Series,
    application_baseline: pd.Series,
    fit_actual: pd.Series,
    minimum_rows: int,
) -> tuple[ActionUpliftPolicy, pd.Series, pd.DataFrame, dict[str, Any]]:
    policy = ActionUpliftPolicy(minimum_eligible_fit_rows=minimum_rows).fit(
        fit_meta.loc[fit_index],
        fit_baseline.loc[fit_index],
        fit_actual.loc[fit_index],
        capacity_kwh=CAPACITY_KWH[group],
    )
    candidate, diagnostics = policy.predict(
        application_meta.loc[application_index],
        application_baseline.loc[application_index],
        capacity_kwh=CAPACITY_KWH[group],
    )
    metadata = policy.metadata()
    metadata.update(
        {
            "application_rows": int(len(application_index)),
            "application_start": application_index.min(),
            "application_end": application_index.max(),
            "selected_counts": {
                str(key): int((diagnostics["selected_factor"] == key).sum())
                for key in (1.0, *ACTION_FACTORS)
            },
            "prediction_labels_used": False,
        }
    )
    return policy, candidate, diagnostics, metadata


def _stage1(
    *,
    raw_dir: Path,
    artifact_root: Path,
    out_dir: Path,
    preregister_path: Path,
    preregister: Mapping[str, Any],
    preregister_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"fresh Stage1 requires empty output directory: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(preregister_path, out_dir / "preregister_v2.json")
    shutil.copyfile(preregister_path.with_suffix(".sha256"), out_dir / "preregister_v2.sha256")
    source_lock_path = _write_source_lock(out_dir, preregister_path, preregister_evidence)

    recipe, recipe_evidence = _load_recipe(preregister)
    components, baseline, component_evidence = _load_stage1_sources(preregister, recipe)
    weather, weather_evidence = _read_stage1_weather(raw_dir, preregister)
    label_path = raw_dir / "train/train_labels.csv"
    prefixes = preregister["physical_label_prefixes"]
    labels_g12, labels_g12_evidence = _read_label_prefix(label_path, prefixes["g12_fit"], TARGET_COLS[:2])
    labels_g3, labels_g3_evidence = _read_label_prefix(label_path, prefixes["g3_fit"], ("kpx_group_3",))

    candidates: dict[str, pd.Series] = {}
    candidate_paths: dict[str, Path] = {}
    diagnostic_paths: dict[str, Path] = {}
    model_paths: dict[str, Path] = {}
    training: dict[str, Any] = {}
    for group in TARGET_COLS:
        fit_index, application_index = _stage1_periods(preregister, group)
        full_index = baseline[group].index
        meta = _meta(group, full_index, baseline, components, weather)
        labels = labels_g12 if group in TARGET_COLS[:2] else labels_g3
        policy, candidate, diagnostics, metadata = _fit_predict_group(
            group=group,
            fit_index=fit_index,
            application_index=application_index,
            fit_meta=meta,
            application_meta=meta,
            fit_baseline=baseline[group],
            application_baseline=baseline[group],
            fit_actual=labels[group],
            minimum_rows=int(preregister["models"]["minimum_eligible_fit_rows"]),
        )
        model_path = out_dir / f"models/stage1_{group}.joblib"
        candidate_path = out_dir / f"predictions/stage1_{group}_outer_candidate.parquet"
        diagnostic_path = out_dir / f"diagnostics/stage1_{group}_outer_actions.parquet"
        _atomic_joblib(policy, model_path)
        _atomic_parquet(candidate.rename(group).to_frame(), candidate_path)
        _atomic_parquet(diagnostics, diagnostic_path)
        reloaded = joblib.load(model_path)
        check, check_diagnostics = reloaded.predict(
            meta.loc[application_index], baseline[group].loc[application_index], capacity_kwh=CAPACITY_KWH[group]
        )
        if not np.array_equal(check.to_numpy(), candidate.to_numpy()):
            raise AssertionError("reloaded Stage1 candidate changed")
        if not np.array_equal(
            check_diagnostics.to_numpy(), diagnostics.to_numpy(), equal_nan=True
        ):
            raise AssertionError("reloaded Stage1 diagnostics changed")
        candidates[group] = candidate
        model_paths[group] = model_path
        candidate_paths[group] = candidate_path
        diagnostic_paths[group] = diagnostic_path
        training[group] = metadata

    _verify_source_lock(out_dir, preregister_path)
    prescore_path = out_dir / "stage1_prescore_lock.json"
    _write_json(
        prescore_path,
        {
            "preregister_sha256": PREREGISTER_SHA256,
            "source_lock": describe_file(source_lock_path),
            "recipe": recipe_evidence,
            "component_inputs": component_evidence,
            "weather_prefix_evidence": weather_evidence,
            "fit_label_evidence": {"g12": labels_g12_evidence, "g3": labels_g3_evidence},
            "models": {group: describe_file(path) for group, path in model_paths.items()},
            "candidates": {group: describe_file(path) for group, path in candidate_paths.items()},
            "diagnostics": {group: describe_file(path) for group, path in diagnostic_paths.items()},
            "training": training,
            "outer_application_label_value_cells_materialized": 0,
            "year_2024_value_bytes_read": 0,
            "year_2025_value_bytes_read": 0,
            "public_or_scale_artifact_bytes_read": 0,
        },
    )

    outer_labels, outer_evidence = _read_label_prefix(label_path, prefixes["stage1_outer_after_lock"], TARGET_COLS)
    results: dict[str, Any] = {}
    passed_groups: list[str] = []
    for group in TARGET_COLS:
        _, application_index = _stage1_periods(preregister, group)
        blocks, full_block = _stage1_blocks(preregister, group)
        comparisons = compare_group_blocks(
            outer_labels.loc[application_index, group],
            baseline[group].loc[application_index],
            candidates[group],
            group=group,
            blocks=blocks,
        )
        passed = passes_all_blocks(comparisons, full_block=full_block)
        if passed:
            passed_groups.append(group)
        results[group] = {
            "training": training[group],
            "comparisons": comparisons,
            "comparisons_sha256": _canonical_sha256(comparisons),
            "passed": passed,
        }
    result_path = out_dir / "stage1_results.json"
    _write_json(
        result_path,
        {
            "experiment_id": preregister["experiment_id"],
            "preregister_sha256": PREREGISTER_SHA256,
            "prescore_lock": describe_file(prescore_path),
            "outer_label_evidence_after_prescore_lock": outer_evidence,
            "groups": results,
            "passed_groups": passed_groups,
            "2024_read": False,
            "2025_read": False,
            "public_or_scale_artifact_bytes_read": 0,
            "leaderboard_score_claim": False,
        },
    )
    lock_path = out_dir / "stage1_promotion_lock.json"
    _write_json(
        lock_path,
        {
            "preregister_sha256": PREREGISTER_SHA256,
            "source_lock": describe_file(source_lock_path),
            "prescore_lock": describe_file(prescore_path),
            "stage1_results": describe_file(result_path),
            "passed_groups": passed_groups,
            "only_passing_groups_may_open_2024": True,
            "no_outer_reselection": True,
        },
    )
    print(f"action-uplift Stage1 passed groups: {passed_groups}", flush=True)
    return json.loads(lock_path.read_text(encoding="utf-8"))


def _load_stage1_lock(out_dir: Path, preregister_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    _verify_source_lock(out_dir, preregister_path)
    lock_path = out_dir / "stage1_promotion_lock.json"
    result_path = out_dir / "stage1_results.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if lock["preregister_sha256"] != PREREGISTER_SHA256:
        raise AssertionError("Stage1 lock preregistration changed")
    if lock["stage1_results"] != describe_file(result_path):
        raise AssertionError("Stage1 result changed after lock")
    if lock["passed_groups"] != result["passed_groups"]:
        raise AssertionError("Stage1 passed groups changed")
    return lock, result


def _macro_triplet(actual: pd.DataFrame, prediction: pd.DataFrame) -> dict[str, float]:
    metrics = score_details(actual.loc[:, list(TARGET_COLS)], prediction.loc[:, list(TARGET_COLS)])
    return {
        "score": float(metrics.total_score),
        "one_minus_nmae": float(metrics.one_minus_nmae),
        "ficr": float(metrics.ficr),
    }


def _macro_compare(
    actual: pd.DataFrame, baseline: pd.DataFrame, candidate: pd.DataFrame
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for name, bounds in STAGE2_BLOCKS.items():
        start, end = map(pd.Timestamp, bounds)
        index = actual.index[(actual.index >= start) & (actual.index <= end)]
        base = _macro_triplet(actual.loc[index], baseline.loc[index])
        cand = _macro_triplet(actual.loc[index], candidate.loc[index])
        output[name] = {
            "rows": int(len(index)),
            "baseline": base,
            "candidate": cand,
            "delta": {metric: float(cand[metric] - base[metric]) for metric in base},
        }
    return output


def _passes_macro(comparisons: Mapping[str, Any]) -> bool:
    if tuple(comparisons) != tuple(STAGE2_BLOCKS):
        raise AssertionError("macro block order changed")
    if any(float(record["delta"]["score"]) <= 0.0 for record in comparisons.values()):
        return False
    full = comparisons["full"]["delta"]
    nmae = float(full["one_minus_nmae"])
    ficr = float(full["ficr"])
    return nmae >= 0.0 and ficr >= 0.0 and (nmae > 0.0 or ficr > 0.0)


def _stage2_no_candidate(out_dir: Path, stage1_lock: Mapping[str, Any]) -> dict[str, Any]:
    result_path = out_dir / "stage2_results.json"
    _write_json(
        result_path,
        {
            "performed": False,
            "reason": "all Stage1 groups locked identity",
            "stage1_promotion_lock": describe_file(out_dir / "stage1_promotion_lock.json"),
            "stage1_passed_groups": list(stage1_lock["passed_groups"]),
            "2024_component_read": False,
            "2024_weather_read": False,
            "2024_label_read": False,
            "recent_v4_read": False,
            "2025_read": False,
            "public_or_scale_artifact_bytes_read": 0,
            "promoted": False,
        },
    )
    lock_path = out_dir / "stage2_promotion_lock.json"
    _write_json(
        lock_path,
        {
            "preregister_sha256": PREREGISTER_SHA256,
            "stage2_results": describe_file(result_path),
            "promoted": False,
            "promoted_groups": [],
        },
    )
    return json.loads(lock_path.read_text(encoding="utf-8"))


def _stage2(
    *,
    raw_dir: Path,
    cache_dir: Path,
    artifact_root: Path,
    out_dir: Path,
    preregister_path: Path,
    preregister: Mapping[str, Any],
) -> dict[str, Any]:
    stage1_lock, _ = _load_stage1_lock(out_dir, preregister_path)
    passed_groups = list(stage1_lock["passed_groups"])
    if not passed_groups:
        return _stage2_no_candidate(out_dir, stage1_lock)

    recipe, recipe_evidence = _load_recipe(preregister)
    stage1_components, stage1_baseline, stage1_component_evidence = _load_stage1_sources(preregister, recipe)
    fit_labels, fit_label_evidence = _read_label_prefix(
        raw_dir / "train/train_labels.csv",
        preregister["physical_label_prefixes"]["stage1_outer_after_lock"],
        TARGET_COLS,
    )
    weather = _read_through_2024_weather(cache_dir)
    stage2_components, stage2_component_evidence = _read_component_set(
        artifact_root,
        preregister["conditional_input_paths"]["stage2_components"],
        YEAR_2024,
    )
    v3_path = PROJECT_DIR / preregister["stage2"]["primary_baseline_path"]
    recent_path = PROJECT_DIR / preregister["stage2"]["interaction_baseline_path"]
    v3 = _read_prediction(v3_path, YEAR_2024, required_columns=TARGET_COLS)
    recent = _read_prediction(recent_path, YEAR_2024, required_columns=TARGET_COLS)
    direct = v3.copy()
    interaction = recent.copy()
    model_paths: dict[str, Path] = {}
    diagnostic_paths: dict[str, Path] = {}
    training: dict[str, Any] = {}
    for group in passed_groups:
        bounds = preregister["stage2"]["refit_history"][group]
        universe = YEAR_2023 if group in TARGET_COLS[:2] else YEAR_2023_H2
        start, end = map(pd.Timestamp, bounds)
        fit_index = universe[(universe >= start) & (universe <= end)]
        fit_meta = _meta(group, fit_index, stage1_baseline, stage1_components, weather)
        apply_baseline = {group: v3[group]}
        apply_components = {group: stage2_components[group]}
        apply_weather = {group: weather[group]}
        apply_meta = build_meta_features(
            v3[group], stage2_components[group], weather[group].loc[YEAR_2024], capacity_kwh=CAPACITY_KWH[group]
        )
        policy, candidate, diagnostics, metadata = _fit_predict_group(
            group=group,
            fit_index=fit_index,
            application_index=YEAR_2024,
            fit_meta=fit_meta,
            application_meta=apply_meta,
            fit_baseline=stage1_baseline[group],
            application_baseline=v3[group],
            fit_actual=fit_labels[group],
            minimum_rows=int(preregister["models"]["minimum_eligible_fit_rows"]),
        )
        direct.loc[:, group] = candidate.to_numpy()
        interaction.loc[:, group] = transfer_delta(
            recent[group], v3[group], candidate, capacity_kwh=CAPACITY_KWH[group]
        ).to_numpy()
        model_path = out_dir / f"models/stage2_{group}.joblib"
        diagnostic_path = out_dir / f"diagnostics/stage2_{group}_actions.parquet"
        _atomic_joblib(policy, model_path)
        _atomic_parquet(diagnostics, diagnostic_path)
        model_paths[group] = model_path
        diagnostic_paths[group] = diagnostic_path
        training[group] = metadata
    direct_path = out_dir / "predictions/stage2_fixed_v3_candidate.parquet"
    interaction_path = out_dir / "predictions/stage2_recent_v4_interaction.parquet"
    _atomic_parquet(direct, direct_path)
    _atomic_parquet(interaction, interaction_path)
    prescore_path = out_dir / "stage2_prescore_lock.json"
    _write_json(
        prescore_path,
        {
            "preregister_sha256": PREREGISTER_SHA256,
            "stage1_promotion_lock": describe_file(out_dir / "stage1_promotion_lock.json"),
            "recipe": recipe_evidence,
            "stage1_component_inputs": stage1_component_evidence,
            "fit_label_prefix": fit_label_evidence,
            "stage2_component_inputs": stage2_component_evidence,
            "primary_v3": describe_file(v3_path),
            "interaction_recent_v4": describe_file(recent_path),
            "models": {group: describe_file(path) for group, path in model_paths.items()},
            "diagnostics": {group: describe_file(path) for group, path in diagnostic_paths.items()},
            "direct_candidate": describe_file(direct_path),
            "interaction_candidate": describe_file(interaction_path),
            "training": training,
            "application_2024_label_value_cells_materialized": 0,
            "public_or_scale_artifact_bytes_read": 0,
        },
    )
    labels, label_evidence = _read_full_labels(raw_dir, preregister)
    actual = labels.loc[YEAR_2024]
    group_results: dict[str, Any] = {}
    all_group_dual = True
    for group in passed_groups:
        direct_comparisons = compare_group_blocks(
            actual[group], v3[group], direct[group], group=group, blocks=STAGE2_BLOCKS
        )
        interaction_comparisons = compare_group_blocks(
            actual[group], recent[group], interaction[group], group=group, blocks=STAGE2_BLOCKS
        )
        direct_passed = passes_all_blocks(direct_comparisons, full_block="full")
        interaction_passed = passes_all_blocks(interaction_comparisons, full_block="full")
        dual = direct_passed and interaction_passed
        all_group_dual = all_group_dual and dual
        group_results[group] = {
            "direct_v3": direct_comparisons,
            "recent_v4_interaction": interaction_comparisons,
            "direct_passed": direct_passed,
            "interaction_passed": interaction_passed,
            "dual_passed": dual,
        }
    mixed_direct = _macro_compare(actual, v3, direct)
    mixed_interaction = _macro_compare(actual, recent, interaction)
    mixed_direct_passed = _passes_macro(mixed_direct)
    mixed_interaction_passed = _passes_macro(mixed_interaction)
    promoted = bool(all_group_dual and mixed_direct_passed and mixed_interaction_passed)
    result_path = out_dir / "stage2_results.json"
    _write_json(
        result_path,
        {
            "performed": True,
            "prescore_lock": describe_file(prescore_path),
            "score_labels_read_after_prescore_lock": label_evidence,
            "stage1_passed_groups": passed_groups,
            "groups": group_results,
            "mixed_direct_v3": {"comparisons": mixed_direct, "passed": mixed_direct_passed},
            "mixed_recent_v4_interaction": {
                "comparisons": mixed_interaction,
                "passed": mixed_interaction_passed,
                "calibration_consumed_not_independent": True,
            },
            "all_locked_groups_dual_passed": all_group_dual,
            "no_partial_group_drop": True,
            "promoted": promoted,
            "promoted_groups": passed_groups if promoted else [],
            "2025_read": False,
            "public_or_scale_artifact_bytes_read": 0,
        },
    )
    lock_path = out_dir / "stage2_promotion_lock.json"
    _write_json(
        lock_path,
        {
            "preregister_sha256": PREREGISTER_SHA256,
            "stage1_promotion_lock": describe_file(out_dir / "stage1_promotion_lock.json"),
            "stage2_results": describe_file(result_path),
            "promoted": promoted,
            "promoted_groups": passed_groups if promoted else [],
        },
    )
    print(f"action-uplift Stage2 promoted={promoted}", flush=True)
    return json.loads(lock_path.read_text(encoding="utf-8"))


def _load_stage2_lock(out_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    lock_path = out_dir / "stage2_promotion_lock.json"
    result_path = out_dir / "stage2_results.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if lock["preregister_sha256"] != PREREGISTER_SHA256:
        raise AssertionError("Stage2 lock preregistration changed")
    if lock["stage2_results"] != describe_file(result_path):
        raise AssertionError("Stage2 result changed after lock")
    return lock, result


def _final_no_candidate(out_dir: Path, stage2_lock: Mapping[str, Any]) -> dict[str, Any]:
    path = out_dir / "final_results.json"
    _write_json(
        path,
        {
            "executed": False,
            "reason": "complete fixed 2024 dual-baseline gate did not pass",
            "stage2_promotion_lock": describe_file(out_dir / "stage2_promotion_lock.json"),
            "promoted_groups": list(stage2_lock["promoted_groups"]),
            "2025_components_read": False,
            "2025_weather_read": False,
            "2025_v3_read": False,
            "2025_recent_v4_read": False,
            "sample_read": False,
            "csv_created": False,
            "public_or_scale_artifact_bytes_read": 0,
        },
    )
    return json.loads(path.read_text(encoding="utf-8"))


def _final(
    *,
    raw_dir: Path,
    cache_dir: Path,
    artifact_root: Path,
    out_dir: Path,
    preregister: Mapping[str, Any],
) -> dict[str, Any]:
    stage2_lock, _ = _load_stage2_lock(out_dir)
    if not stage2_lock["promoted"]:
        return _final_no_candidate(out_dir, stage2_lock)
    promoted_groups = list(stage2_lock["promoted_groups"])
    labels, label_evidence = _read_full_labels(raw_dir, preregister)
    weather_train = _read_through_2024_weather(cache_dir)
    stage2_components, stage2_component_evidence = _read_component_set(
        artifact_root, preregister["conditional_input_paths"]["stage2_components"], YEAR_2024
    )
    v3_2024 = _read_prediction(
        PROJECT_DIR / preregister["stage2"]["primary_baseline_path"], YEAR_2024, required_columns=TARGET_COLS
    )
    test_weather_full = shared._read_test_features(cache_dir, YEAR_2025)
    test_weather = {
        group: test_weather_full[group].loc[:, list(WEATHER_COLUMNS)].copy() for group in TARGET_COLS
    }
    final_components, final_component_evidence = _read_component_set(
        artifact_root, preregister["conditional_input_paths"]["final_components"], YEAR_2025
    )
    v3_path = PROJECT_DIR / preregister["final"]["corrected_v3_path"]
    recent_path = PROJECT_DIR / preregister["final"]["corrected_recent_v4_path"]
    v3 = _read_prediction(v3_path, YEAR_2025, required_columns=TARGET_COLS)
    recent = _read_prediction(recent_path, YEAR_2025, required_columns=TARGET_COLS)
    final = recent.copy()
    models: dict[str, Any] = {}
    diagnostics: dict[str, Any] = {}
    for group in promoted_groups:
        fit_meta = build_meta_features(
            v3_2024[group], stage2_components[group], weather_train[group].loc[YEAR_2024], capacity_kwh=CAPACITY_KWH[group]
        )
        apply_meta = build_meta_features(
            v3[group], final_components[group], test_weather[group], capacity_kwh=CAPACITY_KWH[group]
        )
        policy, candidate, diagnostic, metadata = _fit_predict_group(
            group=group,
            fit_index=YEAR_2024,
            application_index=YEAR_2025,
            fit_meta=fit_meta,
            application_meta=apply_meta,
            fit_baseline=v3_2024[group],
            application_baseline=v3[group],
            fit_actual=labels.loc[YEAR_2024, group],
            minimum_rows=int(preregister["models"]["minimum_eligible_fit_rows"]),
        )
        final.loc[:, group] = transfer_delta(
            recent[group], v3[group], candidate, capacity_kwh=CAPACITY_KWH[group]
        ).to_numpy()
        model_path = out_dir / f"models/final_{group}.joblib"
        diagnostic_path = out_dir / f"diagnostics/final_{group}_actions.parquet"
        _atomic_joblib(policy, model_path)
        _atomic_parquet(diagnostic, diagnostic_path)
        models[group] = {"file": describe_file(model_path), "metadata": metadata}
        diagnostics[group] = describe_file(diagnostic_path)
    for group in TARGET_COLS:
        if group not in promoted_groups and not np.array_equal(final[group].to_numpy(), recent[group].to_numpy()):
            raise AssertionError("final identity group changed")
    prediction_path = out_dir / "final/action_uplift_policy_recent_v4_2025.parquet"
    _atomic_parquet(final, prediction_path)
    sample_path = Path(preregister["final"]["sample_path"])
    sample = pd.read_csv(sample_path, encoding="utf-8-sig")
    if len(sample) != len(final) or tuple(sample.columns[-3:]) != TARGET_COLS:
        raise AssertionError("sample submission schema changed")
    submission = sample.copy()
    submission.loc[:, list(TARGET_COLS)] = final.to_numpy(dtype=np.float64)
    csv_path = out_dir / "final/action_uplift_policy_recent_v4_2025.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = csv_path.with_name(f".{csv_path.name}.tmp-{os.getpid()}")
    submission.to_csv(temporary, index=False, encoding="utf-8-sig", float_format="%.6f", lineterminator="\n")
    os.replace(temporary, csv_path)
    result_path = out_dir / "final_results.json"
    _write_json(
        result_path,
        {
            "executed": True,
            "promoted_groups": promoted_groups,
            "label_input": label_evidence,
            "stage2_component_inputs": stage2_component_evidence,
            "final_component_inputs": final_component_evidence,
            "v3_baseline": describe_file(v3_path),
            "recent_v4_baseline": describe_file(recent_path),
            "models": models,
            "diagnostics": diagnostics,
            "prediction": describe_file(prediction_path),
            "sample": describe_file(sample_path),
            "csv": describe_file(csv_path),
            "csv_created": True,
            "leaderboard_score_claim": False,
            "public_or_scale_artifact_bytes_read": 0,
        },
    )
    return json.loads(result_path.read_text(encoding="utf-8"))


def _write_manifest(out_dir: Path, preregister_path: Path) -> Path:
    outputs = {
        str(path.relative_to(out_dir)).replace("\\", "/"): describe_file(path)
        for path in sorted(out_dir.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }
    stage1 = json.loads((out_dir / "stage1_results.json").read_text(encoding="utf-8"))
    stage2 = json.loads((out_dir / "stage2_results.json").read_text(encoding="utf-8"))
    final = json.loads((out_dir / "final_results.json").read_text(encoding="utf-8"))
    path = out_dir / "manifest.json"
    _write_json(
        path,
        {
            "schema_version": 1,
            "experiment_id": "action_uplift_policy_strict_v2",
            "created_utc": utc_now(),
            "preregister": describe_file(preregister_path),
            "source_lock": describe_file(out_dir / "source_lock_before_fit.json"),
            "stage1_passed_groups": stage1["passed_groups"],
            "stage2_performed": stage2["performed"],
            "stage2_promoted": stage2["promoted"],
            "recent_v4_interaction_gate_executed": bool(stage2["performed"]),
            "final_executed": final["executed"],
            "csv_created": bool(final["csv_created"]),
            "public_or_scale_artifact_bytes_read": 0,
            "leaderboard_score_claim": False,
            "outputs": outputs,
            "git": git_state(PROJECT_DIR),
            "packages": package_versions(("numpy", "pandas", "pyarrow", "lightgbm", "scikit-learn")),
        },
    )
    return path


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    args.raw_dir = args.raw_dir.expanduser().resolve()
    args.cache_dir = (PROJECT_DIR / args.cache_dir).resolve() if not args.cache_dir.is_absolute() else args.cache_dir.resolve()
    args.artifact_root = (
        (PROJECT_DIR / args.artifact_root).resolve() if not args.artifact_root.is_absolute() else args.artifact_root.resolve()
    )
    args.out_dir = (PROJECT_DIR / args.out_dir).resolve() if not args.out_dir.is_absolute() else args.out_dir.resolve()
    args.preregister = (
        (PROJECT_DIR / args.preregister).resolve() if not args.preregister.is_absolute() else args.preregister.resolve()
    )
    preregister, preregister_evidence = _load_preregister(args.preregister)
    if args.stage in ("stage1", "all"):
        _stage1(
            raw_dir=args.raw_dir,
            artifact_root=args.artifact_root,
            out_dir=args.out_dir,
            preregister_path=args.preregister,
            preregister=preregister,
            preregister_evidence=preregister_evidence,
        )
    if args.stage in ("stage2", "all"):
        _stage2(
            raw_dir=args.raw_dir,
            cache_dir=args.cache_dir,
            artifact_root=args.artifact_root,
            out_dir=args.out_dir,
            preregister_path=args.preregister,
            preregister=preregister,
        )
    if args.stage in ("final", "all"):
        _final(
            raw_dir=args.raw_dir,
            cache_dir=args.cache_dir,
            artifact_root=args.artifact_root,
            out_dir=args.out_dir,
            preregister=preregister,
        )
    if args.stage == "all":
        manifest = _write_manifest(args.out_dir, args.preregister)
        print(f"manifest_sha256={sha256_file(manifest)}", flush=True)


if __name__ == "__main__":
    main()
