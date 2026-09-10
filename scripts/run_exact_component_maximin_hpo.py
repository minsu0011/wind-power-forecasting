"""Strict exact-score maximin HPO over corrected-v3 component weights."""

from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from scripts import run_shared_q07_multiseed as shared  # noqa: E402
from src.exact_component_maximin import (  # noqa: E402
    COMPONENTS,
    N_TRIALS,
    assemble_candidate,
    compare_blocks,
    passes_block_gate,
    run_exact_search,
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


PREREGISTER_SHA256 = "dee8d03fb0d150e0732a8a92eb1e7e027c6f1fdf1ea16e4f534a0493762eb635"
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
DEV_COMPONENT_FILES = {
    "lgb_l1": "oof/dev2023_lgb_l1_eligible_n1500.parquet",
    "lgb_q07": "oof/dev2023_lgb_q07_eligible.parquet",
    "shared_l1": "oof/dev2023_shared_l1_eligible.parquet",
    "shared_q07": "oof/dev2023_shared_q07_eligible.parquet",
    "top200_q07": "oof/dev2023_lgb_top200_q07_eligible.parquet",
    "energy_q06": "oof/dev2023_lgb_q06_energywt_eligible.parquet",
}
G3_COLUMN_MAP = {
    "lgb_l1": "l1",
    "lgb_q07": "q07",
    "shared_l1": "shared_l1",
    "shared_q07": "shared_q07",
    "top200_q07": "top200q07",
    "energy_q06": "ewq06",
}
GATE_COMPONENT_FILES = {
    "lgb_l1": "gate/v3/predictions/lgb_l1_gate.parquet",
    "lgb_q07": "gate/v3/predictions/lgb_q07_gate.parquet",
    "shared_l1": "oof/gate2024_shared_l1_cf.parquet",
    "shared_q07": "oof/gate2024_shared_q07_cf.parquet",
    "top200_q07": "gate/v3/predictions/top200_q07_gate.parquet",
    "energy_q06": "gate/v3/predictions/energy_q06_gate.parquet",
}
FINAL_COMPONENT_FILES = {
    "lgb_l1": "final_v3/predictions/v3_locked_full_2025__lgb_l1_test.parquet",
    "lgb_q07": "final_v3/predictions/v3_locked_full_2025__lgb_q07_test.parquet",
    "shared_l1": "final_cf_fix/predictions/shared_l1_cf_seed42_test.parquet",
    "shared_q07": "final_cf_fix/predictions/shared_q07_cf_seed42_test.parquet",
    "top200_q07": "final_v3/predictions/v3_locked_full_2025__top200_q07_test.parquet",
    "energy_q06": "final_v3/predictions/v3_locked_full_2025__energy_q06_test.parquet",
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
FULL_LABEL_IDENTITY = {
    "bytes": 1138967,
    "sha256": "47bb64252195cf4734e67394d6e50485f27a608def3b5a8791fcc7674bbceb03",
    "rows": 26304,
}
SAMPLE_IDENTITY = {
    "bytes": 359229,
    "sha256": "c925d2066a834f937f8091ed55acfe50ff86c8be4745b52c3adc95b056c5aaaa",
}
EXPECTED_SOURCE_CLOSURE: tuple[str, ...] = (
    "scripts/run_exact_component_maximin_hpo.py",
    "scripts/run_shared_q07_multiseed.py",
    "src/exact_component_maximin.py",
    "src/features.py",
    "src/manifest.py",
    "src/metric.py",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("stage1", "stage2", "final", "all"), required=True)
    parser.add_argument(
        "--raw-dir", type=Path, default=Path(r"data/local/open")
    )
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/postgate/exact_component_maximin_hpo_strict_v1"),
    )
    parser.add_argument(
        "--preregister",
        type=Path,
        default=Path("configs/exact_component_maximin_hpo_preregister_v1.json"),
    )
    return parser.parse_args(argv)


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    write_json_atomic(path, _json_ready(payload), overwrite=False)


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        frame.to_parquet(temporary, engine="pyarrow", compression="zstd", index=True)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        frame.to_csv(
            temporary,
            index=False,
            encoding="utf-8-sig",
            float_format="%.6f",
            lineterminator="\n",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        _json_ready(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _verify_preregister(path: Path) -> dict[str, Any]:
    if sha256_file(path) != PREREGISTER_SHA256:
        raise AssertionError("exact-component preregistration changed")
    expected = f"{PREREGISTER_SHA256}  {path.name}\n"
    if path.with_suffix(".sha256").read_text(encoding="utf-8") != expected:
        raise AssertionError("exact-component preregistration sidecar changed")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload["experiment_id"] != "exact_component_maximin_hpo_strict_v1":
        raise AssertionError("experiment id changed")
    if int(payload["optuna"]["n_trials_per_group"]) != N_TRIALS:
        raise AssertionError("trial count changed")
    if tuple(payload["components"]["ordered_names"]) != COMPONENTS:
        raise AssertionError("component order changed")
    if payload["census_and_distinctness"]["go_decision"] != "GO_distinct_objective_and_parameterization":
        raise AssertionError("census decision changed")
    return payload


def _assert_identity(path: Path, spec: Mapping[str, Any]) -> dict[str, Any]:
    record = describe_file(path)
    if record["size_bytes"] != int(spec["bytes"]) or record["sha256"] != spec["sha256"]:
        raise AssertionError(f"input identity changed: {path}")
    return record


def _read_label_prefix(
    path: Path, spec: Mapping[str, Any], groups: Sequence[str]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = int(spec["data_rows"])
    byte_limit = int(spec["bytes"])
    digest, observed_bytes = shared._csv_prefix_identity(
        path, data_rows=rows, byte_limit=byte_limit
    )
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
        raise AssertionError("label parser crossed physical prefix")
    if tuple(frame.columns) != ("kst_dtm", *groups) or len(frame) != rows:
        raise AssertionError("label prefix schema changed")
    index = pd.to_datetime(frame.pop("kst_dtm"), errors="raise")
    frame.index = pd.DatetimeIndex(index, name="forecast_kst_dtm")
    frame = frame.astype(np.float64)
    if frame.index.max() != pd.Timestamp(spec["end"]):
        raise AssertionError("label prefix end changed")
    next_timestamp = shared._next_csv_first_field(
        path, after_data_rows=rows, prefix_bytes=byte_limit
    )
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


def _read_prediction(
    path: Path,
    index: pd.DatetimeIndex,
    *,
    required_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    frame = pd.read_parquet(path, engine="pyarrow")
    frame.index = pd.DatetimeIndex(frame.index, name="forecast_kst_dtm")
    if not frame.index.equals(index):
        raise AssertionError(f"prediction index changed: {path}")
    if required_columns is not None:
        missing = [column for column in required_columns if column not in frame]
        if missing:
            raise AssertionError(f"prediction columns changed: {path}: {missing}")
        frame = frame.loc[:, list(required_columns)]
    if not np.isfinite(frame.to_numpy(dtype=np.float64)).all():
        raise AssertionError(f"prediction contains non-finite values: {path}")
    return frame.astype(np.float64)


def _load_recipe(path: Path, preregister: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    spec = preregister["locked_v3_assembly"]
    identity = _assert_identity(
        path,
        {"bytes": spec["recipe_bytes"], "sha256": spec["recipe_sha256"]},
    )
    recipe = json.loads(path.read_text(encoding="utf-8"))
    if tuple(recipe["ensemble"]["weights"]) != TARGET_COLS:
        raise AssertionError("recipe group order changed")
    for group in TARGET_COLS:
        if tuple(recipe["ensemble"]["weights"][group]) != COMPONENTS:
            raise AssertionError("recipe component order changed")
    return recipe, identity


def _load_stage1_sources(
    artifact_root: Path,
    preregister: Mapping[str, Any],
    recipe: Mapping[str, Any],
) -> tuple[dict[str, dict[str, pd.Series]], dict[str, pd.Series], dict[str, Any]]:
    identities = preregister["stage1_component_identities"]
    source_evidence: dict[str, Any] = {}
    dev_frames: dict[str, pd.DataFrame] = {}
    for component in COMPONENTS:
        spec = identities[component]
        path = PROJECT_DIR / spec["path"]
        source_evidence[component] = _assert_identity(path, spec)
        dev_frames[component] = _read_prediction(
            path, YEAR_2023, required_columns=TARGET_COLS[:2]
        )
    g3_spec = identities["g3_components"]
    g3_path = PROJECT_DIR / g3_spec["path"]
    source_evidence["g3_components"] = _assert_identity(g3_path, g3_spec)
    g3_raw = _read_prediction(
        g3_path, YEAR_2023_H2, required_columns=tuple(G3_COLUMN_MAP.values())
    )
    components: dict[str, dict[str, pd.Series]] = {
        group: {
            component: dev_frames[component][group].rename(group)
            for component in COMPONENTS
        }
        for group in TARGET_COLS[:2]
    }
    components["kpx_group_3"] = {
        component: g3_raw[column].rename("kpx_group_3")
        for component, column in G3_COLUMN_MAP.items()
    }
    base_spec = identities["g12_corrected_v3"]
    base_path = PROJECT_DIR / base_spec["path"]
    source_evidence["g12_corrected_v3"] = _assert_identity(base_path, base_spec)
    g12 = _read_prediction(base_path, YEAR_2023, required_columns=TARGET_COLS[:2])
    group = "kpx_group_3"
    weights = recipe["ensemble"]["weights"][group]
    weighted = sum(
        float(weights[component]) * components[group][component].to_numpy(dtype=np.float64)
        for component in COMPONENTS
    )
    affine = recipe["ensemble"]["affine"][group]
    g3_values = np.clip(
        float(affine["scale"]) * weighted + float(affine["bias_kwh"]),
        0.0,
        1.02 * CAPACITY_KWH[group],
    )
    baseline = {
        "kpx_group_1": g12["kpx_group_1"].rename("kpx_group_1"),
        "kpx_group_2": g12["kpx_group_2"].rename("kpx_group_2"),
        "kpx_group_3": pd.Series(g3_values, index=YEAR_2023_H2, name=group),
    }
    formula_audit: dict[str, Any] = {}
    for group in TARGET_COLS[:2]:
        parameter = {
            "donor": "lgb_l1",
            "recipient_offset": 1,
            "mass_shift": 0.0,
            "scale_delta": 0.0,
            "bias_delta_cf": 0.0,
        }
        identity_prediction, metadata = assemble_candidate(
            components[group],
            group=group,
            recipe=recipe,
            parameter=parameter,
            identity_baseline_kwh=baseline[group],
        )
        if not np.array_equal(identity_prediction.to_numpy(), baseline[group].to_numpy()):
            raise AssertionError("identity short-circuit is not bit exact")
        formula_audit[group] = metadata
    return components, baseline, {
        "source_identities": source_evidence,
        "identity_bit_exact": True,
        "identity_formula_audit": formula_audit,
    }


def _blocks(preregister: Mapping[str, Any], group: str, phase: str) -> Mapping[str, Sequence[str]]:
    if phase == "inner":
        key = "group_1_and_2_blocks" if group in TARGET_COLS[:2] else "group_3_blocks"
        return preregister["inner_objective"][key]
    if phase == "outer":
        key = "group_1_and_2_required_blocks" if group in TARGET_COLS[:2] else "group_3_required_blocks"
        return preregister["stage1_outer"][key]
    raise ValueError(phase)


def _inner_index(group: str) -> pd.DatetimeIndex:
    return YEAR_2023_H1 if group in TARGET_COLS[:2] else YEAR_2023_Q3


def _outer_index(group: str) -> pd.DatetimeIndex:
    return YEAR_2023_H2 if group in TARGET_COLS[:2] else YEAR_2023_Q4


def _full_block(group: str, phase: str) -> str:
    if phase == "inner":
        return "H1" if group in TARGET_COLS[:2] else "Q3"
    return "H2" if group in TARGET_COLS[:2] else "Q4"


def _resolve_local_module(module: str) -> Path | None:
    if not module or module.split(".", 1)[0] not in {"scripts", "src"}:
        return None
    candidate = PROJECT_DIR.joinpath(*module.split(".")).with_suffix(".py")
    return candidate.resolve() if candidate.is_file() else None


def _direct_local_imports(path: Path) -> set[Path]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: set[Path] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                resolved = _resolve_local_module(alias.name)
                if resolved is not None:
                    imports.add(resolved)
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            module = node.module or ""
            resolved = _resolve_local_module(module)
            if resolved is not None:
                imports.add(resolved)
            if module in {"scripts", "src"}:
                for alias in node.names:
                    resolved = _resolve_local_module(f"{module}.{alias.name}")
                    if resolved is not None:
                        imports.add(resolved)
    return imports


def _source_closure() -> tuple[Path, ...]:
    pending = [Path(__file__).resolve()]
    observed: set[Path] = set()
    while pending:
        path = pending.pop()
        if path in observed:
            continue
        observed.add(path)
        pending.extend(dependency for dependency in _direct_local_imports(path) if dependency not in observed)
    output = tuple(
        sorted(observed, key=lambda path: path.relative_to(PROJECT_DIR).as_posix())
    )
    relative = tuple(path.relative_to(PROJECT_DIR).as_posix() for path in output)
    if relative != EXPECTED_SOURCE_CLOSURE:
        raise AssertionError(f"static source closure changed: {relative}")
    return output


def _provenance_paths(preregister_path: Path) -> dict[str, Path]:
    paths = {
        f"repo_source::{path.relative_to(PROJECT_DIR).as_posix()}": path
        for path in _source_closure()
    }
    paths.update({
        "test": PROJECT_DIR / "tests/test_exact_component_maximin.py",
        "preregister": preregister_path.resolve(),
        "preregister_sidecar": preregister_path.with_suffix(".sha256").resolve(),
        "recipe": PROJECT_DIR / "configs/train_final.v3.locked.json",
    })
    forbidden = ("public", "scale")
    if any(token in path.as_posix().lower() for path in paths.values() for token in forbidden):
        raise AssertionError("forbidden provenance path")
    return paths


def _snapshot(paths: Mapping[str, Path]) -> dict[str, Any]:
    return {name: describe_file(path) for name, path in paths.items()}


def _stage1(
    *,
    raw_dir: Path,
    artifact_root: Path,
    out_dir: Path,
    preregister_path: Path,
    preregister: Mapping[str, Any],
) -> dict[str, Any]:
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    provenance_before = _snapshot(_provenance_paths(preregister_path))
    recipe, recipe_identity = _load_recipe(
        PROJECT_DIR / "configs/train_final.v3.locked.json", preregister
    )
    components, baseline, source_evidence = _load_stage1_sources(
        artifact_root, preregister, recipe
    )
    label_path = raw_dir / "train/train_labels.csv"
    prefixes = preregister["physical_label_prefixes"]
    labels_g12, g12_evidence = _read_label_prefix(
        label_path, prefixes["g12_inner"], TARGET_COLS[:2]
    )
    labels_g3, g3_evidence = _read_label_prefix(
        label_path, prefixes["g3_inner"], ("kpx_group_3",)
    )
    best_by_group: dict[str, Any] = {}
    history_paths: list[Path] = []
    inner_passed: list[str] = []
    for group in TARGET_COLS:
        index = _inner_index(group)
        labels = labels_g12 if group in TARGET_COLS[:2] else labels_g3
        print(f"exact component maximin {group}: {N_TRIALS} trials", flush=True)
        best, history = run_exact_search(
            group=group,
            components_kwh={name: components[group][name].loc[index] for name in COMPONENTS},
            actual_kwh=labels.loc[index, group],
            baseline_kwh=baseline[group].loc[index],
            recipe=recipe,
            blocks=_blocks(preregister, group, "inner"),
            full_block=_full_block(group, "inner"),
        )
        best_by_group[group] = best
        if best["inner_passed"]:
            inner_passed.append(group)
        path = out_dir / f"hpo/{group}_study.json"
        _write_json(path, {"group": group, "best": best, "trials": history})
        history_paths.append(path)

    inner_lock_path = out_dir / "inner_hpo_lock.json"
    _write_json(
        inner_lock_path,
        {
            "preregister_sha256": PREREGISTER_SHA256,
            "recipe": recipe_identity,
            "best_by_group": best_by_group,
            "inner_passed_groups": inner_passed,
            "study_outputs": [describe_file(path) for path in history_paths],
            "outer_application_target_values_materialized": 0,
            "2024_read": False,
            "2025_read": False,
        },
    )
    candidate_paths: dict[str, Path] = {}
    candidate_outer: dict[str, pd.Series] = {}
    assembly_outer: dict[str, Any] = {}
    for group in inner_passed:
        index = _outer_index(group)
        candidate, assembly = assemble_candidate(
            {name: components[group][name].loc[index] for name in COMPONENTS},
            group=group,
            recipe=recipe,
            parameter=best_by_group[group]["locked_parameter"],
            identity_baseline_kwh=baseline[group].loc[index],
        )
        candidate_outer[group] = candidate
        assembly_outer[group] = assembly
        path = out_dir / f"predictions/stage1_{group}_fixed_outer_candidate.parquet"
        _atomic_parquet(candidate.rename(group).to_frame(), path)
        candidate_paths[group] = path
    provenance_after = _snapshot(_provenance_paths(preregister_path))
    if provenance_before != provenance_after:
        raise AssertionError("source provenance changed during Stage1")
    prescore_path = out_dir / "stage1_prescore_lock.json"
    _write_json(
        prescore_path,
        {
            "preregister_sha256": PREREGISTER_SHA256,
            "inner_hpo_lock": describe_file(inner_lock_path),
            "recipe": recipe_identity,
            "source_evidence": source_evidence,
            "inner_label_evidence": {"g12": g12_evidence, "g3": g3_evidence},
            "inner_passed_groups": inner_passed,
            "locked_parameters": {
                group: best_by_group[group]["locked_parameter"] for group in inner_passed
            },
            "outer_assembly": assembly_outer,
            "outer_candidate_outputs": {
                group: describe_file(path) for group, path in candidate_paths.items()
            },
            "provenance_before": provenance_before,
            "provenance_after": provenance_after,
            "outer_application_target_values_materialized": 0,
            "2024_read": False,
            "2025_read": False,
        },
    )

    group_results: dict[str, Any] = {
        group: {"inner": best_by_group[group], "outer_performed": False, "outer_passed": False}
        for group in TARGET_COLS
    }
    outer_label_evidence: dict[str, Any] | None = None
    passed_groups: list[str] = []
    if inner_passed:
        outer_labels, outer_label_evidence = _read_label_prefix(
            label_path, prefixes["stage1_outer_after_lock"], TARGET_COLS
        )
        for group in inner_passed:
            index = _outer_index(group)
            comparisons = compare_blocks(
                outer_labels.loc[index, group],
                baseline[group].loc[index],
                candidate_outer[group],
                group=group,
                blocks=_blocks(preregister, group, "outer"),
            )
            passed = passes_block_gate(
                comparisons, full_block=_full_block(group, "outer")
            )
            if passed:
                passed_groups.append(group)
            group_results[group].update(
                {
                    "outer_performed": True,
                    "outer_comparisons": comparisons,
                    "outer_comparisons_sha256": _canonical_sha256(comparisons),
                    "outer_passed": passed,
                }
            )
    result_path = out_dir / "stage1_results.json"
    result = {
        "experiment_id": preregister["experiment_id"],
        "preregister_sha256": PREREGISTER_SHA256,
        "prescore_lock": describe_file(prescore_path),
        "outer_label_evidence_after_prescore_lock": outer_label_evidence,
        "group_results": group_results,
        "inner_passed_groups": inner_passed,
        "passed_groups": passed_groups,
        "2024_read": False,
        "2025_read": False,
        "leaderboard_score_claim": False,
    }
    _write_json(result_path, result)
    lock = {
        "preregister_sha256": PREREGISTER_SHA256,
        "prescore_lock": describe_file(prescore_path),
        "stage1_results": describe_file(result_path),
        "passed_groups": passed_groups,
        "locked_parameters": {
            group: best_by_group[group]["locked_parameter"] for group in passed_groups
        },
        "only_passing_groups_may_open_2024": True,
        "no_outer_reselection": True,
        "2024_read": False,
        "2025_read": False,
    }
    _write_json(out_dir / "stage1_promotion_lock.json", lock)
    print(f"Stage1 exact-component passed groups: {passed_groups}", flush=True)
    return lock


def _load_stage1_lock(out_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    lock_path = out_dir / "stage1_promotion_lock.json"
    result_path = out_dir / "stage1_results.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if lock["preregister_sha256"] != PREREGISTER_SHA256:
        raise AssertionError("Stage1 lock preregistration changed")
    if lock["stage1_results"]["sha256"] != sha256_file(result_path):
        raise AssertionError("Stage1 result changed")
    passed = [
        group
        for group in TARGET_COLS
        if bool(result["group_results"][group].get("outer_passed", False))
    ]
    if passed != lock["passed_groups"]:
        raise AssertionError("Stage1 passed groups cannot be reproduced")
    for group in passed:
        if lock["locked_parameters"][group] != result["group_results"][group]["inner"]["locked_parameter"]:
            raise AssertionError("Stage1 parameter changed")
    return lock, result


def _read_component_set(
    artifact_root: Path,
    mapping: Mapping[str, str],
    index: pd.DatetimeIndex,
) -> tuple[dict[str, dict[str, pd.Series]], dict[str, Any]]:
    frames: dict[str, pd.DataFrame] = {}
    evidence: dict[str, Any] = {}
    for component in COMPONENTS:
        path = artifact_root / mapping[component]
        evidence[component] = describe_file(path)
        frame = _read_prediction(path, index, required_columns=TARGET_COLS)
        if tuple(frame.columns) != TARGET_COLS:
            raise AssertionError("conditional component target order changed")
        frames[component] = frame
    components = {
        group: {
            component: frames[component][group].rename(group)
            for component in COMPONENTS
        }
        for group in TARGET_COLS
    }
    return components, evidence


def _read_full_labels(raw_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = raw_dir / "train/train_labels.csv"
    identity = _assert_identity(path, FULL_LABEL_IDENTITY)
    frame = pd.read_csv(path, encoding="utf-8-sig")
    if tuple(frame.columns) != ("kst_dtm", *TARGET_COLS) or len(frame) != 26304:
        raise AssertionError("full label schema changed")
    index = pd.to_datetime(frame.pop("kst_dtm"), errors="raise")
    frame.index = pd.DatetimeIndex(index, name="forecast_kst_dtm")
    frame = frame.astype(np.float64)
    expected = pd.date_range(
        "2022-01-01 01:00:00", "2025-01-01 00:00:00", freq="h", name="forecast_kst_dtm"
    )
    if not frame.index.equals(expected) or np.isinf(frame.to_numpy()).any():
        raise AssertionError("full label index or values changed")
    return frame, identity


def _macro_triplet(actual: pd.DataFrame, prediction: pd.DataFrame) -> dict[str, float]:
    metrics = score_details(actual.loc[:, list(TARGET_COLS)], prediction.loc[:, list(TARGET_COLS)])
    return {
        "score": float(metrics.total_score),
        "one_minus_nmae": float(metrics.one_minus_nmae),
        "ficr": float(metrics.ficr),
    }


def _macro_compare_blocks(
    actual: pd.DataFrame,
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    blocks: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    if not actual.index.equals(baseline.index) or not actual.index.equals(candidate.index):
        raise ValueError("macro frames are not aligned")
    output: dict[str, Any] = {}
    for name, bounds in blocks.items():
        start, end = map(pd.Timestamp, bounds)
        index = actual.index[(actual.index >= start) & (actual.index <= end)]
        base = _macro_triplet(actual.loc[index], baseline.loc[index])
        cand = _macro_triplet(actual.loc[index], candidate.loc[index])
        output[name] = {
            "rows": len(index),
            "baseline": base,
            "candidate": cand,
            "delta": {metric: float(cand[metric] - base[metric]) for metric in base},
        }
    return output


def _passes_macro(comparisons: Mapping[str, Any]) -> bool:
    required = ("full", "H1", "H2")
    if any(float(comparisons[name]["delta"]["score"]) <= 0.0 for name in required):
        return False
    full = comparisons["full"]["delta"]
    nmae = float(full["one_minus_nmae"])
    ficr = float(full["ficr"])
    return nmae >= 0.0 and ficr >= 0.0 and (nmae > 0.0 or ficr > 0.0)


def _stage2(
    *,
    raw_dir: Path,
    artifact_root: Path,
    out_dir: Path,
    preregister: Mapping[str, Any],
) -> dict[str, Any]:
    stage1_lock, _ = _load_stage1_lock(out_dir)
    passed_groups = list(stage1_lock["passed_groups"])
    result_path = out_dir / "stage2_results.json"
    lock_path = out_dir / "stage2_promotion_lock.json"
    if result_path.exists() or lock_path.exists():
        raise FileExistsError("Stage2 output already exists")
    if not passed_groups:
        result = {
            "preregister_sha256": PREREGISTER_SHA256,
            "performed": False,
            "reason": "no Stage1 group passed fixed outer confirmation",
            "stage1_passed_groups": [],
            "individual_passed_groups": [],
            "promoted_groups": [],
            "2024_component_read": False,
            "2024_label_read": False,
            "recent_v4_read": False,
            "2025_read": False,
        }
        _write_json(result_path, result)
        lock = {
            "preregister_sha256": PREREGISTER_SHA256,
            "stage1_promotion_lock": describe_file(out_dir / "stage1_promotion_lock.json"),
            "stage2_results": describe_file(result_path),
            "promoted_groups": [],
            "2024_read": False,
            "2025_read": False,
        }
        _write_json(lock_path, lock)
        return lock

    recipe, recipe_identity = _load_recipe(
        PROJECT_DIR / "configs/train_final.v3.locked.json", preregister
    )
    components, component_evidence = _read_component_set(
        artifact_root, GATE_COMPONENT_FILES, YEAR_2024
    )
    v3_path = artifact_root / "oof/gate2024_locked_v3_cf_fix.parquet"
    recent_path = artifact_root / "oof/gate2024_recent_v4_cf_fix_calibration_fit.parquet"
    v3_identity = describe_file(v3_path)
    recent_identity = describe_file(recent_path)
    v3 = _read_prediction(v3_path, YEAR_2024, required_columns=TARGET_COLS)
    recent = _read_prediction(recent_path, YEAR_2024, required_columns=TARGET_COLS)
    direct_by_group: dict[str, pd.Series] = {}
    transferred_by_group: dict[str, pd.Series] = {}
    assembly: dict[str, Any] = {}
    for group in passed_groups:
        direct, metadata = assemble_candidate(
            components[group],
            group=group,
            recipe=recipe,
            parameter=stage1_lock["locked_parameters"][group],
            identity_baseline_kwh=v3[group],
        )
        transferred = transfer_delta(
            recent[group], v3[group], direct, capacity_kwh=CAPACITY_KWH[group]
        )
        direct_by_group[group] = direct
        transferred_by_group[group] = transferred
        assembly[group] = metadata
    direct_path = out_dir / "predictions/stage2_fixed_v3_candidates.parquet"
    transfer_path = out_dir / "predictions/stage2_fixed_recent_v4_interactions.parquet"
    _atomic_parquet(
        pd.concat([direct_by_group[group].rename(group) for group in passed_groups], axis=1),
        direct_path,
    )
    _atomic_parquet(
        pd.concat([transferred_by_group[group].rename(group) for group in passed_groups], axis=1),
        transfer_path,
    )
    prescore_path = out_dir / "stage2_prescore_lock.json"
    _write_json(
        prescore_path,
        {
            "preregister_sha256": PREREGISTER_SHA256,
            "stage1_promotion_lock": describe_file(out_dir / "stage1_promotion_lock.json"),
            "passed_groups_only": passed_groups,
            "locked_parameters": stage1_lock["locked_parameters"],
            "recipe": recipe_identity,
            "component_inputs": component_evidence,
            "v3_baseline": v3_identity,
            "recent_v4_baseline": recent_identity,
            "assembly": assembly,
            "direct_candidates": describe_file(direct_path),
            "recent_v4_interactions": describe_file(transfer_path),
            "2024_application_target_values_materialized": 0,
            "created_before_2024_application_labels": True,
            "2025_read": False,
        },
    )

    labels, label_identity = _read_full_labels(raw_dir)
    actual = labels.loc[YEAR_2024]
    group_results: dict[str, Any] = {}
    individual_passed: list[str] = []
    for group in passed_groups:
        direct_comparisons = compare_blocks(
            actual[group],
            v3[group],
            direct_by_group[group],
            group=group,
            blocks=STAGE2_BLOCKS,
        )
        interaction_comparisons = compare_blocks(
            actual[group],
            recent[group],
            transferred_by_group[group],
            group=group,
            blocks=STAGE2_BLOCKS,
        )
        direct_passed = passes_block_gate(direct_comparisons, full_block="full")
        interaction_passed = passes_block_gate(
            interaction_comparisons, full_block="full"
        )
        passed = direct_passed and interaction_passed
        if passed:
            individual_passed.append(group)
        group_results[group] = {
            "direct_v3": direct_comparisons,
            "recent_v4_interaction": interaction_comparisons,
            "direct_passed": direct_passed,
            "interaction_passed": interaction_passed,
            "individually_passed": passed,
        }
    mixed_direct = v3.copy()
    mixed_interaction = recent.copy()
    for group in individual_passed:
        mixed_direct[group] = direct_by_group[group]
        mixed_interaction[group] = transferred_by_group[group]
    mixed_direct_comparisons = _macro_compare_blocks(
        actual, v3, mixed_direct, STAGE2_BLOCKS
    )
    mixed_interaction_comparisons = _macro_compare_blocks(
        actual, recent, mixed_interaction, STAGE2_BLOCKS
    )
    mixed_direct_passed = bool(individual_passed) and _passes_macro(
        mixed_direct_comparisons
    )
    mixed_interaction_passed = bool(individual_passed) and _passes_macro(
        mixed_interaction_comparisons
    )
    promoted = (
        individual_passed
        if mixed_direct_passed and mixed_interaction_passed
        else []
    )
    mixed_direct_path = out_dir / "predictions/stage2_mixed_v3_after_group_gates.parquet"
    mixed_interaction_path = out_dir / "predictions/stage2_mixed_recent_v4_after_group_gates.parquet"
    _atomic_parquet(mixed_direct, mixed_direct_path)
    _atomic_parquet(mixed_interaction, mixed_interaction_path)
    result = {
        "preregister_sha256": PREREGISTER_SHA256,
        "performed": True,
        "stage1_passed_groups": passed_groups,
        "individual_passed_groups": individual_passed,
        "promoted_groups": promoted,
        "prescore_lock": describe_file(prescore_path),
        "score_labels_read_after_prescore_lock": label_identity,
        "group_results": group_results,
        "mixed_direct_v3": {
            "comparisons": mixed_direct_comparisons,
            "passed": mixed_direct_passed,
            "prediction": describe_file(mixed_direct_path),
        },
        "mixed_recent_v4_interaction": {
            "comparisons": mixed_interaction_comparisons,
            "passed": mixed_interaction_passed,
            "prediction": describe_file(mixed_interaction_path),
        },
        "no_2024_parameter_refit_reselection_or_group_substitution": True,
        "2025_read": False,
    }
    _write_json(result_path, result)
    lock = {
        "preregister_sha256": PREREGISTER_SHA256,
        "stage1_promotion_lock": describe_file(out_dir / "stage1_promotion_lock.json"),
        "stage2_results": describe_file(result_path),
        "promoted_groups": promoted,
        "locked_parameters": {
            group: stage1_lock["locked_parameters"][group] for group in promoted
        },
        "recent_v4_interaction_gate_passed": mixed_interaction_passed,
        "direct_v3_gate_passed": mixed_direct_passed,
        "2024_read": True,
        "2025_read": False,
        "only_promoted_groups_may_open_2025": True,
    }
    _write_json(lock_path, lock)
    print(f"Stage2 exact-component promoted groups: {promoted}", flush=True)
    return lock


def _load_stage2_lock(out_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    lock_path = out_dir / "stage2_promotion_lock.json"
    result_path = out_dir / "stage2_results.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if lock["preregister_sha256"] != PREREGISTER_SHA256:
        raise AssertionError("Stage2 preregistration changed")
    if lock["stage2_results"]["sha256"] != sha256_file(result_path):
        raise AssertionError("Stage2 result changed")
    expected = (
        list(result["individual_passed_groups"])
        if result.get("performed")
        and result["mixed_direct_v3"]["passed"]
        and result["mixed_recent_v4_interaction"]["passed"]
        else []
    )
    if expected != lock["promoted_groups"]:
        raise AssertionError("Stage2 promotion cannot be reproduced")
    return lock, result


def _stage_final(
    *,
    raw_dir: Path,
    artifact_root: Path,
    out_dir: Path,
    preregister: Mapping[str, Any],
) -> dict[str, Any]:
    stage2_lock, _ = _load_stage2_lock(out_dir)
    promoted = list(stage2_lock["promoted_groups"])
    result_path = out_dir / "final_results.json"
    if result_path.exists():
        raise FileExistsError(result_path)
    if not promoted:
        result = {
            "preregister_sha256": PREREGISTER_SHA256,
            "performed": False,
            "reason": "complete direct-v3 and recent-v4 interaction gates did not pass",
            "promoted_groups": [],
            "2025_components_read": False,
            "2025_v3_read": False,
            "2025_recent_v4_read": False,
            "sample_read": False,
            "csv_written": False,
        }
        _write_json(result_path, result)
        return result
    recipe, recipe_identity = _load_recipe(
        PROJECT_DIR / "configs/train_final.v3.locked.json", preregister
    )
    components, component_evidence = _read_component_set(
        artifact_root, FINAL_COMPONENT_FILES, YEAR_2025
    )
    v3_path = artifact_root / "final_cf_fix/predictions/corrected_v3_test.parquet"
    recent_path = artifact_root / "final_cf_fix/predictions/corrected_recent_v4_test.parquet"
    v3 = _read_prediction(v3_path, YEAR_2025, required_columns=TARGET_COLS)
    recent = _read_prediction(recent_path, YEAR_2025, required_columns=TARGET_COLS)
    final = recent.copy()
    direct_predictions: dict[str, pd.Series] = {}
    assembly: dict[str, Any] = {}
    for group in promoted:
        direct, metadata = assemble_candidate(
            components[group],
            group=group,
            recipe=recipe,
            parameter=stage2_lock["locked_parameters"][group],
            identity_baseline_kwh=v3[group],
        )
        final[group] = transfer_delta(
            recent[group], v3[group], direct, capacity_kwh=CAPACITY_KWH[group]
        )
        direct_predictions[group] = direct
        assembly[group] = metadata
    for group in TARGET_COLS:
        values = final[group].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all() or np.any(values < 0.0) or np.any(values > 1.02 * CAPACITY_KWH[group]):
            raise AssertionError("final prediction violates bounds")
        if group not in promoted and not np.array_equal(values, recent[group].to_numpy()):
            raise AssertionError("identity group differs from recent-v4")
    direct_path = out_dir / "final/fixed_candidate_v3_2025.parquet"
    prediction_path = out_dir / "final/exact_component_maximin_recent_v4_2025.parquet"
    _atomic_parquet(
        pd.concat([direct_predictions[group].rename(group) for group in promoted], axis=1),
        direct_path,
    )
    _atomic_parquet(final, prediction_path)
    sample_path = raw_dir / "sample_submission.csv"
    sample_identity = _assert_identity(sample_path, SAMPLE_IDENTITY)
    sample = pd.read_csv(sample_path, encoding="utf-8-sig")
    expected_columns = ("forecast_id", "forecast_kst_dtm", *TARGET_COLS)
    if tuple(sample.columns) != expected_columns or len(sample) != len(YEAR_2025):
        raise AssertionError("sample submission changed")
    times = pd.to_datetime(sample["forecast_kst_dtm"], errors="raise")
    if not pd.DatetimeIndex(times).equals(YEAR_2025.rename("forecast_kst_dtm")):
        raise AssertionError("sample timestamp sequence changed")
    submission = sample.copy()
    for group in TARGET_COLS:
        submission[group] = final[group].to_numpy(dtype=np.float64)
    csv_path = out_dir / "final/exact_component_maximin_recent_v4_2025.csv"
    _atomic_csv(submission, csv_path)
    if csv_path.read_bytes()[:3] != b"\xef\xbb\xbf":
        raise AssertionError("CSV BOM missing")
    readback = pd.read_csv(csv_path, encoding="utf-8-sig")
    if tuple(readback.columns) != expected_columns or len(readback) != len(submission):
        raise AssertionError("CSV readback schema changed")
    for group in TARGET_COLS:
        if not np.array_equal(
            readback[group].to_numpy(dtype=np.float64),
            np.round(submission[group].to_numpy(dtype=np.float64), 6),
        ):
            raise AssertionError("CSV readback values changed")
    result = {
        "preregister_sha256": PREREGISTER_SHA256,
        "performed": True,
        "promoted_groups": promoted,
        "recipe": recipe_identity,
        "component_inputs": component_evidence,
        "v3_baseline": describe_file(v3_path),
        "recent_v4_baseline": describe_file(recent_path),
        "sample": sample_identity,
        "assembly": assembly,
        "direct_prediction": describe_file(direct_path),
        "final_prediction": describe_file(prediction_path),
        "csv": describe_file(csv_path),
        "csv_written": True,
        "leaderboard_score_claim": False,
    }
    _write_json(result_path, result)
    return result


def _write_manifest(out_dir: Path, preregister_path: Path) -> Path:
    path = out_dir / "manifest.json"
    if path.exists():
        raise FileExistsError(path)
    outputs = sorted(
        (candidate for candidate in out_dir.rglob("*") if candidate.is_file()),
        key=lambda candidate: candidate.relative_to(out_dir).as_posix(),
    )
    forbidden = ("public", "scale")
    if any(
        token in candidate.relative_to(out_dir).as_posix().lower()
        for candidate in outputs
        for token in forbidden
    ):
        raise AssertionError("forbidden output path")
    stage1 = json.loads((out_dir / "stage1_results.json").read_text(encoding="utf-8"))
    stage2 = json.loads((out_dir / "stage2_results.json").read_text(encoding="utf-8"))
    final = json.loads((out_dir / "final_results.json").read_text(encoding="utf-8"))
    manifest = {
        "schema_version": 1,
        "experiment_id": "exact_component_maximin_hpo_strict_v1",
        "created_utc": utc_now(),
        "preregister_sha256": PREREGISTER_SHA256,
        "runtime": {
            "packages": package_versions(("numpy", "pandas", "optuna", "pyarrow")),
            "git": git_state(PROJECT_DIR),
        },
        "provenance": _snapshot(_provenance_paths(preregister_path)),
        "outputs": [describe_file(candidate) for candidate in outputs],
        "results": {
            "stage1_passed_groups": stage1["passed_groups"],
            "stage2_promoted_groups": stage2["promoted_groups"],
            "recent_v4_interaction_gate_executed": bool(stage2.get("performed", False)),
            "recent_v4_interaction_gate_passed": bool(
                stage2.get("mixed_recent_v4_interaction", {}).get("passed", False)
            ),
            "final_performed": bool(final["performed"]),
            "csv_written": bool(final.get("csv_written", False)),
            "leaderboard_score_claim": False,
        },
        "forbidden_input_audit": {
            "public_artifacts_read": False,
            "scale_artifacts_read": False,
            "public_metric_triplets_used": False,
            "forbidden_paths_in_input_or_provenance": 0,
        },
    }
    _write_json(path, manifest)
    return path


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    preregister_path = args.preregister.resolve()
    preregister = _verify_preregister(preregister_path)
    raw_dir = args.raw_dir.resolve()
    artifact_root = args.artifact_root.resolve()
    out_dir = args.out_dir.resolve()
    if args.stage in ("stage1", "all"):
        _stage1(
            raw_dir=raw_dir,
            artifact_root=artifact_root,
            out_dir=out_dir,
            preregister_path=preregister_path,
            preregister=preregister,
        )
    if args.stage in ("stage2", "all"):
        _stage2(
            raw_dir=raw_dir,
            artifact_root=artifact_root,
            out_dir=out_dir,
            preregister=preregister,
        )
    if args.stage in ("final", "all"):
        _stage_final(
            raw_dir=raw_dir,
            artifact_root=artifact_root,
            out_dir=out_dir,
            preregister=preregister,
        )
    if args.stage == "all":
        manifest = _write_manifest(out_dir, preregister_path)
        print(f"manifest: {manifest} ({sha256_file(manifest)})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
