"""Independent read-only auditor for the nonwind-cap/scale-0.97 overlay.

Two deliberately separate modes are provided:

``--static-audit``
    Reads only the three append-only science-head JSON files, their sidecars,
    and (when present) runner source text.  It never opens labels, models,
    Parquet arrays, or CSV data.  Missing postrun outputs are a conditional
    skip, not a failure.

``--postrun``
    Uses the runner's hash-bound locks and root/stress manifests (or an optional
    compatible semantic contract) and independently replays every formula and
    veto.  The resulting audit JSON is written once to a caller-selected,
    previously absent path.  The auditor never fits a model, changes an
    artifact, retries a candidate, or creates a submission.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
from importlib import metadata as importlib_metadata
import io
import json
import math
import os
from pathlib import Path
import pickle
import platform
import re
import sys
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.metric import CAPACITY_KWH, TARGET_COLS, group_metrics  # noqa: E402


SCIENCE_HEADS: tuple[dict[str, Any], ...] = (
    {
        "name": "v1",
        "relative": "configs/feature_exit_nonwind_cap025143_scale097_overlay_preregister_v1.json",
        "bytes": 15_828,
        "sha256": "91fa72355bab5bef80dcb73816e9bea162bf80ef26183333f9798d32770e42f9",
    },
    {
        "name": "v2",
        "relative": "configs/feature_exit_nonwind_cap025143_scale097_overlay_preregister_v2.json",
        "bytes": 2_110,
        "sha256": "802dc38d6692eb6ecc546bc42743609e06d4bef79deaa6103dd5fba9843cb0da",
    },
    {
        "name": "v3",
        "relative": "configs/feature_exit_nonwind_cap025143_scale097_overlay_preregister_v3.json",
        "bytes": 2_704,
        "sha256": "09e2b3ca41b8be49c5084b986dc6f0b2f7eedc432f65525ceb65cf2e3b3ac650",
    },
)

RUNNER_RELATIVE = "scripts/run_feature_exit_nonwind_cap025143_scale097_overlay.py"
OUTPUT_RELATIVE = "artifacts/postgate/feature_exit_nonwind_cap025143_scale097_overlay_v1"
CAP_CF = 0.02514322112138523
CAP_HEX = "0x1.9bf2501bad320p-6"
PREFIX_BYTES = 742_551
PREFIX_SHA256 = "ee9707cc81229db760fc035d0bf1d10eb3586d2fef95849bdf463d73fae3aacd"
SUFFIX_OFFSET = 742_551
SUFFIX_BYTES = 396_416
SUFFIX_SHA256 = "0678740abe23800de9be272d2992ff71e59571ac169a410da1d8a3ab28f544da"
FULL_LABEL_SHA256 = "47bb64252195cf4734e67394d6e50485f27a608def3b5a8791fcc7674bbceb03"
FULL_LABEL_BYTES = 1_138_967
SAMPLE_SHA256 = "c925d2066a834f937f8091ed55acfe50ff86c8be4745b52c3adc95b056c5aaaa"
FINAL_BASE_SHA256 = "85ad8c63eaf7bff4b0390fb8c42dbb31c22acd5e7a64fbec350d9c3878818621"
ORIGINAL_GATE_A0_SHA256 = "1c0a2e70996566c766a90a557c83ab1ba010277c31d202a1e4209188b94d112c"
CONTROL_FEATURE_SHA256 = "55835269be52c5ceb681435a17c8e1ff8c8f40f36393706691996cfc4c143276"
NONWIND_FEATURE_SHA256 = "9022f9459366dcdbe91e59866c8b8d82c75617716c38f939e06daf97f27586cb"
FEATURE_COUNTS = {"control_all": 612, "nonwind_atmospheric": 398}
STAGE_ELIGIBLE_COUNTS = {
    "stress": {"kpx_group_1": 10_925, "kpx_group_2": 10_914, "kpx_group_3": 4_847},
    "final": {"kpx_group_1": 15_915, "kpx_group_2": 15_891, "kpx_group_3": 9_414},
}
MODEL_PARAMS = {
    "objective": "quantile",
    "alpha": 0.7,
    "n_estimators": 1500,
    "learning_rate": 0.025,
    "num_leaves": 31,
    "min_child_samples": 30,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.75,
    "reg_alpha": 0.05,
    "reg_lambda": 2.0,
    "verbosity": -1,
    "deterministic": True,
    "force_col_wise": True,
    "random_state": 42,
    "n_jobs": 7,
}
COMPONENT_NAMES = (
    "lgb_l1",
    "lgb_q07",
    "shared_l1",
    "shared_q07",
    "top200_q07",
    "energy_q06",
)
PAIRED_VARIANTS = ("control_all", "nonwind_atmospheric")
CONTROL = PAIRED_VARIANTS[0]
VARIANT_ALIASES = {
    "control": "control_all",
    "control_all": "control_all",
    "nonwind": "nonwind_atmospheric",
    "nonwind_atmospheric": "nonwind_atmospheric",
}
SLICE_ORDER = ("FULL", "H1", "H2", "Q1", "Q2", "Q3", "Q4")
TEST_INDEX = pd.date_range(
    "2025-01-01 01:00:00",
    "2026-01-01 00:00:00",
    freq="h",
    name="forecast_kst_dtm",
)
STRESS_INDEX = pd.date_range(
    "2024-01-01 01:00:00",
    "2025-01-01 00:00:00",
    freq="h",
    name="forecast_kst_dtm",
)

# Exact runner ``SMALL_PROVENANCE`` order.  Keeping this independent mirror in
# the auditor makes omission, duplication, or reordering in source_lock a hard
# failure instead of accepting whichever subset the runner happened to list.
SMALL_PROVENANCE_IDENTITIES: tuple[tuple[str, int, str], ...] = (
    (
        "data/local/open/info.xlsx",
        3_823_422,
        "89e83a52e0eb2ce367a3573a96d6795ed4b4d4ac624965cb3530beec0cbd2bd6",
    ),
    (
        "artifacts/incidents/feature_exit_nonwind_cap025143_champion_overlay_v1_v2_preexecution_supersession_20260810.json",
        1_918,
        "d94a4d453c908c27404b5061957e322b60e7154141231173c86aa97ea7290479",
    ),
    (
        "artifacts/feature_exit_virtual_v1/audits/nonwind_cap_v3_evidence/nonwind_cap_evidence.json",
        25_202,
        "315b3e5d43ec62d395bf73e98bfbc85fc8d5fffb34c24c9c8f53a34307322840",
    ),
    (
        "scripts/build_nonwind_cap_v3_evidence.py",
        26_898,
        "d5edd8330c9a3f4b5c6d86f58d71993599f6de11259766f9c1d2a1794676159d",
    ),
    (
        "artifacts/feature_exit_virtual_v1/inner/inner_oof_predictions.parquet",
        826_334,
        "6907c9c27e3d98f0aaac6ca59e41c069889b235b5303a61a61793ea9e3278f2d",
    ),
    (
        "artifacts/gate/v3/gate_manifest.json",
        19_529,
        "c95ca55ef8d09668b6376aee3773f80fd2f89358c00cdc4e9f5bc9727952aeaa",
    ),
    (
        "artifacts/final_v3/v3_locked_full_2025__manifest.json",
        17_646,
        "f2181ebbb1666d278942c7555ad9bf6edb2ca97d7608c4981c3e6f719344a22a",
    ),
    (
        "artifacts/postgate/public_adaptive_scale097_g2_delta_v2/manifest_frozen_postrun_v2.json",
        14_871,
        "3ef6fabf059f3bd67e3b933e8eb633e5ebf3b422d7c123f1b27af461fcc2e153",
    ),
    (
        "configs/public_adaptive_scale097_g2_delta_preregister_v2.json",
        4_349,
        "9ed9bc8d4ec73f714cbd4d1b1c41ac8b5c45a7812a1bd2bbe28a8c5b5b24c475",
    ),
    (
        "src/metric.py",
        7_248,
        "555950e6892a808d9091b4e6749128b9f96888ce0a83270b6959a4e927dc5f1d",
    ),
    (
        "src/density_ratio.py",
        10_259,
        "f287cac2f1497525615b391fe97356684c2344b80fb6fd66f3fa6fc9c3bab6a5",
    ),
    (
        "src/virtual_feature_exit.py",
        13_892,
        "9ba3ce9445ac58cd4db57456ea11acd2f18e6e20daa2e55845f5e42f3d49bdfa",
    ),
    (
        "src/features.py",
        34_098,
        "34c7bf46444c9a2a88eabdc23f41254f66a9ba04eda1adfa53e19b1263a81a5e",
    ),
    (
        "scripts/build_features.py",
        5_857,
        "423a936e0f857f031fb891e11451ed037d68a99ee2aa1acf03b364f368d99f10",
    ),
)

OFFICIAL_TRAIN_RAW_IDENTITIES: tuple[tuple[str, int, str], ...] = (
    (
        "data/local/open/train/ldaps_train.csv",
        129_687_357,
        "61ae944e7ae1fcb17391be6737792a2205c6507bf2446ed5d9d0daf07fdea026",
    ),
    (
        "data/local/open/train/gfs_train.csv",
        84_315_594,
        "cd56b67d357e7bbaff5d0d51d3537d935c9e7a3f012e9f37516bdc4d38c66a5d",
    ),
)

SOURCE_TAIL_IDENTITIES: tuple[tuple[str, int, str], ...] = (
    (
        "artifacts/gate/v3/gate_recipe_snapshot.json",
        5_613,
        "c26e0b0b4268ed3780d85e1c3af5db38253d42e8a4230fbd910691f2ec6acd19",
    ),
    (
        "artifacts/final_v3/v3_locked_full_2025__recipe.json",
        5_613,
        "c26e0b0b4268ed3780d85e1c3af5db38253d42e8a4230fbd910691f2ec6acd19",
    ),
    (
        "artifacts/gate/v3/gate_oof.parquet",
        314_530,
        ORIGINAL_GATE_A0_SHA256,
    ),
    (
        "artifacts/postgate/public_adaptive_scale097_g2_delta_v2/diagnostic_2024/A_scale097.parquet",
        335_807,
        "93d8e2b7d19971ba46a29d548e77ef4f85553bbbdaad2b5983a5b6f1d84e406c",
    ),
)


class AuditFailure(AssertionError):
    """Raised when a fail-closed postrun invariant is not satisfied."""


@dataclass(frozen=True)
class CapReplay:
    raw_delta_cf: pd.DataFrame
    capped_delta_cf: pd.DataFrame
    q1_kwh: pd.DataFrame
    cap_hit_count: dict[str, int]
    q1_outer_clip_count: dict[str, int]


@dataclass(frozen=True)
class AssemblyReplay:
    a0_kwh: pd.DataFrame
    a1_safe_kwh: pd.DataFrame
    increment_d_kwh: pd.DataFrame
    prebin_z0_kwh: pd.DataFrame
    prebin_z1_kwh: pd.DataFrame
    g2_b0: np.ndarray
    g2_hypothetical_b1: np.ndarray
    g2_hypothetical_crossing_count: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ordered_names_sha256(values: Sequence[str]) -> str:
    payload = ("\n".join(map(str, values)) + "\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _record_size(record: Mapping[str, Any]) -> int:
    for key in ("size_bytes", "bytes"):
        if key in record:
            return int(record[key])
    raise AuditFailure("snapshot lacks bytes/size_bytes")


def verify_snapshot(record: Mapping[str, Any], *, root: Path) -> Path:
    raw = Path(str(record["path"]))
    path = raw if raw.is_absolute() else root / raw
    path = path.resolve()
    if not path.is_file():
        raise AuditFailure(f"snapshot file missing: {path}")
    if path.stat().st_size != _record_size(record):
        raise AuditFailure(f"snapshot size changed: {path}")
    if sha256_file(path) != str(record["sha256"]):
        raise AuditFailure(f"snapshot hash changed: {path}")
    return path


def _locked_source_identity(
    project_root: Path, path_text: str, size_bytes: int, sha256: str
) -> dict[str, Any]:
    raw = Path(path_text)
    path = raw if raw.is_absolute() else project_root / raw
    return {
        "path": str(path.resolve()),
        "size_bytes": int(size_bytes),
        "sha256": sha256,
        "hash_verified": True,
    }


def _science_source_identity(
    project_root: Path, record: Mapping[str, Any]
) -> dict[str, Any]:
    return _locked_source_identity(
        project_root,
        str(record["path"]),
        _record_size(record),
        str(record["sha256"]),
    )


def expected_verified_source_inputs(
    project_root: Path, science_v1: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Return the exact ordered 29-record source closure used by the runner."""

    expected = [
        _locked_source_identity(project_root, path, size, digest)
        for path, size, digest in SMALL_PROVENANCE_IDENTITIES
    ]
    expected.extend(
        _locked_source_identity(project_root, path, size, digest)
        for path, size, digest in OFFICIAL_TRAIN_RAW_IDENTITIES
    )
    data = science_v1.get("data_contract")
    historical = science_v1.get("historical_veto_2024")
    if not isinstance(data, Mapping) or not isinstance(historical, Mapping):
        raise AuditFailure("science v1 source declarations are absent")
    train_caches = data.get("train_caches")
    gate_components = historical.get("components_2024")
    if (
        not isinstance(train_caches, Mapping)
        or list(train_caches) != list(TARGET_COLS)
        or not isinstance(gate_components, Mapping)
        or list(gate_components) != list(COMPONENT_NAMES)
    ):
        raise AuditFailure("science v1 ordered source declarations differ")
    expected.extend(
        _science_source_identity(project_root, train_caches[group])
        for group in TARGET_COLS
    )
    expected.extend(
        _science_source_identity(project_root, gate_components[name])
        for name in COMPONENT_NAMES
    )
    expected.extend(
        _locked_source_identity(project_root, path, size, digest)
        for path, size, digest in SOURCE_TAIL_IDENTITIES
    )
    if len(expected) != 29:
        raise AuditFailure("internal exact source-closure cardinality differs")
    return expected


def audit_verified_source_inputs(
    source_inputs: Any,
    *,
    project_root: Path,
    science_v1: Mapping[str, Any],
) -> list[dict[str, Any]]:
    expected = expected_verified_source_inputs(project_root, science_v1)
    if not isinstance(source_inputs, list) or source_inputs != expected:
        raise AuditFailure(
            "source lock exact verified-input set/order/count differs"
        )
    verified: list[dict[str, Any]] = []
    for expected_record, record in zip(expected, source_inputs, strict=True):
        if not isinstance(record, Mapping) or record.get("hash_verified") is not True:
            raise AuditFailure("source lock contains an unverified input")
        path = verify_snapshot(record, root=project_root)
        actual = file_record(path)
        if actual != {
            key: expected_record[key] for key in ("path", "size_bytes", "sha256")
        }:
            raise AuditFailure("source lock verified-input identity changed")
        verified.append(actual)
    return verified


def _sidecar_hash(path: Path) -> str:
    sidecar = path.with_name(path.name + ".sha256")
    if not sidecar.is_file():
        raise AuditFailure(f"science sidecar missing: {sidecar}")
    fields = sidecar.read_text(encoding="ascii").split()
    if not fields:
        raise AuditFailure(f"science sidecar empty: {sidecar}")
    return fields[0].lower()


def load_effective_science_head(project_root: Path) -> tuple[dict[str, Any], ...]:
    documents: list[dict[str, Any]] = []
    for spec in SCIENCE_HEADS:
        path = project_root / spec["relative"]
        if not path.is_file():
            raise AuditFailure(f"science head missing: {path}")
        if path.stat().st_size != int(spec["bytes"]):
            raise AuditFailure(f"science head size changed: {path}")
        digest = sha256_file(path)
        if digest != spec["sha256"] or _sidecar_hash(path) != digest:
            raise AuditFailure(f"science head or sidecar hash changed: {path}")
        documents.append(json.loads(path.read_text(encoding="utf-8")))
    v1, v2, v3 = documents
    if v2["supersedes"]["sha256"] != SCIENCE_HEADS[0]["sha256"]:
        raise AuditFailure("v2 does not bind v1")
    if (
        v3["supersedes"]["science_v1"]["sha256"]
        != SCIENCE_HEADS[0]["sha256"]
        or v3["supersedes"]["reference_correction_v2"]["sha256"]
        != SCIENCE_HEADS[1]["sha256"]
    ):
        raise AuditFailure("v3 does not bind v1+v2")
    return tuple(documents)


def static_audit(project_root: Path) -> dict[str, Any]:
    """Audit only immutable text contracts; never open data arrays."""

    v1, v2, v3 = load_effective_science_head(project_root)
    checks = {
        "effective_head_chain_exact": True,
        "v2_original_gate_A0_corrected": (
            v2["correction"]["effective_original_gate_A0_reference"]["sha256"]
            == ORIGINAL_GATE_A0_SHA256
            and v2["correction"]["incorrect_v1_stress_reference"][
                "must_not_be_used_by_this_experiment"
            ]
        ),
        "cap_decimal_and_binary64_exact": (
            float(v1["cap_evidence"]["cap_cf"]) == CAP_CF
            and str(v1["cap_evidence"]["cap_binary64_hex"]) == CAP_HEX
            and float(CAP_CF).hex() == CAP_HEX
        ),
        "cap_once_and_q1_outer_clip_frozen": (
            v1["delta_transfer_formula"]["cap_applied_exactly_once"]
            and "canonical_q07_kwh/capacity_kwh + capped_delta_cf"
            in v1["delta_transfer_formula"]["per_group"][2]
        ),
        "g2_b0_frozen_and_b1_report_only": (
            "b0" in v1["locked_assembly_contract"]["g2_frozen_baseline_bin"]
            and "report-only"
            in v1["locked_assembly_contract"]["g2_frozen_baseline_bin"][
                "hypothetical_b1_and_crossing_count"
            ]
        ),
        "seven_veto_slices_exact": tuple(v1["historical_veto_2024"]["slices"])
        == SLICE_ORDER,
        "seven_veto_rules_exact": (
            v1["historical_veto_2024"]["pass_gates"]
            == {
                "all_seven_mixed_score_deltas_strictly_positive": True,
                "full_one_minus_nmae_delta_min": 0.0,
                "full_ficr_delta_strictly_positive": True,
            }
        ),
        "final_base_85ad_and_formula_exact": (
            v1["final_base_contract"]["prediction"]["sha256"]
            == FINAL_BASE_SHA256
            and v1["final_base_contract"]["formula"]
            == "final_kwh = clip(full_precision_scale097_base_kwh + D, 0, 1.02*capacity_kwh)"
        ),
        "csv_contract_exact": (
            v1["execution_and_output_contract"]["utf8_bom"]
            and v1["execution_and_output_contract"]["line_endings"] == "LF"
            and int(v1["execution_and_output_contract"]["decimal_places"]) == 6
            and int(v1["execution_and_output_contract"]["rows"]) == 8760
            and v1["execution_and_output_contract"]["text_roundtrip_exact"]
        ),
        "v3_bounded_access_order_exact": (
            len(v3["effective_label_access_order"]) == 6
            and "[0,742551)" in v3["effective_label_access_order"][0]
            and "[742551,1138967)" in v3["effective_label_access_order"][2]
            and "Do not reread" in v3["effective_label_access_order"][3]
        ),
        "terminal_no_rescue_frozen": (
            "terminal reject"
            in v1["historical_veto_2024"]["failure_action"]
            and v1["risk_disclosure"][
                "no_further_feature_cap_weight_group_month_threshold_base_or_formula_search"
            ]
        ),
    }
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise AuditFailure(f"static science audit failed: {failures}")

    conditional_skips: list[dict[str, str]] = []
    runner = project_root / RUNNER_RELATIVE
    if runner.is_file():
        runner_record: dict[str, Any] | None = file_record(runner)
        source = runner.read_text(encoding="utf-8")
        required_literals = (
            SCIENCE_HEADS[0]["sha256"],
            SCIENCE_HEADS[1]["sha256"],
            SCIENCE_HEADS[2]["sha256"],
            repr(CAP_CF),
            FINAL_BASE_SHA256,
            "side=\"right\"",
        )
        absent = [literal for literal in required_literals if literal not in source]
        if absent:
            conditional_skips.append(
                {
                    "check": "runner_literal_preflight",
                    "reason": f"runner exists but literals may be indirect: {absent}",
                }
            )
    else:
        runner_record = None
        conditional_skips.append(
            {"check": "runner_source", "reason": "runner not present yet"}
        )

    output_root = project_root / OUTPUT_RELATIVE
    if not output_root.exists():
        conditional_skips.append(
            {
                "check": "postrun_outputs",
                "reason": "output root absent at static audit; postrun replay skipped",
            }
        )
    else:
        conditional_skips.append(
            {
                "check": "postrun_outputs",
                "reason": (
                    "output root exists but static mode deliberately reads no arrays; "
                    "run --postrun after root manifest is durable"
                ),
            }
        )
    return {
        "schema_version": 1,
        "audit_mode": "static_only_no_arrays_no_labels_no_models_no_csv",
        "status": "PASS_STATIC",
        "checks": checks,
        "conditional_skips": conditional_skips,
        "science_heads": [
            file_record(project_root / spec["relative"]) for spec in SCIENCE_HEADS
        ],
        "runner": runner_record,
        "output_root_exists": output_root.exists(),
        "access_ledger": {
            "config_json_files_read": 3,
            "sidecar_text_files_read": 3,
            "runner_source_text_read": int(runner.is_file()),
            "label_bytes_read": 0,
            "parquet_arrays_read": 0,
            "model_files_loaded": 0,
            "test_arrays_read": 0,
            "csv_files_read": 0,
            "files_written": 0,
            "model_fits": 0,
        },
    }


def cap_q07_component(
    canonical_q07_kwh: pd.DataFrame,
    paired_control_cf: pd.DataFrame,
    paired_exit_cf: pd.DataFrame,
    *,
    cap_cf: float = CAP_CF,
) -> CapReplay:
    """Replay raw CF delta -> one cap -> q1 outer clip for all groups."""

    for frame in (paired_control_cf, paired_exit_cf):
        if not canonical_q07_kwh.index.equals(frame.index):
            raise AuditFailure("paired CF and canonical q07 indices differ")
        if tuple(frame.columns) != TARGET_COLS:
            raise AuditFailure("paired CF columns differ from target groups")
    if tuple(canonical_q07_kwh.columns) != TARGET_COLS:
        raise AuditFailure("canonical q07 columns differ from target groups")
    raw = paired_exit_cf.astype(np.float64) - paired_control_cf.astype(np.float64)
    capped = raw.clip(lower=-float(cap_cf), upper=float(cap_cf))
    q1 = canonical_q07_kwh.astype(np.float64).copy()
    cap_hits: dict[str, int] = {}
    outer_hits: dict[str, int] = {}
    for group in TARGET_COLS:
        capacity = float(CAPACITY_KWH[group])
        raw_values = raw[group].to_numpy(dtype=np.float64)
        cap_hits[group] = int(np.count_nonzero(np.abs(raw_values) > cap_cf))
        unbounded = (
            canonical_q07_kwh[group].to_numpy(dtype=np.float64) / capacity
            + capped[group].to_numpy(dtype=np.float64)
        )
        outer_hits[group] = int(np.count_nonzero((unbounded < 0.0) | (unbounded > 1.02)))
        q1[group] = capacity * np.clip(unbounded, 0.0, 1.02)
    if not np.isfinite(q1.to_numpy(dtype=np.float64)).all():
        raise AuditFailure("q1 replay produced non-finite values")
    return CapReplay(raw, capped, q1, cap_hits, outer_hits)


def replay_locked_assembly(
    components: Mapping[str, pd.DataFrame],
    q1_kwh: pd.DataFrame,
    *,
    ensemble: Mapping[str, Any],
) -> AssemblyReplay:
    """Replay A0/A1 with G2's original b0 frozen for the A1 adjustment."""

    if set(components) != set(COMPONENT_NAMES):
        raise AuditFailure("six-component set changed")
    reference = components["lgb_q07"]
    if tuple(reference.columns) != TARGET_COLS:
        raise AuditFailure("component target columns changed")
    if not reference.index.equals(q1_kwh.index):
        raise AuditFailure("q1 and components have different indices")
    for name, frame in components.items():
        if not frame.index.equals(reference.index) or tuple(frame.columns) != TARGET_COLS:
            raise AuditFailure(f"component alignment changed: {name}")

    a0 = pd.DataFrame(index=reference.index, columns=TARGET_COLS, dtype=np.float64)
    a1 = a0.copy()
    z0_frame = a0.copy()
    z1_frame = a0.copy()
    g2_b0 = np.empty(0, dtype=np.int64)
    g2_b1 = np.empty(0, dtype=np.int64)
    for group in TARGET_COLS:
        capacity = float(CAPACITY_KWH[group])
        weights = ensemble["weights"][group]
        expected_order = tuple(weights)
        if set(expected_order) != set(COMPONENT_NAMES):
            raise AuditFailure(f"locked weights changed for {group}")
        arrays0 = {
            name: components[name][group].to_numpy(dtype=np.float64)
            for name in expected_order
        }
        arrays1 = dict(arrays0)
        arrays1["lgb_q07"] = q1_kwh[group].to_numpy(dtype=np.float64)
        weighted0 = sum(float(weights[name]) * arrays0[name] for name in expected_order)
        weighted1 = sum(float(weights[name]) * arrays1[name] for name in expected_order)
        affine = ensemble["affine"][group]
        unbounded0 = float(affine["scale"]) * weighted0 + float(affine["bias_kwh"])
        unbounded1 = float(affine["scale"]) * weighted1 + float(affine["bias_kwh"])
        clip_spec = ensemble["clip"][group]
        lower = float(clip_spec["lower_capacity_fraction"]) * capacity
        upper = float(clip_spec["upper_capacity_fraction"]) * capacity
        z0 = np.clip(unbounded0, lower, upper)
        z1 = np.clip(unbounded1, lower, upper)
        z0_frame[group] = z0
        z1_frame[group] = z1
        bin_spec = ensemble["power_bins"][group]
        if bin_spec is None:
            a0[group] = z0
            a1[group] = z1
            continue
        edges = np.asarray(bin_spec["edges_cf"], dtype=np.float64)
        deltas = np.asarray(bin_spec["delta_kwh"], dtype=np.float64)
        b0 = np.searchsorted(edges[1:-1], z0 / capacity, side="right")
        b1 = np.searchsorted(edges[1:-1], z1 / capacity, side="right")
        a0[group] = np.clip(z0 + deltas[b0], lower, upper)
        # Scientific head v2/v3 freezes b0 for A1_safe.  b1 is diagnostic only.
        a1[group] = np.clip(z1 + deltas[b0], lower, upper)
        if group == "kpx_group_2":
            g2_b0 = b0.astype(np.int64, copy=False)
            g2_b1 = b1.astype(np.int64, copy=False)
    increment = a1 - a0
    return AssemblyReplay(
        a0_kwh=a0,
        a1_safe_kwh=a1,
        increment_d_kwh=increment,
        prebin_z0_kwh=z0_frame,
        prebin_z1_kwh=z1_frame,
        g2_b0=g2_b0,
        g2_hypothetical_b1=g2_b1,
        g2_hypothetical_crossing_count=int(np.count_nonzero(g2_b0 != g2_b1)),
    )


def final_base_plus_increment(
    base_scale097_kwh: pd.DataFrame,
    increment_d_kwh: pd.DataFrame,
) -> pd.DataFrame:
    if not base_scale097_kwh.index.equals(increment_d_kwh.index):
        raise AuditFailure("base and D indices differ")
    if tuple(base_scale097_kwh.columns) != TARGET_COLS:
        raise AuditFailure("base columns differ from target groups")
    output = base_scale097_kwh.astype(np.float64).copy()
    for group in TARGET_COLS:
        output[group] = np.clip(
            base_scale097_kwh[group].to_numpy(dtype=np.float64)
            + increment_d_kwh[group].to_numpy(dtype=np.float64),
            0.0,
            1.02 * float(CAPACITY_KWH[group]),
        )
    return output


def exact_frame_equal(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    return (
        left.index.equals(right.index)
        and left.columns.equals(right.columns)
        and np.array_equal(
            left.to_numpy(dtype=np.float64), right.to_numpy(dtype=np.float64)
        )
    )


def _read_target_frame(path: Path, *, expected_index: pd.DatetimeIndex) -> pd.DataFrame:
    frame = pd.read_parquet(path, engine="pyarrow")
    if "forecast_kst_dtm" in frame.columns:
        frame = frame.set_index("forecast_kst_dtm")
    frame.index = pd.DatetimeIndex(frame.index, name="forecast_kst_dtm")
    if not frame.index.equals(expected_index):
        raise AuditFailure(f"prediction index changed: {path}")
    if tuple(frame.columns) != TARGET_COLS:
        raise AuditFailure(f"prediction target schema/order changed: {path}")
    result = frame.loc[:, list(TARGET_COLS)].astype(np.float64)
    if not np.isfinite(result.to_numpy(dtype=np.float64)).all():
        raise AuditFailure(f"prediction contains non-finite values: {path}")
    return result


def _read_apply_feature_cache(
    record: Mapping[str, Any],
    *,
    project_root: Path,
    expected_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    path = verify_snapshot(record, root=project_root)
    frame = pd.read_parquet(path, engine="pyarrow").astype(np.float32, copy=False)
    frame.index = pd.DatetimeIndex(frame.index, name="forecast_kst_dtm")
    if len(frame.columns) != FEATURE_COUNTS["control_all"]:
        raise AuditFailure(f"feature cache column count changed: {path}")
    names = list(map(str, frame.columns))
    if _ordered_names_sha256(names) != CONTROL_FEATURE_SHA256:
        raise AuditFailure(f"feature cache order SHA changed: {path}")
    missing = expected_index.difference(frame.index)
    if len(missing):
        raise AuditFailure(f"feature cache apply index incomplete: {path}")
    apply = frame.loc[expected_index]
    if not apply.index.equals(expected_index) or not np.isfinite(
        apply.to_numpy(dtype=np.float32)
    ).all():
        raise AuditFailure(f"feature cache apply rows changed: {path}")
    return apply


def _read_paired_cf(
    path: Path,
    *,
    expected_index: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.read_parquet(path, engine="pyarrow")
    if {"forecast_kst_dtm", "group", *PAIRED_VARIANTS}.issubset(frame.columns):
        frame["forecast_kst_dtm"] = pd.to_datetime(frame["forecast_kst_dtm"])
        controls: dict[str, pd.Series] = {}
        exits: dict[str, pd.Series] = {}
        for group in TARGET_COLS:
            rows = frame.loc[frame["group"] == group].set_index("forecast_kst_dtm")
            rows.index = pd.DatetimeIndex(rows.index, name="forecast_kst_dtm")
            controls[group] = rows[CONTROL].reindex(expected_index)
            exits[group] = rows["nonwind_atmospheric"].reindex(expected_index)
        control = pd.DataFrame(controls, index=expected_index, dtype=np.float64)
        exited = pd.DataFrame(exits, index=expected_index, dtype=np.float64)
    else:
        if "forecast_kst_dtm" in frame.columns:
            frame = frame.set_index("forecast_kst_dtm")
        frame.index = pd.DatetimeIndex(frame.index, name="forecast_kst_dtm")
        patterns = (
            ("{group}__{variant}",),
            ("{variant}__{group}",),
        )
        resolved: tuple[dict[str, str], dict[str, str]] | None = None
        for (pattern,) in patterns:
            control_cols = {
                group: pattern.format(group=group, variant=CONTROL)
                for group in TARGET_COLS
            }
            exit_cols = {
                group: pattern.format(group=group, variant="nonwind_atmospheric")
                for group in TARGET_COLS
            }
            if set(control_cols.values()) | set(exit_cols.values()) <= set(frame.columns):
                resolved = control_cols, exit_cols
                break
        if resolved is None:
            raise AuditFailure(f"unsupported paired-CF schema: {tuple(frame.columns)}")
        control_cols, exit_cols = resolved
        control = pd.DataFrame(
            {group: frame[column] for group, column in control_cols.items()},
            index=frame.index,
        ).reindex(expected_index)
        exited = pd.DataFrame(
            {group: frame[column] for group, column in exit_cols.items()},
            index=frame.index,
        ).reindex(expected_index)
    if control.isna().any().any() or exited.isna().any().any():
        raise AuditFailure("paired-CF rows are incomplete")
    return control.astype(np.float64), exited.astype(np.float64)


def _slice_indices_2024() -> dict[str, pd.DatetimeIndex]:
    bounds = {
        "FULL": ("2024-01-01 01:00:00", "2025-01-01 00:00:00"),
        "H1": ("2024-01-01 01:00:00", "2024-07-01 00:00:00"),
        "H2": ("2024-07-01 01:00:00", "2025-01-01 00:00:00"),
        "Q1": ("2024-01-01 01:00:00", "2024-04-01 00:00:00"),
        "Q2": ("2024-04-01 01:00:00", "2024-07-01 00:00:00"),
        "Q3": ("2024-07-01 01:00:00", "2024-10-01 00:00:00"),
        "Q4": ("2024-10-01 01:00:00", "2025-01-01 00:00:00"),
    }
    return {
        name: pd.date_range(start, end, freq="h", name="forecast_kst_dtm")
        for name, (start, end) in bounds.items()
    }


def _macro_metric(
    labels: pd.DataFrame,
    prediction: pd.DataFrame,
    index: pd.DatetimeIndex,
) -> dict[str, float]:
    metrics = {
        group: group_metrics(
            labels.loc[index, group].to_numpy(dtype=np.float64),
            prediction.loc[index, group].to_numpy(dtype=np.float64),
            CAPACITY_KWH[group],
            group_name=group,
        )
        for group in TARGET_COLS
    }
    n_value = float(np.mean([metric.one_minus_nmae for metric in metrics.values()]))
    f_value = float(np.mean([metric.ficr for metric in metrics.values()]))
    return {
        "score": 0.5 * (n_value + f_value),
        "one_minus_nmae": n_value,
        "ficr": f_value,
        "by_group": {group: metrics[group].as_dict() for group in TARGET_COLS},
        "rows": len(index),
    }


def replay_seven_veto_metrics(
    labels_2024: pd.DataFrame,
    base_scale097: pd.DataFrame,
    candidate: pd.DataFrame,
) -> dict[str, Any]:
    slices: dict[str, Any] = {}
    for name, index in _slice_indices_2024().items():
        baseline = _macro_metric(labels_2024, base_scale097, index)
        proposed = _macro_metric(labels_2024, candidate, index)
        slices[name] = {
            "base": baseline,
            "candidate": proposed,
            "delta": {
                "score": float(proposed["score"] - baseline["score"]),
                "one_minus_nmae": float(
                    proposed["one_minus_nmae"] - baseline["one_minus_nmae"]
                ),
                "ficr": float(proposed["ficr"] - baseline["ficr"]),
            },
        }
    gates = {
        "all_seven_score_positive": all(
            slices[name]["delta"]["score"] > 0.0 for name in SLICE_ORDER
        ),
        "full_n_nonnegative": (
            slices["FULL"]["delta"]["one_minus_nmae"] >= 0.0
        ),
        "full_f_positive": slices["FULL"]["delta"]["ficr"] > 0.0,
    }
    return {"slices": slices, "gates": gates, "passed": all(gates.values())}


def _read_2024_suffix_after_lock(
    label_path: Path,
    *,
    durable_lock_verified: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if not durable_lock_verified:
        raise AuditFailure("2024 suffix read attempted before durable prediction lock")
    _require_exact_label_file_size(label_path)
    with label_path.open("rb", buffering=0) as stream:
        if os.fstat(stream.fileno()).st_size != FULL_LABEL_BYTES:
            raise AuditFailure("label file fstat size changed before prefix range read")
        prefix = stream.read(PREFIX_BYTES)
        if stream.tell() != PREFIX_BYTES:
            raise AuditFailure("label prefix short read")
    with label_path.open("rb", buffering=0) as stream:
        if os.fstat(stream.fileno()).st_size != FULL_LABEL_BYTES:
            raise AuditFailure("label file fstat size changed before suffix range read")
        stream.seek(SUFFIX_OFFSET, os.SEEK_SET)
        suffix = stream.read(SUFFIX_BYTES)
        if stream.tell() != SUFFIX_OFFSET + SUFFIX_BYTES:
            raise AuditFailure("label suffix short read")
    if hashlib.sha256(prefix).hexdigest() != PREFIX_SHA256:
        raise AuditFailure("label prefix hash changed")
    if hashlib.sha256(suffix).hexdigest() != SUFFIX_SHA256:
        raise AuditFailure("label suffix hash changed")
    if hashlib.sha256(prefix + suffix).hexdigest() != FULL_LABEL_SHA256:
        raise AuditFailure("in-memory concatenated full label identity changed")
    prefix_frame = pd.read_csv(io.BytesIO(prefix), encoding="utf-8-sig")
    if tuple(prefix_frame.columns) != ("kst_dtm", *TARGET_COLS) or len(prefix_frame) != 17_520:
        raise AuditFailure("label prefix schema changed")
    prefix_frame.index = pd.DatetimeIndex(
        pd.to_datetime(prefix_frame.pop("kst_dtm"), errors="raise"),
        name="forecast_kst_dtm",
    )
    expected_prefix_index = pd.date_range(
        "2022-01-01 01:00:00",
        "2024-01-01 00:00:00",
        freq="h",
        name="forecast_kst_dtm",
    )
    if not prefix_frame.index.equals(expected_prefix_index):
        raise AuditFailure("label prefix index changed")
    header = prefix.splitlines(keepends=True)[0]
    frame = pd.read_csv(io.BytesIO(header + suffix), encoding="utf-8-sig")
    if tuple(frame.columns) != ("kst_dtm", *TARGET_COLS) or len(frame) != len(STRESS_INDEX):
        raise AuditFailure("2024 suffix schema changed")
    frame.index = pd.DatetimeIndex(
        pd.to_datetime(frame.pop("kst_dtm"), errors="raise"),
        name="forecast_kst_dtm",
    )
    if not frame.index.equals(STRESS_INDEX):
        raise AuditFailure("2024 suffix index changed")
    return prefix_frame.astype(np.float64), frame.astype(np.float64), {
        "prefix_bytes": len(prefix),
        "prefix_sha256": PREFIX_SHA256,
        "suffix_offset": SUFFIX_OFFSET,
        "suffix_bytes": len(suffix),
        "suffix_sha256": SUFFIX_SHA256,
        "full_identity_from_in_memory_concatenation": FULL_LABEL_SHA256,
        "exact_file_size_bytes": FULL_LABEL_BYTES,
        "path_stat_and_two_stream_fstats_exact": True,
        "independent_prefix_disk_reads": 1,
        "independent_suffix_disk_reads": 1,
        "full_file_stream_reads": 0,
    }


def _require_exact_label_file_size(label_path: Path) -> int:
    observed = label_path.stat().st_size
    if observed != FULL_LABEL_BYTES:
        raise AuditFailure(
            f"label file size changed/appended: {observed} != {FULL_LABEL_BYTES}"
        )
    return observed


def _eligible_identity_from_labels(
    labels: pd.DataFrame, *, group: str
) -> dict[str, Any]:
    capacity = float(CAPACITY_KWH[group])
    values = labels[group]
    mask = (
        values.notna().to_numpy()
        & np.isfinite(values.to_numpy(dtype=np.float64))
        & (values.to_numpy(dtype=np.float64) >= 0.10 * capacity)
    )
    index = labels.index[mask]
    target_cf = values.iloc[np.flatnonzero(mask)].to_numpy(dtype=np.float64) / capacity
    digest = hashlib.sha256()
    digest.update(np.asarray(index.asi8, dtype="<i8").tobytes())
    digest.update(np.asarray(target_cf, dtype="<f8").tobytes())
    return {"eligible_count": int(mask.sum()), "digest": digest.hexdigest()}


def _eligible_identities_from_labels(
    labels: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    return {
        group: _eligible_identity_from_labels(labels, group=group)
        for group in TARGET_COLS
    }


def validate_submission_csv(
    csv_path: Path,
    sample_path: Path,
    final_prediction: pd.DataFrame,
    *,
    expected_sample_sha256: str = SAMPLE_SHA256,
) -> dict[str, Any]:
    if sha256_file(sample_path) != expected_sample_sha256:
        raise AuditFailure("sample submission identity changed")
    raw = csv_path.read_bytes()
    checks: dict[str, bool] = {
        "utf8_bom": raw.startswith(b"\xef\xbb\xbf"),
        "lf_only": b"\r" not in raw,
    }
    sample = pd.read_csv(
        sample_path,
        encoding="utf-8-sig",
        dtype="string",
        keep_default_na=False,
    )
    candidate = pd.read_csv(
        io.BytesIO(raw),
        encoding="utf-8-sig",
        dtype="string",
        keep_default_na=False,
    )
    id_columns = [column for column in sample.columns if column not in TARGET_COLS]
    checks["sample_schema_and_identifier_order"] = (
        tuple(candidate.columns) == tuple(sample.columns)
        and len(candidate) == 8760
        and candidate[id_columns].equals(sample[id_columns])
    )
    six_decimal = re.compile(r"^\d+\.\d{6}$")
    checks["all_target_text_six_decimal"] = all(
        candidate[group].map(lambda value: bool(six_decimal.fullmatch(value))).all()
        for group in TARGET_COLS
    )
    numeric = candidate.loc[:, list(TARGET_COLS)].astype(np.float64)
    checks["finite_and_bounds"] = bool(
        np.isfinite(numeric.to_numpy(dtype=np.float64)).all()
        and all(
            (numeric[group] >= 0.0).all()
            and (numeric[group] <= 1.02 * CAPACITY_KWH[group]).all()
            for group in TARGET_COLS
        )
    )
    checks["six_decimal_matches_final_prediction"] = all(
        candidate[group].reset_index(drop=True).equals(
            pd.Series(
                [f"{value:.6f}" for value in final_prediction[group].to_numpy(dtype=np.float64)],
                dtype="string",
            )
        )
        for group in TARGET_COLS
    )
    expected = sample.copy()
    for group in TARGET_COLS:
        expected[group] = [
            f"{value:.6f}" for value in final_prediction[group].to_numpy(dtype=np.float64)
        ]
    buffer = io.StringIO(newline="")
    expected.to_csv(buffer, index=False, lineterminator="\n")
    expected_raw = b"\xef\xbb\xbf" + buffer.getvalue().encode("utf-8")
    checks["byte_exact_text_roundtrip"] = raw == expected_raw
    return {
        "checks": checks,
        "pass": all(checks.values()),
        "csv": file_record(csv_path),
        "sample": file_record(sample_path),
    }


def _model_fingerprint(model: Any) -> str:
    estimator = model.get("model") if isinstance(model, Mapping) else model
    booster = getattr(estimator, "booster_", None)
    if booster is not None and hasattr(booster, "model_to_string"):
        prefix = ""
        if isinstance(model, Mapping):
            prefix = json.dumps(
                {
                    "group": model.get("group"),
                    "kind": model.get("kind"),
                    "features": list(model.get("features", ())),
                    "feature_names_sha256": model.get("feature_names_sha256"),
                    "eligible_count": model.get("eligible_count"),
                    "eligible_index_target_digest": model.get(
                        "eligible_index_target_digest"
                    ),
                    "parameters": model.get("parameters"),
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
        payload = (prefix + "\n" + booster.model_to_string()).encode("utf-8")
    else:
        payload = pickle.dumps(model, protocol=5)
    return hashlib.sha256(payload).hexdigest()


def audit_double_reload_records(
    records: Sequence[Mapping[str, Any]],
    *,
    project_root: Path,
    stage: str,
    expected_eligible_identities: Mapping[str, Mapping[str, Any]],
    apply_features: Mapping[str, pd.DataFrame],
    expected_predictions: Mapping[str, pd.DataFrame],
) -> dict[str, Any]:
    if len(records) != 6:
        raise AuditFailure(f"{stage} must bind exactly six paired models")
    expected = {(group, variant) for group in TARGET_COLS for variant in PAIRED_VARIANTS}
    if set(apply_features) != set(TARGET_COLS) or set(expected_predictions) != set(
        PAIRED_VARIANTS
    ):
        raise AuditFailure(f"{stage} apply/prediction semantic maps differ")
    observed: set[tuple[str, str]] = set()
    details: list[dict[str, Any]] = []
    for record in records:
        group = str(record["group"])
        raw_variant = str(record.get("variant", record.get("kind", "")))
        variant = VARIANT_ALIASES.get(raw_variant, raw_variant)
        observed.add((group, variant))
        model_record = record.get(
            "model", record.get("model_file", record.get("file"))
        )
        if not isinstance(model_record, Mapping):
            raise AuditFailure(f"{stage} model snapshot missing for {group}/{variant}")
        model_path = verify_snapshot(model_record, root=project_root)
        first = joblib.load(model_path)
        second = joblib.load(model_path)
        first_hash = _model_fingerprint(first)
        second_hash = _model_fingerprint(second)
        expected_feature_count = FEATURE_COUNTS[variant]
        expected_feature_sha = (
            CONTROL_FEATURE_SHA256
            if variant == "control_all"
            else NONWIND_FEATURE_SHA256
        )
        expected_eligible_count = STAGE_ELIGIBLE_COUNTS[stage][group]
        independent_eligible = expected_eligible_identities.get(group, {})
        independent_digest = str(independent_eligible.get("digest", ""))
        if not isinstance(first, Mapping) or not isinstance(second, Mapping):
            raise AuditFailure(f"{stage} paired joblib payload schema changed")
        payload_variant = VARIANT_ALIASES.get(
            str(first.get("kind", first.get("variant", ""))), ""
        )
        features = list(map(str, first.get("features", ())))
        digest = str(first.get("eligible_index_target_digest", ""))
        payload_checks = {
            "group_variant": str(first.get("group")) == group
            and payload_variant == variant,
            "feature_count": len(features) == expected_feature_count,
            "feature_sha_recomputed": _ordered_names_sha256(features)
            == expected_feature_sha,
            "feature_sha_payload": first.get("feature_names_sha256")
            == expected_feature_sha,
            "eligible_count": int(first.get("eligible_count", -1))
            == expected_eligible_count,
            "eligible_count_independently_recomputed": int(
                independent_eligible.get("eligible_count", -1)
            )
            == expected_eligible_count,
            "eligible_digest_independently_recomputed": (
                bool(re.fullmatch(r"[0-9a-f]{64}", digest))
                and digest == independent_digest
            ),
            "parameters_payload": first.get("parameters") == MODEL_PARAMS,
            "record_feature_count": int(record.get("feature_count", -1))
            == expected_feature_count,
            "record_feature_sha": record.get("feature_names_sha256")
            == expected_feature_sha,
            "record_eligible_count": int(record.get("eligible_count", -1))
            == expected_eligible_count,
            "record_eligible_digest": record.get("eligible_index_target_digest")
            == digest,
            "record_parameters": record.get("parameters") == MODEL_PARAMS,
            "reloaded_payload_metadata_equal": all(
                first.get(key) == second.get(key)
                for key in (
                    "group",
                    "kind",
                    "features",
                    "feature_names_sha256",
                    "eligible_count",
                    "eligible_index_target_digest",
                    "parameters",
                )
            ),
        }
        estimator = first.get("model")
        observed_params = estimator.get_params() if hasattr(estimator, "get_params") else {}
        payload_checks["estimator_params_exact"] = all(
            observed_params.get(key) == value for key, value in MODEL_PARAMS.items()
        )
        if not all(payload_checks.values()):
            raise AuditFailure(
                f"{stage} model payload/record contract changed: "
                f"{group}/{variant} {payload_checks}"
            )
        second_estimator = second.get("model")
        apply = apply_features[group]
        expected_frame = expected_predictions[variant]
        if (
            not apply.index.equals(expected_frame.index)
            or tuple(expected_frame.columns) != TARGET_COLS
            or not set(features).issubset(apply.columns)
        ):
            raise AuditFailure(f"{stage} model apply/prediction alignment changed")
        reload1_prediction = np.asarray(
            estimator.predict(apply.loc[:, features]), dtype=np.float64
        )
        reload2_prediction = np.asarray(
            second_estimator.predict(apply.loc[:, features]), dtype=np.float64
        )
        expected_prediction = expected_frame[group].to_numpy(dtype=np.float64)
        if (
            not np.array_equal(reload1_prediction, expected_prediction)
            or not np.array_equal(reload2_prediction, expected_prediction)
        ):
            raise AuditFailure(
                f"{stage} reloaded model prediction differs from saved CF: "
                f"{group}/{variant}"
            )
        reload_claims = record.get("reload_checks", record)
        first_claim = bool(
            reload_claims.get(
                "in_memory_vs_reload1_array_equal",
                reload_claims.get("reload_1_equal", False),
            )
        )
        second_claim = bool(
            reload_claims.get(
                "reload1_vs_reload2_array_equal",
                reload_claims.get("reload_2_equal", False),
            )
        )
        if not first_claim or not second_claim:
            raise AuditFailure(f"{stage} saved prediction reload claim failed")
        if first_hash != second_hash:
            raise AuditFailure(f"{stage} independent double reload differs")
        details.append(
            {
                "group": group,
                "variant": variant,
                "feature_count": expected_feature_count,
                "feature_names_sha256": expected_feature_sha,
                "eligible_count": expected_eligible_count,
                "eligible_index_target_digest": digest,
                "parameters_exact": True,
                "reload1_reload2_saved_cf_bit_exact": True,
                "saved_cf_sha256": hashlib.sha256(
                    np.ascontiguousarray(expected_prediction, dtype="<f8").tobytes()
                ).hexdigest(),
                "model": file_record(model_path),
                "independent_reload1_fingerprint": first_hash,
                "independent_reload2_fingerprint": second_hash,
                "saved_prediction_reload_claims": {
                    "in_memory_vs_reload1_array_equal": True,
                    "reload1_vs_reload2_array_equal": True,
                },
            }
        )
    if observed != expected:
        raise AuditFailure(f"{stage} paired model identities changed: {observed}")
    return {"stage": stage, "model_count": 6, "pass": True, "models": details}


def audit_canonical_q07_replay(
    *,
    science_v1: Mapping[str, Any],
    project_root: Path,
    test_features: Mapping[str, pd.DataFrame],
    manifest_claim: Mapping[str, Any],
) -> dict[str, Any]:
    contract = science_v1["canonical_component_contract"]
    model_path = verify_snapshot(contract["canonical_q07_model"], root=project_root)
    prediction_path = verify_snapshot(
        contract["canonical_q07_prediction"], root=project_root
    )
    stored = _read_target_frame(prediction_path, expected_index=TEST_INDEX)
    payload = joblib.load(model_path)
    if not isinstance(payload, Mapping):
        raise AuditFailure("canonical q07 model payload schema changed")
    models = payload.get("models")
    feature_names = payload.get("feature_names")
    if (
        not isinstance(models, Mapping)
        or set(models) != set(TARGET_COLS)
        or not isinstance(feature_names, Mapping)
        or set(feature_names) != set(TARGET_COLS)
    ):
        raise AuditFailure("canonical q07 group model/features changed")
    replay: dict[str, dict[str, Any]] = {}
    for group in TARGET_COLS:
        names = list(map(str, feature_names[group]))
        apply = test_features[group]
        if not set(names).issubset(apply.columns):
            raise AuditFailure(f"canonical q07 features absent for {group}")
        predicted = np.asarray(
            models[group].predict(apply.loc[:, names]), dtype=np.float64
        )
        expected = stored[group].to_numpy(dtype=np.float64)
        equal = bool(np.array_equal(predicted, expected))
        max_abs = float(np.max(np.abs(predicted - expected)))
        replay[group] = {"array_equal": equal, "max_abs_kwh": max_abs}
        if not equal or max_abs != 0.0:
            raise AuditFailure(f"canonical q07 exact replay differs for {group}")
    _compare_nested_numbers(replay, manifest_claim, path="final.canonical_q07_replay")
    return {
        "model": file_record(model_path),
        "stored_component": file_record(prediction_path),
        "by_group": replay,
        "manifest_claim_exact": True,
    }


def _contract_stage(contract: Mapping[str, Any], stage: str) -> Mapping[str, Any]:
    stages = contract.get("stages")
    if not isinstance(stages, Mapping) or not isinstance(stages.get(stage), Mapping):
        raise AuditFailure(f"postrun contract lacks stages.{stage}")
    return stages[stage]


def _root_record(contract: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    files = contract.get("files", {})
    value = files.get(key) if isinstance(files, Mapping) else None
    if value is None:
        value = contract.get(key)
    if not isinstance(value, Mapping):
        raise AuditFailure(f"postrun contract lacks root semantic file {key}")
    return value


def _stage_record(stage: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    files = stage.get("files", stage)
    aliases = {
        "paired_control_cf": ("paired_control_cf", "control_cf"),
        "paired_exit_cf": ("paired_exit_cf", "paired_nonwind_cf", "exit_cf"),
        "q1_kwh": ("q1_kwh", "candidate_q07_kwh", "candidate_q07"),
        "A0": ("A0",),
        "A1_safe": ("A1_safe",),
        "D": (
            "D",
            "component_increment_kwh",
            "feature_exit_component_increment_kwh",
            "increment",
        ),
        "final_candidate": ("final_candidate", "candidate_scale097_kwh", "candidate"),
        "final_prediction": ("final_prediction", "final_prediction_kwh"),
        "prediction_lock": ("prediction_lock", "candidate_lock"),
        "candidate_durability_lock": ("candidate_durability_lock",),
        "A0_preflight": ("A0_preflight",),
        "score_label_access": ("score_label_access",),
        "stress_manifest": ("stress_manifest",),
        "promotion_lock": ("promotion_lock",),
        "rejection": ("rejection",),
        "base_scale097": ("base_scale097",),
        "final_csv": ("final_csv",),
        "sample": ("sample",),
        "results": ("results",),
    }
    value = None
    if isinstance(files, Mapping):
        for candidate in aliases.get(key, (key,)):
            if isinstance(files.get(candidate), Mapping):
                value = files[candidate]
                break
    if not isinstance(value, Mapping):
        raise AuditFailure(f"postrun contract lacks semantic file {key}")
    return value


def _optional_stage_record(
    stage: Mapping[str, Any], key: str
) -> Mapping[str, Any] | None:
    try:
        return _stage_record(stage, key)
    except AuditFailure:
        return None


def _effective_science_heads_exact(contract: Mapping[str, Any]) -> bool:
    expected = {spec["name"]: spec["sha256"] for spec in SCIENCE_HEADS}
    bound = contract.get("effective_science_head")
    if isinstance(bound, Mapping):
        observed: dict[str, str] = {}
        for name in expected:
            value = bound.get(name)
            if isinstance(value, Mapping):
                value = value.get("sha256")
            observed[name] = str(value)
        return observed == expected
    legacy = contract.get("effective_science_head_sha256")
    return legacy == SCIENCE_HEADS[2]["sha256"] or legacy == [
        spec["sha256"] for spec in SCIENCE_HEADS
    ]


def _stage_models(stage: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    models = stage.get("models")
    if isinstance(models, list):
        return [record for record in models if isinstance(record, Mapping)]
    if isinstance(models, Mapping):
        flattened: list[Mapping[str, Any]] = []
        for group, variants in models.items():
            if not isinstance(variants, Mapping):
                continue
            for variant, record in variants.items():
                if not isinstance(record, Mapping):
                    continue
                flattened.append(
                    {"group": group, "kind": variant, **dict(record)}
                )
        return flattened
    return []


def validate_v3_access_ledger(
    ledger: Mapping[str, Any],
    *,
    promoted: bool,
    source_lock: Mapping[str, Any] | None = None,
    score_label_access: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate the exact bounded-read order frozen by effective head v3."""

    events = ledger.get("events")
    if isinstance(events, list):
        expected_names = [
            "bounded_label_prefix_read",
            "source_lock_durable",
            "bounded_label_prefix_parsed_after_source_lock",
            "original_A0_reconstruction_verified_before_fit",
            "six_stress_models_double_reload_verified",
            "stress_candidate_lock_durable",
            "candidate_models_predictions_and_lock_reopened_rehashed",
            "label_suffix_single_read_and_in_memory_full_hash",
            "historical_veto_scored",
        ]
        if promoted:
            expected_names.extend(
                [
                    "promotion_and_stress_manifest_durable",
                    "conditional_official_test_raw_test_caches_and_final_inputs_verified_and_read",
                    "six_final_models_double_reload_verified",
                    "final_csv_byte_roundtrip_verified",
                ]
            )
        if len(events) != len(expected_names) or not all(
            isinstance(event, Mapping) for event in events
        ):
            raise AuditFailure("v3 access event count/schema differs")
        observed_sequence = [int(event.get("sequence", -1)) for event in events]
        observed_names = [str(event.get("event")) for event in events]
        event_checks = {
            "event_sequence_contiguous_and_exact": (
                observed_sequence == list(range(1, len(expected_names) + 1))
                and observed_names == expected_names
            ),
            "no_suffix_access_in_events_one_through_seven": all(
                int(events[index].get("suffix_reads_so_far", -1)) == 0
                for index in range(min(7, len(events)))
            ),
            "prefix_event_exact": (
                int(events[0].get("bytes", -1)) == PREFIX_BYTES
            ),
            "prefix_parse_after_source_lock_exact": (
                int(events[2].get("rows", -1)) == 17_520
            ),
            "stress_models_before_suffix_exact": (
                int(events[3].get("stress_model_fits_so_far", -1)) == 0
                and int(events[4].get("stress_model_fits", -1)) == 6
            ),
            "suffix_event_exact": (
                int(events[7].get("bytes", -1)) == SUFFIX_BYTES
                and int(events[7].get("suffix_reads_total", -1)) == 1
            ),
            "veto_event_branch_exact": (
                events[8].get("passed") is bool(promoted)
                and int(events[8].get("suffix_reads_total", -1)) == 1
            ),
            "flat_counters_exact": (
                int(ledger.get("prefix_range_disk_reads", -1)) == 1
                and int(ledger.get("suffix_range_disk_reads", -1)) == 1
                and int(ledger.get("whole_file_single_stream_reads", -1)) == 0
                and ledger.get(
                    "full_file_identity_computed_from_two_in_memory_ranges"
                )
                is True
                and int(ledger.get("stress_model_fits", -1)) == 6
                and int(ledger.get("final_model_fits", -1))
                == (6 if promoted else 0)
                and int(ledger.get("test_cache_files_opened", -1))
                == (3 if promoted else 0)
                and int(ledger.get("final_component_files_opened", -1))
                == (6 if promoted else 0)
                and int(ledger.get("sample_files_opened", -1))
                == (1 if promoted else 0)
                and int(ledger.get("csv_files_written", -1))
                == (1 if promoted else 0)
                and ledger.get("suffix_buffer_reused_for_final_fit")
                is bool(promoted)
            ),
        }
        if promoted:
            event_checks["postpromotion_events_keep_single_suffix_read"] = all(
                int(event.get("suffix_reads_total", -1)) == 1
                for event in events[9:]
            )
            event_checks["six_final_models_exact"] = (
                int(events[11].get("final_model_fits", -1)) == 6
            )
        if not isinstance(source_lock, Mapping) or not isinstance(
            score_label_access, Mapping
        ):
            raise AuditFailure("flat access ledger needs source/score access locks")
        prefix_lock = source_lock.get("label_prefix", {})
        event_checks["source_lock_prefix_and_deferred_identity_exact"] = (
            isinstance(prefix_lock, Mapping)
            and int(prefix_lock.get("bytes", -1)) == PREFIX_BYTES
            and str(prefix_lock.get("sha256")) == PREFIX_SHA256
            and int(prefix_lock.get("suffix_bytes_read", -1)) == 0
            and str(source_lock.get("deferred_full_label_sha"))
            == FULL_LABEL_SHA256
        )
        event_checks["score_suffix_lock_exact"] = (
            int(score_label_access.get("bytes", -1)) == SUFFIX_BYTES
            and str(score_label_access.get("sha256")) == SUFFIX_SHA256
            and int(score_label_access.get("disk_reads", -1)) == 1
            and int(score_label_access.get("rows", -1)) == len(STRESS_INDEX)
            and str(score_label_access.get("full_sha_from_memory"))
            == FULL_LABEL_SHA256
            and int(score_label_access.get("physical_file_size_bytes", -1))
            == FULL_LABEL_BYTES
            and score_label_access.get("full_file_size_exact") is True
        )
        failures = [name for name, passed in event_checks.items() if not passed]
        if failures:
            raise AuditFailure(f"v3 flat access ledger failed: {failures}")
        return {
            "schema": "runner_flat_ordered_events_v1",
            "checks": event_checks,
            "pass": True,
        }

    prefix = ledger.get("prefix")
    suffix = ledger.get("suffix")
    full_hash = ledger.get("full_hash")
    stress = ledger.get("stress")
    final = ledger.get("final")
    if not all(
        isinstance(section, Mapping)
        for section in (prefix, suffix, full_hash, stress, final)
    ):
        raise AuditFailure("v3 access ledger lacks prefix/suffix/full/stress/final")
    assert isinstance(prefix, Mapping)
    assert isinstance(suffix, Mapping)
    assert isinstance(full_hash, Mapping)
    assert isinstance(stress, Mapping)
    assert isinstance(final, Mapping)

    checks = {
        "prefix_one_exact_bounded_disk_read": (
            int(prefix.get("disk_reads", -1)) == 1
            and int(prefix.get("offset", -1)) == 0
            and int(prefix.get("bytes", -1)) == PREFIX_BYTES
            and str(prefix.get("sha256")) == PREFIX_SHA256
        ),
        "prefix_not_parsed_before_source_lock": (
            prefix.get("parsed_before_source_lock") is False
        ),
        "suffix_one_exact_disk_read_after_prediction_lock": (
            int(suffix.get("disk_reads", -1)) == 1
            and int(suffix.get("offset", -1)) == SUFFIX_OFFSET
            and int(suffix.get("bytes", -1)) == SUFFIX_BYTES
            and str(suffix.get("sha256")) == SUFFIX_SHA256
            and suffix.get("read_after_prediction_lock") is True
        ),
        "suffix_parsed_once_and_never_reread": (
            int(suffix.get("parse_count", -1)) == 1
            and int(suffix.get("disk_rereads", -1)) == 0
        ),
        "retained_suffix_reuse_matches_branch": (
            bool(suffix.get("reused_for_final_fit", False)) is bool(promoted)
        ),
        "full_hash_from_retained_buffers_only": (
            full_hash.get("computed_from_retained_buffers") is True
            and str(full_hash.get("sha256")) == FULL_LABEL_SHA256
        ),
        "six_stress_fits_and_durable_lock_before_suffix": (
            int(stress.get("model_fits", -1)) == 6
            and stress.get("prediction_lock_durable_before_suffix") is True
        ),
        "conditional_final_access_counts_exact": (
            final.get("promotion_verified") is True
            and int(final.get("model_fits", -1)) == (6 if promoted else 0)
            and int(final.get("test_cache_array_reads", -1))
            == (3 if promoted else 0)
            and int(final.get("final_component_array_reads", -1))
            == (6 if promoted else 0)
            and int(final.get("csv_files", -1)) == (1 if promoted else 0)
        ),
    }
    fail_branch = ledger.get("fail_branch_zero_read_assertions")
    if isinstance(fail_branch, Mapping):
        checks["fail_branch_zero_read_assertions"] = bool(fail_branch) and all(
            value is True for value in fail_branch.values()
        )
    else:
        checks["fail_branch_zero_read_assertions"] = fail_branch is True
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise AuditFailure(f"v3 access ledger failed: {failures}")
    return {"checks": checks, "pass": True}


def _load_recipe(path: Path) -> Mapping[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))["ensemble"]


def _component_paths_from_science(
    v1: Mapping[str, Any],
    *,
    project_root: Path,
    stage: str,
) -> tuple[dict[str, Path], Path, Path]:
    if stage == "stress":
        section = v1["historical_veto_2024"]
        specs = section["components_2024"]
        reference_spec = {
            "path": "artifacts/gate/v3/gate_oof.parquet",
            "bytes": 314_530,
            "sha256": ORIGINAL_GATE_A0_SHA256,
        }
        recipe_spec = {
            "path": "artifacts/gate/v3/gate_recipe_snapshot.json",
            "bytes": 5_613,
            "sha256": "c26e0b0b4268ed3780d85e1c3af5db38253d42e8a4230fbd910691f2ec6acd19",
        }
    else:
        section = v1["locked_assembly_contract"]
        specs = section["components_2025"]
        reference_spec = section["assembled_reference_2025"]
        recipe_spec = v1["canonical_component_contract"]["actual_generation_recipe_snapshot"]
    components = {
        name: verify_snapshot(specs[name], root=project_root)
        for name in COMPONENT_NAMES
    }
    reference = verify_snapshot(reference_spec, root=project_root)
    recipe = verify_snapshot(recipe_spec, root=project_root)
    return components, reference, recipe


def _audit_stage_formula(
    *,
    stage_name: str,
    stage_contract: Mapping[str, Any],
    project_root: Path,
    science_v1: Mapping[str, Any],
    expected_index: pd.DatetimeIndex,
) -> tuple[dict[str, Any], AssemblyReplay, pd.DataFrame, pd.DataFrame]:
    paired_record = _optional_stage_record(stage_contract, "paired_cf")
    if paired_record is not None:
        paired_path = verify_snapshot(paired_record, root=project_root)
        paired_control, paired_exit = _read_paired_cf(
            paired_path, expected_index=expected_index
        )
    else:
        paired_control = _read_target_frame(
            verify_snapshot(
                _stage_record(stage_contract, "paired_control_cf"),
                root=project_root,
            ),
            expected_index=expected_index,
        )
        paired_exit = _read_target_frame(
            verify_snapshot(
                _stage_record(stage_contract, "paired_exit_cf"),
                root=project_root,
            ),
            expected_index=expected_index,
        )
    saved_q1 = _read_target_frame(
        verify_snapshot(_stage_record(stage_contract, "q1_kwh"), root=project_root),
        expected_index=expected_index,
    )
    saved_optional: dict[str, pd.DataFrame] = {}
    for name in ("A0", "A1_safe"):
        record = _optional_stage_record(stage_contract, name)
        if record is not None:
            saved_optional[name] = _read_target_frame(
                verify_snapshot(record, root=project_root),
                expected_index=expected_index,
            )
    saved_d = _read_target_frame(
        verify_snapshot(_stage_record(stage_contract, "D"), root=project_root),
        expected_index=expected_index,
    )
    component_paths, a0_reference_path, recipe_path = _component_paths_from_science(
        science_v1,
        project_root=project_root,
        stage=stage_name,
    )
    components = {
        name: _read_target_frame(path, expected_index=expected_index)
        for name, path in component_paths.items()
    }
    cap = cap_q07_component(
        components["lgb_q07"], paired_control, paired_exit, cap_cf=CAP_CF
    )
    control_vs_canonical_report_only: dict[str, dict[str, Any]] = {}
    for group in TARGET_COLS:
        control_vs_canonical = (
            float(CAPACITY_KWH[group])
            * paired_control[group].to_numpy(dtype=np.float64)
            - components["lgb_q07"][group].to_numpy(dtype=np.float64)
        )
        control_vs_canonical_report_only[group] = {
            "mean_signed_kwh": float(np.mean(control_vs_canonical)),
            "mean_abs_kwh": float(np.mean(np.abs(control_vs_canonical))),
            "max_abs_kwh": float(np.max(np.abs(control_vs_canonical))),
            "used_by_gate_abort_formula_or_selection": False,
        }
    if not exact_frame_equal(saved_q1, cap.q1_kwh):
        raise AuditFailure(f"{stage_name} q1 does not replay cap exactly once")
    assembly = replay_locked_assembly(
        components, cap.q1_kwh, ensemble=_load_recipe(recipe_path)
    )
    reference = _read_target_frame(a0_reference_path, expected_index=expected_index)
    a0_max_abs_by_group = {
        group: float(
            np.max(
                np.abs(
                    assembly.a0_kwh[group].to_numpy(dtype=np.float64)
                    - reference[group].to_numpy(dtype=np.float64)
                )
            )
        )
        for group in TARGET_COLS
    }
    max_abs_a0 = float(
        max(a0_max_abs_by_group.values())
    )
    six_decimal_a0 = bool(
        np.array_equal(
            np.round(assembly.a0_kwh.to_numpy(), 6),
            np.round(reference.to_numpy(), 6),
        )
    )
    if max_abs_a0 > 1e-9 or not six_decimal_a0:
        raise AuditFailure(f"{stage_name} original A0 reconstruction changed")
    assembly_diagnostics: dict[str, dict[str, Any]] = {}
    for group in TARGET_COLS:
        raw_values = cap.raw_delta_cf[group].to_numpy(dtype=np.float64)
        capped_values = cap.capped_delta_cf[group].to_numpy(dtype=np.float64)
        assembly_diagnostics[group] = {
            "cap_hits": cap.cap_hit_count[group],
            "max_abs_raw_delta_cf": float(np.max(np.abs(raw_values))),
            "max_abs_capped_delta_cf": float(np.max(np.abs(capped_values))),
            "hypothetical_g2_bin_crossings": (
                assembly.g2_hypothetical_crossing_count
                if group == "kpx_group_2"
                else 0
            ),
            "A0_reference_max_abs_kwh": a0_max_abs_by_group[group],
            "A0_six_decimal_exact": bool(
                np.array_equal(
                    np.round(assembly.a0_kwh[group].to_numpy(dtype=np.float64), 6),
                    np.round(reference[group].to_numpy(dtype=np.float64), 6),
                )
            ),
            "new_cf_control_vs_canonical_kwh_q07_report_only": (
                control_vs_canonical_report_only[group]
            ),
        }
    comparisons = [
        ("D", saved_d, assembly.increment_d_kwh),
        *(
            [("A0", saved_optional["A0"], assembly.a0_kwh)]
            if "A0" in saved_optional
            else []
        ),
        *(
            [("A1_safe", saved_optional["A1_safe"], assembly.a1_safe_kwh)]
            if "A1_safe" in saved_optional
            else []
        ),
    ]
    for label, actual, expected in comparisons:
        if not exact_frame_equal(actual, expected):
            raise AuditFailure(f"{stage_name} saved {label} formula replay failed")
    return {
        "stage": stage_name,
        "cap_applied_once_q1_bit_exact": True,
        "cap_hit_count": cap.cap_hit_count,
        "q1_outer_clip_count": cap.q1_outer_clip_count,
        "original_A0_max_abs_kwh": max_abs_a0,
        "original_A0_max_abs_kwh_by_group": a0_max_abs_by_group,
        "original_A0_six_decimal_exact": six_decimal_a0,
        "saved_formula_arrays_bit_exact": [label for label, _, _ in comparisons],
        "g2_b0_sha256": hashlib.sha256(
            np.ascontiguousarray(assembly.g2_b0, dtype="<i8").tobytes()
        ).hexdigest(),
        "g2_hypothetical_b1_sha256": hashlib.sha256(
            np.ascontiguousarray(assembly.g2_hypothetical_b1, dtype="<i8").tobytes()
        ).hexdigest(),
        "g2_hypothetical_crossing_count_report_only": (
            assembly.g2_hypothetical_crossing_count
        ),
        "new_cf_control_vs_canonical_kwh_q07_report_only": (
            control_vs_canonical_report_only
        ),
        "runner_assembly_diagnostics_exact_shape": assembly_diagnostics,
    }, assembly, paired_control, paired_exit


def _compare_nested_numbers(left: Any, right: Any, *, path: str = "root") -> None:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        if set(left) != set(right):
            raise AuditFailure(f"metric keys differ at {path}")
        for key in left:
            _compare_nested_numbers(left[key], right[key], path=f"{path}.{key}")
        return
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            raise AuditFailure(f"metric list length differs at {path}")
        for index, (a, b) in enumerate(zip(left, right, strict=True)):
            _compare_nested_numbers(a, b, path=f"{path}[{index}]")
        return
    if isinstance(left, (int, float, bool)) and isinstance(right, (int, float, bool)):
        if isinstance(left, bool) or isinstance(right, bool):
            if bool(left) != bool(right):
                raise AuditFailure(f"metric boolean differs at {path}")
        elif float(left) != float(right):
            raise AuditFailure(f"metric float differs at {path}: {left!r}/{right!r}")
        return
    if left != right:
        raise AuditFailure(f"metric value differs at {path}")


def _manifest_records(manifest: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    for key in ("outputs", "files", "outputs_excluding_manifest"):
        value = manifest.get(key)
        if isinstance(value, list):
            return [record for record in value if isinstance(record, Mapping)]
    raise AuditFailure("root manifest has no output snapshot list")


def _require_bit_exact_copy(copied: Path, canonical: Path) -> None:
    if not canonical.is_file() or copied.read_bytes() != canonical.read_bytes():
        raise AuditFailure(f"copied/canonical execution lineage differs: {canonical}")


def _require_review_binding(
    review: Mapping[str, Any], *, amendment_sha: str, runner_sha: str
) -> None:
    if (
        review.get("experiment_id")
        != "feature_exit_nonwind_cap025143_scale097_overlay_v1"
        or review.get("verdict") != "PASS"
        or int(review.get("blocking_defects", -1)) != 0
        or review.get("execution_amendment", {}).get("sha256") != amendment_sha
        or review.get("bound_runner", {}).get("sha256") != runner_sha
    ):
        raise AuditFailure("independent review does not bind amendment/runner")


def _audit_attempt_closure(
    source_record: Mapping[str, Any],
    root_record: Mapping[str, Any],
    *,
    canonical_attempt: Path,
    project_root: Path,
) -> dict[str, Any]:
    if source_record != root_record:
        raise AuditFailure("source/root attempt-lock snapshots differ")
    path = verify_snapshot(source_record, root=project_root)
    if path != canonical_attempt.resolve():
        raise AuditFailure("source-lock attempt snapshot path is noncanonical")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("experiment_id")
        != "feature_exit_nonwind_cap025143_scale097_overlay_v1"
        or payload.get("single_attempt") is not True
    ):
        raise AuditFailure("canonical attempt-lock payload differs")
    return file_record(path)


def _audit_root_manifest_semantics(
    manifest: Mapping[str, Any],
    *,
    stored_results: Mapping[str, Any],
    promoted: bool,
    decision_payload: Mapping[str, Any],
) -> dict[str, Any]:
    expected_verdict = (
        "PASS_PROTOCOL_AND_HISTORICAL_VETO_CSV_READY"
        if promoted
        else "PASS_PROTOCOL_PERFORMANCE_REJECT"
    )
    checks = {
        "artifact_type": manifest.get("artifact_type")
        == "feature_exit_nonwind_cap025143_scale097_overlay_v1",
        "branch_verdict": manifest.get("verdict") == expected_verdict,
        "embedded_stress_exact": manifest.get("stress") == stored_results,
    }
    if promoted:
        checks["embedded_promotion_exact"] = manifest.get("promotion") == decision_payload
        checks["final_present"] = isinstance(manifest.get("final"), Mapping)
        checks["reject_only_final_access_absent"] = "final_access" not in manifest
    else:
        checks["promotion_absent"] = "promotion" not in manifest
        checks["final_absent"] = "final" not in manifest
        checks["reject_final_access_all_zero"] = manifest.get("final_access") == {
            "test_caches": 0,
            "final_components": 0,
            "sample": 0,
            "models": 0,
            "csv": 0,
        }
    if not all(checks.values()):
        raise AuditFailure(f"root manifest semantic branch differs: {checks}")
    return {"checks": checks, "verdict": expected_verdict, "pass": True}


def _audit_execution_lineage(
    *,
    source_lock: Mapping[str, Any],
    root_manifest: Mapping[str, Any],
    project_root: Path,
    output_root: Path,
    science_v1: Mapping[str, Any],
) -> dict[str, Any]:
    lineage = source_lock.get("execution_lineage")
    if not isinstance(lineage, Mapping) or set(lineage) != {
        "execution_amendment",
        "execution_amendment_sidecar",
        "independent_prelaunch_review",
    }:
        raise AuditFailure("source lock execution-lineage schema differs")
    if root_manifest.get("execution_lineage") != lineage:
        raise AuditFailure("source/root execution-lineage snapshots differ")
    resolved: dict[str, Path] = {}
    for key, record in lineage.items():
        if not isinstance(record, Mapping):
            raise AuditFailure(f"execution-lineage snapshot differs: {key}")
        path = verify_snapshot(record, root=project_root)
        try:
            path.relative_to(output_root / "provenance")
        except ValueError as exc:
            raise AuditFailure("execution lineage escapes output provenance") from exc
        resolved[key] = path
    amendment_path = resolved["execution_amendment"]
    amendment_sha = sha256_file(amendment_path)
    canonical_amendment = (
        project_root
        / "configs/feature_exit_nonwind_cap025143_scale097_overlay_execution_v4.json"
    ).resolve()
    canonical_sidecar = canonical_amendment.with_suffix(
        canonical_amendment.suffix + ".sha256"
    )
    canonical_review = (
        project_root
        / "artifacts/audits/feature_exit_nonwind_cap025143_scale097_overlay_v1_prelaunch_independent_review.json"
    ).resolve()
    for copied, canonical in (
        (amendment_path, canonical_amendment),
        (resolved["execution_amendment_sidecar"], canonical_sidecar),
        (resolved["independent_prelaunch_review"], canonical_review),
    ):
        _require_bit_exact_copy(copied, canonical)
    expected_sidecar = f"{amendment_sha}  {amendment_path.name}\n"
    if resolved["execution_amendment_sidecar"].read_text(encoding="utf-8") != expected_sidecar:
        raise AuditFailure("copied execution-amendment sidecar differs")
    amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
    if (
        amendment.get("schema_version") != 1
        or amendment.get("experiment_id")
        != "feature_exit_nonwind_cap025143_scale097_overlay_v1"
        or amendment.get("status") != "AUTHORIZED_SINGLE_EXECUTION"
        or amendment.get("single_attempt") is not True
        or Path(str(amendment.get("canonical_output"))).resolve() != output_root
    ):
        raise AuditFailure("execution-amendment authorization identity differs")
    science = amendment.get("science_heads")
    expected_science = [
        {
            "role": f"science_v{position}",
            "path": str((project_root / spec["relative"]).resolve()),
            "size_bytes": int(spec["bytes"]),
            "sha256": spec["sha256"],
        }
        for position, spec in enumerate(SCIENCE_HEADS, start=1)
    ]
    if science != expected_science:
        raise AuditFailure("execution amendment science heads differ")

    def declared(role: str, record: Mapping[str, Any]) -> dict[str, Any]:
        raw = Path(str(record["path"]))
        path = raw if raw.is_absolute() else project_root / raw
        return {
            "role": role,
            "path": str(path.resolve()),
            "size_bytes": _record_size(record),
            "sha256": str(record["sha256"]),
        }

    data = science_v1["data_contract"]
    historical = science_v1["historical_veto_2024"]
    locked = science_v1["locked_assembly_contract"]
    canonical = science_v1["canonical_component_contract"]
    expected_data: list[dict[str, Any]] = [
        declared("train_labels", data["labels"]),
        declared("sample_submission", data["sample_submission"]),
        declared(
            "info_xlsx",
            {
                "path": "data/local/open/info.xlsx",
                "bytes": 3_823_422,
                "sha256": "89e83a52e0eb2ce367a3573a96d6795ed4b4d4ac624965cb3530beec0cbd2bd6",
            },
        ),
        declared(
            "gate_recipe",
            {
                "path": "artifacts/gate/v3/gate_recipe_snapshot.json",
                "bytes": 5_613,
                "sha256": "c26e0b0b4268ed3780d85e1c3af5db38253d42e8a4230fbd910691f2ec6acd19",
            },
        ),
        declared("final_recipe", canonical["actual_generation_recipe_snapshot"]),
        declared(
            "gate_reference",
            {
                "path": "artifacts/gate/v3/gate_oof.parquet",
                "bytes": 314_530,
                "sha256": ORIGINAL_GATE_A0_SHA256,
            },
        ),
        declared("stress_scale097_base", historical["full_precision_scale097_base"]),
        declared("final_scale097_base", science_v1["final_base_contract"]["prediction"]),
        declared("final_locked_reference", locked["assembled_reference_2025"]),
        declared("final_canonical_q07_model", canonical["canonical_q07_model"]),
    ]
    for prefix, specs in (
        (
            "official_train_raw",
            (
                {
                    "path": "data/local/open/train/ldaps_train.csv",
                    "bytes": 129_687_357,
                    "sha256": "61ae944e7ae1fcb17391be6737792a2205c6507bf2446ed5d9d0daf07fdea026",
                },
                {
                    "path": "data/local/open/train/gfs_train.csv",
                    "bytes": 84_315_594,
                    "sha256": "cd56b67d357e7bbaff5d0d51d3537d935c9e7a3f012e9f37516bdc4d38c66a5d",
                },
            ),
        ),
        (
            "official_test_raw",
            (
                {
                    "path": "data/local/open/test/ldaps_test.csv",
                    "bytes": 43_122_637,
                    "sha256": "60e94f7cc80384eee335e90dc896b6cf4d36b35cde8d37bc03bd5c08a788b0fa",
                },
                {
                    "path": "data/local/open/test/gfs_test.csv",
                    "bytes": 28_037_722,
                    "sha256": "aa33febb24ecd46b82be34880a06910e16a3382319287548e6ce2af721b4f848",
                },
            ),
        ),
    ):
        expected_data.extend(
            declared(f"{prefix}_{position}", spec)
            for position, spec in enumerate(specs, start=1)
        )
    for prefix, specs in (
        ("train_cache", data["train_caches"]),
        ("test_cache", data["test_caches"]),
        ("gate_component", historical["components_2024"]),
        ("final_component", locked["components_2025"]),
    ):
        expected_data.extend(
            declared(f"{prefix}_{name}", spec) for name, spec in specs.items()
        )
    if amendment.get("data_declarations") != expected_data:
        raise AuditFailure("execution amendment exact data declarations differ")
    bound_files = amendment.get("bound_files")
    if not isinstance(bound_files, list) or not bound_files:
        raise AuditFailure("execution amendment lacks bound files")
    by_role = {
        str(record.get("role")): record
        for record in bound_files
        if isinstance(record, Mapping)
    }
    if len(by_role) != len(bound_files):
        raise AuditFailure("execution amendment bound-file roles duplicate")
    required_paths = {
        "runner": project_root
        / "scripts/run_feature_exit_nonwind_cap025143_scale097_overlay.py",
        "independent_postrun_auditor": Path(__file__).resolve(),
        "focused_runner_tests": project_root
        / "tests/test_feature_exit_nonwind_cap025143_scale097_overlay.py",
        "independent_auditor_tests": project_root
        / "tests/test_feature_exit_nonwind_cap025143_scale097_overlay_auditor.py",
        "root_contract_tests": project_root
        / "tests/test_feature_exit_nonwind_cap025143_scale097_overlay_root.py",
        "src_metric": project_root / "src/metric.py",
        "src_virtual_feature_exit": project_root / "src/virtual_feature_exit.py",
        "src_density_ratio": project_root / "src/density_ratio.py",
        "src_manifest": project_root / "src/manifest.py",
        "src_features": project_root / "src/features.py",
        "build_features": project_root / "scripts/build_features.py",
        "info_xlsx": Path("data/local/open/info.xlsx"),
    }
    verified_bound_files: list[dict[str, Any]] = []
    for role, record in by_role.items():
        path = verify_snapshot(record, root=project_root)
        if role in required_paths and path != required_paths[role].resolve():
            raise AuditFailure(f"execution amendment canonical role path differs: {role}")
        verified_bound_files.append({"role": role, **file_record(path)})
    if set(required_paths) != set(by_role):
        raise AuditFailure("execution amendment exact bound-role set differs")
    runner_sha = sha256_file(required_paths["runner"])
    if by_role["runner"].get("sha256") != runner_sha:
        raise AuditFailure("execution amendment bound_runner_sha differs")
    expected_runtime = {
        "python": platform.python_version(),
        "packages": {
            package: importlib_metadata.version(package)
            for package in (
                "numpy",
                "pandas",
                "scikit-learn",
                "lightgbm",
                "pyarrow",
                "joblib",
            )
        },
    }
    if amendment.get("runtime") != expected_runtime:
        raise AuditFailure("execution amendment runtime lock differs")
    if amendment.get("prelaunch_zero_state") != {
        "canonical_output_absent": True,
        "attempt_absent": True,
        "heavy_guard_absent": True,
    }:
        raise AuditFailure("execution amendment prelaunch zero-state differs")
    if Path(str(amendment.get("required_independent_prelaunch_review"))).resolve() != canonical_review:
        raise AuditFailure("execution amendment review path differs")
    review = json.loads(
        resolved["independent_prelaunch_review"].read_text(encoding="utf-8")
    )
    _require_review_binding(
        review, amendment_sha=amendment_sha, runner_sha=runner_sha
    )
    attempt_record = source_lock.get("attempt_lock")
    if not isinstance(attempt_record, Mapping):
        raise AuditFailure("source lock lacks attempt-lock snapshot")
    canonical_attempt = (
        project_root
        / "artifacts/locks/feature_exit_nonwind_cap025143_scale097_overlay_v1.attempt.json"
    ).resolve()
    attempt_file = _audit_attempt_closure(
        attempt_record,
        root_manifest.get("attempt_lock"),
        canonical_attempt=canonical_attempt,
        project_root=project_root,
    )
    return {
        "execution_amendment": file_record(amendment_path),
        "execution_amendment_sidecar": file_record(
            resolved["execution_amendment_sidecar"]
        ),
        "independent_prelaunch_review": file_record(
            resolved["independent_prelaunch_review"]
        ),
        "attempt_lock": attempt_file,
        "bound_runner_sha256": runner_sha,
        "verified_bound_files": verified_bound_files,
        "pass": True,
    }


def _manifest_lookup(
    records: Sequence[Mapping[str, Any]], *, output_root: Path
) -> dict[str, Mapping[str, Any]]:
    lookup: dict[str, Mapping[str, Any]] = {}
    for record in records:
        raw = Path(str(record.get("path", "")))
        path = raw if raw.is_absolute() else output_root / raw
        path = path.resolve()
        try:
            relative = path.relative_to(output_root).as_posix()
        except ValueError as exc:
            raise AuditFailure(f"manifest output escapes output root: {path}") from exc
        if relative in lookup:
            raise AuditFailure(f"duplicate manifest output record: {relative}")
        lookup[relative] = record
    return lookup


def _build_contract_from_actual_layout(
    *,
    output_root: Path,
    manifest_path: Path,
    manifest: Mapping[str, Any],
    manifest_records: Sequence[Mapping[str, Any]],
    science_v1: Mapping[str, Any],
) -> dict[str, Any]:
    """Map the runner's immutable locks/manifests into semantic audit names."""

    lookup = _manifest_lookup(manifest_records, output_root=output_root)

    def record(relative: str) -> Mapping[str, Any]:
        value = lookup.get(relative)
        if not isinstance(value, Mapping):
            raise AuditFailure(f"manifest lacks runner output {relative}")
        return value

    candidate_lock_record = record("stress/candidate_lock.json")
    candidate_lock_path = verify_snapshot(candidate_lock_record, root=output_root)
    candidate_lock = json.loads(candidate_lock_path.read_text(encoding="utf-8"))
    outputs = candidate_lock.get("outputs")
    models = candidate_lock.get("models")
    if not isinstance(outputs, Mapping) or not isinstance(models, Mapping):
        raise AuditFailure("stress candidate lock lacks outputs/models")
    results_record = record("stress/results.json")
    results = json.loads(
        verify_snapshot(results_record, root=output_root).read_text(encoding="utf-8")
    )
    promoted = results.get("passed") is True
    stress_files: dict[str, Any] = {
        "paired_control_cf": outputs.get("control_cf"),
        "paired_nonwind_cf": outputs.get("exit_cf"),
        "candidate_q07_kwh": outputs.get("candidate_q07"),
        "A0": outputs.get("A0"),
        "A1_safe": outputs.get("A1_safe"),
        "component_increment_kwh": outputs.get("increment"),
        "candidate_scale097_kwh": outputs.get("candidate"),
        "base_scale097": science_v1["historical_veto_2024"][
            "full_precision_scale097_base"
        ],
        "candidate_lock": candidate_lock_record,
        "A0_preflight": record("stress/A0_reconstruction_preflight.json"),
        "candidate_durability_lock": record(
            "stress/candidate_durability_lock.json"
        ),
        "score_label_access": record("stress/score_label_access.json"),
        "results": results_record,
    }
    if promoted:
        stress_files["promotion_lock"] = record("stress/promotion_lock.json")
    else:
        stress_files["rejection"] = record("stress/rejection.json")
    stress_files["stress_manifest"] = record("stress/manifest.json")
    if not all(isinstance(value, Mapping) for value in stress_files.values()):
        raise AuditFailure("stress candidate lock has incomplete semantic snapshots")

    stages: dict[str, Any] = {
        "stress": {
            "status": "promoted" if promoted else "rejected",
            "models": models,
            "files": stress_files,
        }
    }
    if promoted:
        final_section = manifest.get("final")
        if not isinstance(final_section, Mapping) or not isinstance(
            final_section.get("models"), Mapping
        ):
            raise AuditFailure("promoted root manifest lacks final models")
        stages["final"] = {
            "status": "complete",
            "models": final_section["models"],
            "files": {
                "paired_control_cf": record(
                    "final/predictions/paired_control_cf_2025.parquet"
                ),
                "paired_nonwind_cf": record(
                    "final/predictions/paired_nonwind_cf_2025.parquet"
                ),
                "candidate_q07_kwh": record(
                    "final/predictions/candidate_q07_kwh_2025.parquet"
                ),
                "A0": record(
                    "final/predictions/A0_locked_v3_kwh_2025.parquet"
                ),
                "A1_safe": record(
                    "final/predictions/A1_safe_locked_v3_kwh_2025.parquet"
                ),
                "feature_exit_component_increment_kwh": record(
                    "final/predictions/feature_exit_component_increment_kwh_2025.parquet"
                ),
                "final_prediction_kwh": record(
                    "final/predictions/final_prediction_kwh_2025.parquet"
                ),
                "base_scale097": science_v1["final_base_contract"]["prediction"],
                "sample": science_v1["data_contract"]["sample_submission"],
                "final_csv": record(
                    "feature_exit_nonwind_cap025143_scale097_overlay_2025.csv"
                ),
            },
        }
    else:
        stages["final"] = {"status": "not_run"}

    return {
        "schema_version": 1,
        "semantic_contract_origin": "runner_locks_and_hash_manifest",
        "effective_science_head": {
            spec["name"]: spec["sha256"] for spec in SCIENCE_HEADS
        },
        "files": {
            "source_lock": record("source_lock.json"),
            "access_ledger": record("access_ledger.json"),
            "root_manifest": file_record(manifest_path),
        },
        "stages": stages,
        "actual_layout": True,
    }


def _audit_stress_manifest(
    stage: Mapping[str, Any],
    *,
    expected_verdict: str,
    project_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    path = verify_snapshot(_stage_record(stage, "stress_manifest"), root=project_root)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("verdict") != expected_verdict:
        raise AuditFailure("durable stress manifest verdict differs")
    records = _manifest_records(payload)
    lookup = _manifest_lookup(records, output_root=output_root / "stress")
    registered = {(output_root / "stress" / relative).resolve() for relative in lookup}
    observed = {
        item.resolve()
        for item in (output_root / "stress").rglob("*")
        if item.is_file() and item.resolve() != path.resolve()
    }
    if registered != observed:
        raise AuditFailure("stress manifest registered file set differs")
    verified = [
        file_record(verify_snapshot(record, root=project_root)) for record in records
    ]
    semantic_keys = {
        "candidate_lock": "prediction_lock",
        "durability_lock": "candidate_durability_lock",
        "results": "results",
        "promotion_lock" if expected_verdict == "PASS" else "rejection": (
            "promotion_lock" if expected_verdict == "PASS" else "rejection"
        ),
    }
    for key, stage_key in semantic_keys.items():
        record = payload.get(key)
        if not isinstance(record, Mapping):
            raise AuditFailure(f"stress manifest lacks named {key} snapshot")
        named_path = verify_snapshot(record, root=project_root)
        expected_record = _stage_record(stage, stage_key)
        expected_path = verify_snapshot(expected_record, root=project_root)
        if (
            named_path != expected_path
            or _record_size(record) != _record_size(expected_record)
            or record.get("sha256") != expected_record.get("sha256")
        ):
            raise AuditFailure(f"stress manifest named {key} binding differs")
    return {
        "file": file_record(path),
        "verdict": expected_verdict,
        "registered_output_count": len(records),
        "verified_outputs": verified,
    }


def _atomic_no_overwrite_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        # A hard-link create is atomic and fails when ``path`` already exists;
        # unlike os.replace/rename it cannot overwrite a concurrently created
        # audit.  The temporary name and destination share a parent/filesystem.
        os.link(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def postrun_audit(
    project_root: Path,
    *,
    output_root: Path,
    contract_path: Path | None,
    audit_output: Path,
    label_path: Path,
    sample_path: Path,
) -> dict[str, Any]:
    """Perform the heavy independent replay.  This function never fits."""

    if audit_output.exists():
        raise FileExistsError(audit_output)
    v1, _, _ = load_effective_science_head(project_root)
    if not output_root.is_dir():
        raise AuditFailure("postrun output root is absent")
    explicit_contract = contract_path is not None and contract_path.is_file()
    if explicit_contract:
        assert contract_path is not None
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        manifest_path = verify_snapshot(
            _root_record(contract, "root_manifest"), root=project_root
        )
    else:
        manifest_path = output_root / "manifest.json"
        if not manifest_path.is_file():
            raise AuditFailure("postrun root manifest is absent")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_records = _manifest_records(manifest)
    if not explicit_contract:
        contract = _build_contract_from_actual_layout(
            output_root=output_root,
            manifest_path=manifest_path,
            manifest=manifest,
            manifest_records=manifest_records,
            science_v1=v1,
        )
    if not _effective_science_heads_exact(contract):
        raise AuditFailure("postrun contract is not bound to effective v1+v2+v3 head")
    source_lock_path = verify_snapshot(
        _root_record(contract, "source_lock"), root=project_root
    )
    verified_outputs: list[dict[str, Any]] = []
    for record in manifest_records:
        path = verify_snapshot(record, root=project_root)
        try:
            path.relative_to(output_root)
        except ValueError as exc:
            raise AuditFailure(f"manifest output escapes output root: {path}") from exc
        verified_outputs.append(file_record(path))
    output_lookup = _manifest_lookup(manifest_records, output_root=output_root)
    registered_paths = {
        (output_root / relative).resolve() for relative in output_lookup
    }
    exclusions = {manifest_path.resolve(), audit_output.resolve()}
    if explicit_contract and contract_path is not None:
        exclusions.add(contract_path.resolve())
    observed_paths = {
        path.resolve()
        for path in output_root.rglob("*")
        if path.is_file() and path.resolve() not in exclusions
    }
    if observed_paths != registered_paths:
        raise AuditFailure(
            "root manifest file set differs: "
            f"missing={sorted(map(str, registered_paths - observed_paths))}, "
            f"unregistered={sorted(map(str, observed_paths - registered_paths))}"
        )

    source_lock = json.loads(source_lock_path.read_text(encoding="utf-8"))
    execution_lineage_report = _audit_execution_lineage(
        source_lock=source_lock,
        root_manifest=manifest,
        project_root=project_root,
        output_root=output_root,
        science_v1=v1,
    )
    source_science = source_lock.get("science")
    if not isinstance(source_science, list) or len(source_science) != 3:
        raise AuditFailure("source lock does not bind three copied science heads")
    observed_science_hashes = []
    for record in source_science:
        if not isinstance(record, Mapping):
            raise AuditFailure("source-lock science snapshot schema differs")
        copied = verify_snapshot(record, root=project_root)
        try:
            copied.relative_to(output_root / "science")
        except ValueError as exc:
            raise AuditFailure("copied science head escapes output science dir") from exc
        observed_science_hashes.append(sha256_file(copied))
    if observed_science_hashes != [spec["sha256"] for spec in SCIENCE_HEADS]:
        raise AuditFailure("source-lock copied science head order/hash differs")
    source_inputs = source_lock.get("verified_inputs")
    verified_source_inputs = audit_verified_source_inputs(
        source_inputs, project_root=project_root, science_v1=v1
    )

    stress = _contract_stage(contract, "stress")
    prediction_lock_path = verify_snapshot(
        _stage_record(stress, "prediction_lock"), root=project_root
    )
    prediction_lock = json.loads(prediction_lock_path.read_text(encoding="utf-8"))
    expected_formula_sha = hashlib.sha256(
        json.dumps(v1["delta_transfer_formula"], sort_keys=True).encode("utf-8")
    ).hexdigest()
    if (
        int(prediction_lock.get("label_suffix_reads_before_lock", -1)) != 0
        or prediction_lock.get("formula_sha") != expected_formula_sha
    ):
        raise AuditFailure("candidate lock formula/suffix prelock contract differs")
    durability_path = verify_snapshot(
        _stage_record(stress, "candidate_durability_lock"), root=project_root
    )
    durability = json.loads(durability_path.read_text(encoding="utf-8"))
    if durability.get("all_reopened_and_rehashed") is not True:
        raise AuditFailure("candidate durability rehash did not pass")
    if verify_snapshot(durability["candidate_lock"], root=project_root) != prediction_lock_path:
        raise AuditFailure("durability lock binds a different candidate lock")
    durability_outputs = durability.get("outputs")
    durability_models = durability.get("models")
    if (
        not isinstance(durability_outputs, Mapping)
        or set(durability_outputs) != set(prediction_lock.get("outputs", {}))
        or not all(
            isinstance(record, Mapping)
            and record.get("identity_equal") is True
            and int(record.get("reopened_rows", -1)) == len(STRESS_INDEX)
            for record in durability_outputs.values()
        )
        or not isinstance(durability_models, Mapping)
        or set(durability_models) != set(TARGET_COLS)
    ):
        raise AuditFailure("candidate durability output/model schema differs")
    for group in TARGET_COLS:
        pair = durability_models[group]
        if not isinstance(pair, Mapping) or set(pair) != {"control", "nonwind"}:
            raise AuditFailure("candidate durability paired model schema differs")
        if not all(
            isinstance(record, Mapping)
            and record.get("identity_equal") is True
            and record.get("joblib_reopened") is True
            for record in pair.values()
        ):
            raise AuditFailure("candidate durability model reopen differs")
    stress_formula, stress_assembly, stress_control_cf, stress_exit_cf = _audit_stage_formula(
        stage_name="stress",
        stage_contract=stress,
        project_root=project_root,
        science_v1=v1,
        expected_index=STRESS_INDEX,
    )
    a0_preflight_path = verify_snapshot(
        _stage_record(stress, "A0_preflight"), root=project_root
    )
    a0_preflight = json.loads(a0_preflight_path.read_text(encoding="utf-8"))
    preflight_groups = a0_preflight.get("groups")
    if (
        a0_preflight.get("before_any_stress_fit") is not True
        or not isinstance(preflight_groups, Mapping)
        or set(preflight_groups) != set(TARGET_COLS)
    ):
        raise AuditFailure("prefit original-A0 lock schema differs")
    for group in TARGET_COLS:
        record = preflight_groups[group]
        if (
            not isinstance(record, Mapping)
            or record.get("six_decimal_exact") is not True
            or float(record.get("max_abs_kwh", math.inf))
            != stress_formula["original_A0_max_abs_kwh_by_group"][group]
        ):
            raise AuditFailure(f"prefit original-A0 lock differs for {group}")
    locked_stress_assembly = prediction_lock.get("assembly")
    if not isinstance(locked_stress_assembly, Mapping):
        raise AuditFailure("candidate lock lacks assembly diagnostics")
    _compare_nested_numbers(
        locked_stress_assembly,
        stress_formula["runner_assembly_diagnostics_exact_shape"],
        path="stress.assembly",
    )
    stress_base = _read_target_frame(
        verify_snapshot(_stage_record(stress, "base_scale097"), root=project_root),
        expected_index=STRESS_INDEX,
    )
    stress_candidate_saved = _read_target_frame(
        verify_snapshot(_stage_record(stress, "final_candidate"), root=project_root),
        expected_index=STRESS_INDEX,
    )
    stress_candidate_expected = final_base_plus_increment(
        stress_base, stress_assembly.increment_d_kwh
    )
    if not exact_frame_equal(stress_candidate_saved, stress_candidate_expected):
        raise AuditFailure("2024 scale097+D candidate formula changed")

    score_access_path = verify_snapshot(
        _stage_record(stress, "score_label_access"), root=project_root
    )
    score_access_lock = json.loads(score_access_path.read_text(encoding="utf-8"))
    prefix_labels, labels_2024, suffix_access = _read_2024_suffix_after_lock(
        label_path,
        durable_lock_verified=(
            durability_path.is_file()
            and durability.get("all_reopened_and_rehashed") is True
        ),
    )
    stress_eligible_identities = _eligible_identities_from_labels(prefix_labels)
    stress_apply_features = {
        group: _read_apply_feature_cache(
            v1["data_contract"]["train_caches"][group],
            project_root=project_root,
            expected_index=STRESS_INDEX,
        )
        for group in TARGET_COLS
    }
    stress_reload = audit_double_reload_records(
        _stage_models(stress),
        project_root=project_root,
        stage="stress",
        expected_eligible_identities=stress_eligible_identities,
        apply_features=stress_apply_features,
        expected_predictions={
            "control_all": stress_control_cf,
            "nonwind_atmospheric": stress_exit_cf,
        },
    )
    veto = replay_seven_veto_metrics(
        labels_2024, stress_base, stress_candidate_expected
    )
    results_path = verify_snapshot(
        _stage_record(stress, "results"), root=project_root
    )
    stored_results = json.loads(results_path.read_text(encoding="utf-8"))
    stored_veto = stored_results.get(
        "historical_veto", stored_results.get("veto", stored_results)
    )
    if not isinstance(stored_veto, Mapping):
        raise AuditFailure("stress results lack canonical historical_veto")
    _compare_nested_numbers(veto, stored_veto, path="historical_veto")

    promoted = bool(veto["passed"])
    if promoted:
        promotion_path = verify_snapshot(
            _stage_record(stress, "promotion_lock"), root=project_root
        )
        promotion = json.loads(promotion_path.read_text(encoding="utf-8"))
        if str(promotion.get("verdict", "")).upper() != "PASS":
            raise AuditFailure("promotion lock disagrees with exact seven-veto replay")
        expected_promotion_paths = {
            "candidate_lock": prediction_lock_path,
            "results": results_path,
            "suffix_access": score_access_path,
        }
        for key, expected_path in expected_promotion_paths.items():
            reference = promotion.get(key)
            if not isinstance(reference, Mapping):
                raise AuditFailure(f"promotion lock lacks {key} snapshot")
            if verify_snapshot(reference, root=project_root) != expected_path:
                raise AuditFailure(f"promotion lock binds different {key}")
        decision_payload = promotion
        decision_record = file_record(promotion_path)
    else:
        rejection_path = verify_snapshot(
            _stage_record(stress, "rejection"), root=project_root
        )
        rejection = json.loads(rejection_path.read_text(encoding="utf-8"))
        if rejection.get("verdict") != "REJECT_NO_FINAL_CSV" or not rejection.get(
            "no_rescue", False
        ):
            raise AuditFailure("rejection lock disagrees with exact seven-veto replay")
        _compare_nested_numbers(rejection.get("gates"), veto["gates"], path="rejection.gates")
        decision_payload = rejection
        decision_record = file_record(rejection_path)

    root_manifest_semantics = _audit_root_manifest_semantics(
        manifest,
        stored_results=stored_results,
        promoted=promoted,
        decision_payload=decision_payload,
    )
    stress_manifest_report = _audit_stress_manifest(
        stress,
        expected_verdict="PASS" if promoted else "REJECT",
        project_root=project_root,
        output_root=output_root,
    )

    final_report: dict[str, Any]
    if promoted:
        final_stage = _contract_stage(contract, "final")
        full_labels = pd.concat([prefix_labels, labels_2024])
        expected_full_index = pd.date_range(
            "2022-01-01 01:00:00",
            "2025-01-01 00:00:00",
            freq="h",
            name="forecast_kst_dtm",
        )
        if (
            len(full_labels) != 26_304
            or tuple(full_labels.columns) != TARGET_COLS
            or not full_labels.index.equals(expected_full_index)
            or not full_labels.index.is_unique
        ):
            raise AuditFailure("in-memory full fit-label frame differs")
        final_eligible_identities = _eligible_identities_from_labels(full_labels)
        final_apply_features = {
            group: _read_apply_feature_cache(
                v1["data_contract"]["test_caches"][group],
                project_root=project_root,
                expected_index=TEST_INDEX,
            )
            for group in TARGET_COLS
        }
        verified_test_cache_inputs = [
            dict(v1["data_contract"]["test_caches"][group])
            for group in TARGET_COLS
        ]
        canonical_claim = manifest.get("final", {}).get("canonical_q07_replay")
        if not isinstance(canonical_claim, Mapping):
            raise AuditFailure("root manifest lacks canonical q07 replay claim")
        canonical_q07_audit = audit_canonical_q07_replay(
            science_v1=v1,
            project_root=project_root,
            test_features=final_apply_features,
            manifest_claim=canonical_claim,
        )
        final_formula, final_assembly, final_control_cf, final_exit_cf = _audit_stage_formula(
            stage_name="final",
            stage_contract=final_stage,
            project_root=project_root,
            science_v1=v1,
            expected_index=TEST_INDEX,
        )
        final_reload = audit_double_reload_records(
            _stage_models(final_stage),
            project_root=project_root,
            stage="final",
            expected_eligible_identities=final_eligible_identities,
            apply_features=final_apply_features,
            expected_predictions={
                "control_all": final_control_cf,
                "nonwind_atmospheric": final_exit_cf,
            },
        )
        stored_final_assembly = manifest.get("final", {}).get("assembly")
        if not isinstance(stored_final_assembly, Mapping):
            raise AuditFailure("root manifest lacks final assembly diagnostics")
        _compare_nested_numbers(
            stored_final_assembly,
            final_formula["runner_assembly_diagnostics_exact_shape"],
            path="final.assembly",
        )
        final_base_record = _stage_record(final_stage, "base_scale097")
        final_base_path = verify_snapshot(final_base_record, root=project_root)
        if sha256_file(final_base_path) != FINAL_BASE_SHA256:
            raise AuditFailure("final base is not immutable 85ad... scale097")
        final_base = _read_target_frame(final_base_path, expected_index=TEST_INDEX)
        final_saved = _read_target_frame(
            verify_snapshot(_stage_record(final_stage, "final_prediction"), root=project_root),
            expected_index=TEST_INDEX,
        )
        final_expected = final_base_plus_increment(
            final_base, final_assembly.increment_d_kwh
        )
        if not exact_frame_equal(final_saved, final_expected):
            raise AuditFailure("final base85ad+D formula changed")
        csv_path = verify_snapshot(
            _stage_record(final_stage, "final_csv"), root=project_root
        )
        contract_sample_path = verify_snapshot(
            _stage_record(final_stage, "sample"), root=project_root
        )
        if contract_sample_path != sample_path.resolve():
            raise AuditFailure("CLI sample path differs from hash-bound sample")
        csv_report = validate_submission_csv(csv_path, contract_sample_path, final_expected)
        if not csv_report["pass"]:
            raise AuditFailure("final CSV contract failed")
        final_report = {
            "conditional_stage": "executed_after_exact_veto_pass",
            "verified_test_cache_inputs": verified_test_cache_inputs,
            "canonical_q07_replay": canonical_q07_audit,
            "reload": final_reload,
            "formula": final_formula,
            "base85ad_plus_D_bit_exact": True,
            "csv": csv_report,
        }
    else:
        if contract.get("stages", {}).get("final") not in (None, {}, {"status": "not_run"}):
            raise AuditFailure("veto failed but final-stage artifacts are registered")
        csv_files = list(output_root.rglob("*.csv"))
        if csv_files:
            raise AuditFailure("veto failed but a CSV exists")
        final_report = {
            "conditional_stage": "correctly_absent_after_veto_failure",
            "csv_count": 0,
        }

    access_path = verify_snapshot(
        _root_record(contract, "access_ledger"), root=project_root
    )
    access_claims = json.loads(access_path.read_text(encoding="utf-8"))
    access_events = access_claims.get("events", [])
    if (
        not isinstance(access_events, list)
        or len(access_events) < 7
        or not isinstance(access_events[6], Mapping)
        or not isinstance(access_events[6].get("durability_lock"), Mapping)
        or verify_snapshot(
            access_events[6]["durability_lock"], root=project_root
        )
        != durability_path
    ):
        raise AuditFailure("access event does not bind exact candidate durability lock")
    access_audit = validate_v3_access_ledger(
        access_claims,
        promoted=promoted,
        source_lock=source_lock,
        score_label_access=score_access_lock,
    )
    deferred_test_raw = source_lock.get("deferred_official_test_raw")
    if (
        not isinstance(deferred_test_raw, list)
        or len(deferred_test_raw) != 2
        or not all(
            isinstance(record, Mapping)
            and record.get("hash_verified") is False
            and int(record.get("size_bytes", -1)) > 0
            and bool(re.fullmatch(r"[0-9a-f]{64}", str(record.get("sha256", ""))))
            for record in deferred_test_raw
        )
    ):
        raise AuditFailure("deferred official-test raw input lock differs")
    if promoted:
        official_test_raw = access_events[10].get("official_test_raw")
        if not isinstance(official_test_raw, list) or len(official_test_raw) != 2:
            raise AuditFailure("promotion event lacks official-test raw snapshots")
        verified_official_test_raw: list[dict[str, Any]] = []
        for deferred, verified in zip(
            deferred_test_raw, official_test_raw, strict=True
        ):
            if not isinstance(verified, Mapping) or verified.get("hash_verified") is not True:
                raise AuditFailure("official-test raw hash verification claim differs")
            for key in ("path", "size_bytes", "sha256"):
                if deferred.get(key) != verified.get(key):
                    raise AuditFailure("deferred/verified official-test snapshot differs")
            verified_official_test_raw.append(
                file_record(verify_snapshot(verified, root=project_root))
            )
        official_test_raw_audit: Mapping[str, Any] = {
            "conditional_branch": "verified_after_promotion",
            "files": verified_official_test_raw,
        }
    else:
        official_test_raw_audit = {
            "conditional_branch": "correctly_not_opened_after_rejection",
            "deferred_snapshot_count": 2,
        }
    if contract.get("actual_layout") is True:
        event_names = [str(event.get("event")) for event in access_claims["events"]]
        forbidden_tokens = ("retry", "alternate", "router", "rescue", "retune")
        attempt_path = (
            project_root
            / "artifacts/locks/feature_exit_nonwind_cap025143_scale097_overlay_v1.attempt.json"
        )
        if not attempt_path.is_file():
            raise AuditFailure("single-attempt lock is absent")
        attempt = json.loads(attempt_path.read_text(encoding="utf-8"))
        no_rescue_checks = {
            "single_attempt_lock_exact": (
                attempt.get("experiment_id")
                == "feature_exit_nonwind_cap025143_scale097_overlay_v1"
                and attempt.get("single_attempt") is True
            ),
            "decision_lock_no_rescue": decision_payload.get("no_rescue") is True,
            "no_retry_or_rescue_access_event": not any(
                token in name.lower()
                for name in event_names
                for token in forbidden_tokens
            ),
            "single_stress_candidate_lock": sum(
                relative.endswith("stress/candidate_lock.json")
                for relative in output_lookup
            )
            == 1,
            "no_alternate_candidate_artifact": not any(
                any(token in relative.lower() for token in forbidden_tokens)
                for relative in output_lookup
            ),
        }
        if not all(no_rescue_checks.values()):
            raise AuditFailure(f"actual no-rescue audit failed: {no_rescue_checks}")
        no_rescue: Mapping[str, Any] = {
            "schema": "runner_single_attempt_decision_and_output_set_v1",
            "checks": no_rescue_checks,
            "attempt_lock": file_record(attempt_path),
        }
    else:
        no_rescue = contract.get("no_rescue", {})
        required_zeroes = (
            "retry_count",
            "alternate_cap_count",
            "alternate_base_count",
            "group_router_count",
            "month_router_count",
            "rescue_count",
        )
        if not all(int(no_rescue.get(key, -1)) == 0 for key in required_zeroes):
            raise AuditFailure("no-rescue ledger is incomplete or nonzero")

    checks = {
        "execution_lineage_and_attempt_three_way_exact": execution_lineage_report[
            "pass"
        ],
        "root_manifest_semantic_branch_exact": root_manifest_semantics["pass"],
        "all_manifest_hashes_exact": True,
        "stress_six_models_independent_double_reload": stress_reload["pass"],
        "stress_cap_q1_reconstructed_A0_A1_and_saved_D_exact": True,
        "original_2024_A0_reconstruction": True,
        "g2_b0_frozen_b1_report_only": True,
        "seven_veto_metrics_exact": True,
        "promotion_or_rejection_lock_exact": True,
        "conditional_final_contract_exact": True,
        "v3_access_ledger_exact": access_audit["pass"],
        "no_rescue_contract_exact": True,
    }
    report = {
        "schema_version": 1,
        "audit_id": "feature_exit_nonwind_cap025143_scale097_overlay_postrun_v1",
        "status": "PASS_POSTRUN",
        "checks": checks,
        "science_heads": [
            file_record(project_root / spec["relative"]) for spec in SCIENCE_HEADS
        ],
        "semantic_contract": (
            file_record(contract_path)
            if explicit_contract and contract_path is not None
            else {"origin": "runner_locks_and_hash_manifest", "file": None}
        ),
        "root_manifest": file_record(manifest_path),
        "root_manifest_semantics": root_manifest_semantics,
        "source_lock": file_record(source_lock_path),
        "execution_lineage": execution_lineage_report,
        "verified_source_inputs": verified_source_inputs,
        "verified_manifest_outputs": verified_outputs,
        "stress": {
            "reload": stress_reload,
            "formula": stress_formula,
            "seven_veto": veto,
            "prediction_lock": file_record(prediction_lock_path),
            "prefit_original_A0_lock": file_record(a0_preflight_path),
            "candidate_durability_lock": file_record(durability_path),
            "stress_manifest": stress_manifest_report,
            "score_label_access": file_record(score_access_path),
            "decision_lock": decision_record,
        },
        "final": final_report,
        "runner_access_claims": access_claims,
        "runner_access_audit": access_audit,
        "official_test_raw_audit": official_test_raw_audit,
        "independent_suffix_access": suffix_access,
        "no_rescue": no_rescue,
        "auditor": file_record(Path(__file__).resolve()),
        "auditor_did_not_fit_or_write_submission": True,
    }
    _atomic_no_overwrite_json(audit_output, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--static-audit", action="store_true")
    mode.add_argument("--postrun", action="store_true")
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--postrun-contract", type=Path)
    parser.add_argument("--audit-output", type=Path)
    parser.add_argument(
        "--label-file",
        type=Path,
        default=Path("data/local/open/train/train_labels.csv"),
    )
    parser.add_argument(
        "--sample-submission",
        type=Path,
        default=Path("data/local/open/sample_submission.csv"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = args.project_root.expanduser().resolve()
    if args.static_audit:
        report = static_audit(project_root)
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root is not None
        else (project_root / OUTPUT_RELATIVE).resolve()
    )
    contract_path = (
        args.postrun_contract.expanduser().resolve()
        if args.postrun_contract is not None
        else output_root / "postrun_contract.json"
    )
    audit_output = (
        args.audit_output.expanduser().resolve()
        if args.audit_output is not None
        else project_root
        / "artifacts/audits/feature_exit_nonwind_cap025143_scale097_overlay_postrun_v1.json"
    )
    report = postrun_audit(
        project_root,
        output_root=output_root,
        contract_path=contract_path,
        audit_output=audit_output,
        label_path=args.label_file.expanduser().resolve(),
        sample_path=args.sample_submission.expanduser().resolve(),
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "audit_output": str(audit_output),
                "sha256": sha256_file(audit_output),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
