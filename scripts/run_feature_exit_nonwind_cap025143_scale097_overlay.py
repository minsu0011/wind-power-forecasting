"""Run the frozen non-wind feature-exit, scale-0.97 overlay experiment.

The command is intentionally single-process.  It first fits and locks the
2024 stress candidate from a physically bounded label prefix.  Only after the
candidate lock exists does it read the 2024 label suffix once.  A failed veto
terminates before any test cache, final component, or sample file is opened.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import shutil
import sys
from typing import Any, Mapping, Sequence
import uuid

import joblib
from lightgbm import LGBMRegressor
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.density_ratio import assemble_locked_group  # noqa: E402
from src.manifest import describe_file, package_versions, sha256_file, utc_now  # noqa: E402
from src.metric import CAPACITY_KWH, TARGET_COLS, group_metrics  # noqa: E402
from src.virtual_feature_exit import kept_features  # noqa: E402


EXPERIMENT_ID = "feature_exit_nonwind_cap025143_scale097_overlay_v1"
SCIENCE_FILES = (
    (
        ROOT / "configs/feature_exit_nonwind_cap025143_scale097_overlay_preregister_v1.json",
        15_828,
        "91fa72355bab5bef80dcb73816e9bea162bf80ef26183333f9798d32770e42f9",
    ),
    (
        ROOT / "configs/feature_exit_nonwind_cap025143_scale097_overlay_preregister_v2.json",
        2_110,
        "802dc38d6692eb6ecc546bc42743609e06d4bef79deaa6103dd5fba9843cb0da",
    ),
    (
        ROOT / "configs/feature_exit_nonwind_cap025143_scale097_overlay_preregister_v3.json",
        2_704,
        "09e2b3ca41b8be49c5084b986dc6f0b2f7eedc432f65525ceb65cf2e3b3ac650",
    ),
)
DEFAULT_OUT = ROOT / f"artifacts/postgate/{EXPERIMENT_ID}"
ATTEMPT = ROOT / f"artifacts/locks/{EXPERIMENT_ID}.attempt.json"
HEAVY_GUARD = ROOT / "artifacts/locks/heavy_cpu_fit.pid.json"
EXECUTION_AMENDMENT = ROOT / "configs/feature_exit_nonwind_cap025143_scale097_overlay_execution_v4.json"
PRELAUNCH_REVIEW = ROOT / "artifacts/audits/feature_exit_nonwind_cap025143_scale097_overlay_v1_prelaunch_independent_review.json"
RAW_LABELS = Path(r"data/local/open/train/train_labels.csv")
SAMPLE = Path(r"data/local/open/sample_submission.csv")
INFO_XLSX = (
    Path(r"data/local/open/info.xlsx"),
    3_823_422,
    "89e83a52e0eb2ce367a3573a96d6795ed4b4d4ac624965cb3530beec0cbd2bd6",
)
PREFIX_BYTES = 742_551
PREFIX_SHA = "ee9707cc81229db760fc035d0bf1d10eb3586d2fef95849bdf463d73fae3aacd"
SUFFIX_BYTES = 396_416
SUFFIX_SHA = "0678740abe23800de9be272d2992ff71e59571ac169a410da1d8a3ab28f544da"
FULL_LABEL_SHA = "47bb64252195cf4734e67394d6e50485f27a608def3b5a8791fcc7674bbceb03"
FULL_LABEL_BYTES = 1_138_967
CAP_CF = 0.02514322112138523
EXPECTED_FEATURE_SHA = "55835269be52c5ceb681435a17c8e1ff8c8f40f36393706691996cfc4c143276"
EXPECTED_KEEP_SHA = "9022f9459366dcdbe91e59866c8b8d82c75617716c38f939e06daf97f27586cb"
STRESS_END = pd.Timestamp("2024-01-01 00:00:00")
STRESS_INDEX = pd.date_range(
    "2024-01-01 01:00:00", "2025-01-01 00:00:00", freq="h", name="forecast_kst_dtm"
)
TEST_INDEX = pd.date_range(
    "2025-01-01 01:00:00", "2026-01-01 00:00:00", freq="h", name="forecast_kst_dtm"
)
STRESS_COUNTS = {"kpx_group_1": 10_925, "kpx_group_2": 10_914, "kpx_group_3": 4_847}
FINAL_COUNTS = {"kpx_group_1": 15_915, "kpx_group_2": 15_891, "kpx_group_3": 9_414}
PARAMS = {
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
TRAIN_CACHES = {
    "kpx_group_1": (ROOT / "artifacts/cache/kpx_group_1_weather_train.parquet", 77_942_069, "c3526f861184a16fef4a68c20ad8f4defab04b7dc0650b571c6beba05a867579"),
    "kpx_group_2": (ROOT / "artifacts/cache/kpx_group_2_weather_train.parquet", 77_929_519, "0e6fc7334e7094af1a2fffceffeee7628930aaa102c43515fde3315ec806a31e"),
    "kpx_group_3": (ROOT / "artifacts/cache/kpx_group_3_weather_train.parquet", 77_950_424, "eb61868a0fd564a60f31fdd9b1fd6324545a168fda9b0137049754d014b332ce"),
}
TEST_CACHES = {
    "kpx_group_1": (ROOT / "artifacts/cache/kpx_group_1_weather_test.parquet", 26_037_151, "8de6fe0ba96ee5c4d0bae16c1b57bc761863152064fda490caf048a5b6e87771"),
    "kpx_group_2": (ROOT / "artifacts/cache/kpx_group_2_weather_test.parquet", 26_031_426, "3450c23bd1e6a8270f416956d65f7df0a0265d3d67ea30c1b69dd4f214b43e99"),
    "kpx_group_3": (ROOT / "artifacts/cache/kpx_group_3_weather_test.parquet", 26_038_173, "54ee1011e1c9f328347500fe1f6113919f542bf7697126201cedc16bfdec2ed6"),
}
OFFICIAL_TRAIN_RAW = (
    (Path(r"data/local/open/train/ldaps_train.csv"), 129_687_357, "61ae944e7ae1fcb17391be6737792a2205c6507bf2446ed5d9d0daf07fdea026"),
    (Path(r"data/local/open/train/gfs_train.csv"), 84_315_594, "cd56b67d357e7bbaff5d0d51d3537d935c9e7a3f012e9f37516bdc4d38c66a5d"),
)
OFFICIAL_TEST_RAW = (
    (Path(r"data/local/open/test/ldaps_test.csv"), 43_122_637, "60e94f7cc80384eee335e90dc896b6cf4d36b35cde8d37bc03bd5c08a788b0fa"),
    (Path(r"data/local/open/test/gfs_test.csv"), 28_037_722, "aa33febb24ecd46b82be34880a06910e16a3382319287548e6ce2af721b4f848"),
)
GATE_RECIPE = (ROOT / "artifacts/gate/v3/gate_recipe_snapshot.json", 5_613, "c26e0b0b4268ed3780d85e1c3af5db38253d42e8a4230fbd910691f2ec6acd19")
FINAL_RECIPE = (ROOT / "artifacts/final_v3/v3_locked_full_2025__recipe.json", 5_613, "c26e0b0b4268ed3780d85e1c3af5db38253d42e8a4230fbd910691f2ec6acd19")
GATE_REFERENCE = (ROOT / "artifacts/gate/v3/gate_oof.parquet", 314_530, "1c0a2e70996566c766a90a557c83ab1ba010277c31d202a1e4209188b94d112c")
GATE_COMPONENTS = {
    "lgb_l1": (ROOT / "artifacts/gate/v3/predictions/lgb_l1_gate.parquet", 315_123, "eb1297928c23838d791ae476107abba10923d4e8bf3d7bda6681625f5fc30cdb"),
    "lgb_q07": (ROOT / "artifacts/gate/v3/predictions/lgb_q07_gate.parquet", 315_197, "81d5614cd29f6409da9900dcc628b1a82cc998ddabfe9bab66dabb4f6a67b068"),
    "shared_l1": (ROOT / "artifacts/gate/v3/predictions/shared_l1_gate.parquet", 315_202, "1f82759bc61c8230a8c9fcb3426da28dd77a610dca3e7784e899493099158265"),
    "shared_q07": (ROOT / "artifacts/gate/v3/predictions/shared_q07_gate.parquet", 315_236, "3e47a52d62ebbe5d9988f15d5a424cecf65e6485cfe930a8ccee216450cdd818"),
    "top200_q07": (ROOT / "artifacts/gate/v3/predictions/top200_q07_gate.parquet", 315_236, "c631e5d62379c8f3c5b5012905c3355004a54529dbf095d667a8beacf7d7c4e1"),
    "energy_q06": (ROOT / "artifacts/gate/v3/predictions/energy_q06_gate.parquet", 315_239, "fb41ff67fc95328431ef174273f265e8ae25c153721294bcdf466ac30ef5eeec"),
}
STRESS_BASE = (ROOT / "artifacts/postgate/public_adaptive_scale097_g2_delta_v2/diagnostic_2024/A_scale097.parquet", 335_807, "93d8e2b7d19971ba46a29d548e77ef4f85553bbbdaad2b5983a5b6f1d84e406c")
FINAL_COMPONENTS = {
    "lgb_l1": (ROOT / "artifacts/final_v3/predictions/v3_locked_full_2025__lgb_l1_test.parquet", 314_371, "42ddac574b65e5d245d85bf16fcde3814b0e97bb47db622c37e1f0c00d67d89d"),
    "lgb_q07": (ROOT / "artifacts/final_v3/predictions/v3_locked_full_2025__lgb_q07_test.parquet", 314_318, "3945e6b05c9f5c89f53406bd5a993212d9c4bcd5a5f9f1a7ffb6e2d3b3530d24"),
    "shared_l1": (ROOT / "artifacts/final_v3/predictions/v3_locked_full_2025__shared_l1_test.parquet", 314_378, "55137b29c45c7eed6b91439032918d0fd627d386d4c25e78484c4e4dcda757a4"),
    "shared_q07": (ROOT / "artifacts/final_v3/predictions/v3_locked_full_2025__shared_q07_test.parquet", 314_400, "8242b4d729ae90f0532e6d2a5f7b99cb5552fd1ef248545ba7034a143bcf9c85"),
    "top200_q07": (ROOT / "artifacts/final_v3/predictions/v3_locked_full_2025__top200_q07_test.parquet", 314_299, "58cb4def779e181d7f57d20e4a139ece6526ebe9533cc6970684a12b4bc372f2"),
    "energy_q06": (ROOT / "artifacts/final_v3/predictions/v3_locked_full_2025__energy_q06_test.parquet", 314_352, "2116d66343d8a8241a3fdf1bb40cae357ce10c87a0b0974523506f6346790203"),
}
FINAL_REFERENCE = (ROOT / "artifacts/final_v3/predictions/v3_locked_full_2025__final_test.parquet", 311_860, "1063fa6e96d34947d794d117fc8a122de3de19977b2a95499aba029dcae5f1ff")
FINAL_Q07_MODEL = (ROOT / "artifacts/final_v3/models/v3_locked_full_2025__lgb_q07.joblib", 5_375_178, "077d6b99af939f35ecc556823844c60aec86aaa857a79075fa64c88294d4605e")
FINAL_BASE = (ROOT / "artifacts/postgate/public_adaptive_scale097_g2_delta_v2/final/A_plain_scale097.parquet", 334_789, "85ad8c63eaf7bff4b0390fb8c42dbb31c22acd5e7a64fbec350d9c3878818621")
SAMPLE_IDENTITY = (SAMPLE, 359_229, "c925d2066a834f937f8091ed55acfe50ff86c8be4745b52c3adc95b056c5aaaa")
SMALL_PROVENANCE = (
    INFO_XLSX,
    (ROOT / "artifacts/incidents/feature_exit_nonwind_cap025143_champion_overlay_v1_v2_preexecution_supersession_20260810.json", 1_918, "d94a4d453c908c27404b5061957e322b60e7154141231173c86aa97ea7290479"),
    (ROOT / "artifacts/feature_exit_virtual_v1/audits/nonwind_cap_v3_evidence/nonwind_cap_evidence.json", 25_202, "315b3e5d43ec62d395bf73e98bfbc85fc8d5fffb34c24c9c8f53a34307322840"),
    (ROOT / "scripts/build_nonwind_cap_v3_evidence.py", 26_898, "d5edd8330c9a3f4b5c6d86f58d71993599f6de11259766f9c1d2a1794676159d"),
    (ROOT / "artifacts/feature_exit_virtual_v1/inner/inner_oof_predictions.parquet", 826_334, "6907c9c27e3d98f0aaac6ca59e41c069889b235b5303a61a61793ea9e3278f2d"),
    (ROOT / "artifacts/gate/v3/gate_manifest.json", 19_529, "c95ca55ef8d09668b6376aee3773f80fd2f89358c00cdc4e9f5bc9727952aeaa"),
    (ROOT / "artifacts/final_v3/v3_locked_full_2025__manifest.json", 17_646, "f2181ebbb1666d278942c7555ad9bf6edb2ca97d7608c4981c3e6f719344a22a"),
    (ROOT / "artifacts/postgate/public_adaptive_scale097_g2_delta_v2/manifest_frozen_postrun_v2.json", 14_871, "3ef6fabf059f3bd67e3b933e8eb633e5ebf3b422d7c123f1b27af461fcc2e153"),
    (ROOT / "configs/public_adaptive_scale097_g2_delta_preregister_v2.json", 4_349, "9ed9bc8d4ec73f714cbd4d1b1c41ac8b5c45a7812a1bd2bbe28a8c5b5b24c475"),
    (ROOT / "src/metric.py", 7_248, "555950e6892a808d9091b4e6749128b9f96888ce0a83270b6959a4e927dc5f1d"),
    (ROOT / "src/density_ratio.py", 10_259, "f287cac2f1497525615b391fe97356684c2344b80fb6fd66f3fa6fc9c3bab6a5"),
    (ROOT / "src/virtual_feature_exit.py", 13_892, "9ba3ce9445ac58cd4db57456ea11acd2f18e6e20daa2e55845f5e42f3d49bdfa"),
    (ROOT / "src/features.py", 34_098, "34c7bf46444c9a2a88eabdc23f41254f66a9ba04eda1adfa53e19b1263a81a5e"),
    (ROOT / "scripts/build_features.py", 5_857, "423a936e0f857f031fb891e11451ed037d68a99ee2aa1acf03b364f368d99f10"),
)


def _sha_lines(values: Sequence[str]) -> str:
    return hashlib.sha256(("\n".join(map(str, values)) + "\n").encode()).hexdigest()


def _identity(spec: tuple[Path, int, str], *, hash_file: bool = True) -> dict[str, Any]:
    path, size, digest = spec
    if not path.is_file() or path.stat().st_size != size:
        raise AssertionError(f"input size differs: {path}")
    if hash_file and sha256_file(path) != digest:
        raise AssertionError(f"input SHA differs: {path}")
    return {"path": str(path.resolve()), "size_bytes": size, "sha256": digest, "hash_verified": hash_file}


def _declared_identity(role: str, spec: tuple[Path, int, str]) -> dict[str, Any]:
    path, size, digest = spec
    return {
        "role": role,
        "path": str(path.resolve()),
        "size_bytes": int(size),
        "sha256": digest,
    }


def _authorization_data_declarations() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = [
        _declared_identity("train_labels", (RAW_LABELS, FULL_LABEL_BYTES, FULL_LABEL_SHA)),
        _declared_identity("sample_submission", SAMPLE_IDENTITY),
        _declared_identity("info_xlsx", INFO_XLSX),
        _declared_identity("gate_recipe", GATE_RECIPE),
        _declared_identity("final_recipe", FINAL_RECIPE),
        _declared_identity("gate_reference", GATE_REFERENCE),
        _declared_identity("stress_scale097_base", STRESS_BASE),
        _declared_identity("final_scale097_base", FINAL_BASE),
        _declared_identity("final_locked_reference", FINAL_REFERENCE),
        _declared_identity("final_canonical_q07_model", FINAL_Q07_MODEL),
    ]
    for prefix, specs in (
        ("official_train_raw", OFFICIAL_TRAIN_RAW),
        ("official_test_raw", OFFICIAL_TEST_RAW),
    ):
        records.extend(_declared_identity(f"{prefix}_{i}", spec) for i, spec in enumerate(specs, start=1))
    for prefix, specs in (
        ("train_cache", TRAIN_CACHES),
        ("test_cache", TEST_CACHES),
        ("gate_component", GATE_COMPONENTS),
        ("final_component", FINAL_COMPONENTS),
    ):
        records.extend(_declared_identity(f"{prefix}_{name}", spec) for name, spec in specs.items())
    return records


def _authorization_bound_role_paths() -> dict[str, Path]:
    return {
        "runner": Path(__file__).resolve(),
        "independent_postrun_auditor": (ROOT / "scripts/audit_feature_exit_nonwind_cap025143_scale097_overlay.py").resolve(),
        "focused_runner_tests": (ROOT / "tests/test_feature_exit_nonwind_cap025143_scale097_overlay.py").resolve(),
        "independent_auditor_tests": (ROOT / "tests/test_feature_exit_nonwind_cap025143_scale097_overlay_auditor.py").resolve(),
        "root_contract_tests": (ROOT / "tests/test_feature_exit_nonwind_cap025143_scale097_overlay_root.py").resolve(),
        "src_metric": (ROOT / "src/metric.py").resolve(),
        "src_virtual_feature_exit": (ROOT / "src/virtual_feature_exit.py").resolve(),
        "src_density_ratio": (ROOT / "src/density_ratio.py").resolve(),
        "src_manifest": (ROOT / "src/manifest.py").resolve(),
        "src_features": (ROOT / "src/features.py").resolve(),
        "build_features": (ROOT / "scripts/build_features.py").resolve(),
        "info_xlsx": INFO_XLSX[0].resolve(),
    }


def _verify_execution_authorization(path: Path, out: Path) -> dict[str, Any]:
    if path.resolve() != EXECUTION_AMENDMENT.resolve():
        raise AssertionError("--execution-amendment must be the canonical frozen v4 path")
    if not path.is_file():
        raise FileNotFoundError(f"execution amendment is absent: {path}")
    observed = describe_file(path)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    expected_sidecar = f"{observed['sha256']}  {path.name}\n"
    if not sidecar.is_file() or sidecar.read_text(encoding="utf-8") != expected_sidecar:
        raise AssertionError("execution amendment sidecar differs")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or payload.get("experiment_id") != EXPERIMENT_ID:
        raise AssertionError("execution amendment identity differs")
    if payload.get("status") != "AUTHORIZED_SINGLE_EXECUTION" or payload.get("single_attempt") is not True:
        raise AssertionError("execution amendment does not authorize one attempt")
    if payload.get("canonical_output") != str(out.resolve()):
        raise AssertionError("execution amendment output differs")
    if payload.get("science_heads") != [
        _declared_identity(f"science_v{i}", spec) for i, spec in enumerate(SCIENCE_FILES, start=1)
    ]:
        raise AssertionError("execution amendment science heads differ")
    if payload.get("data_declarations") != _authorization_data_declarations():
        raise AssertionError("execution amendment data declarations differ")
    required_paths = _authorization_bound_role_paths()
    required_roles = set(required_paths)
    bound_files = payload.get("bound_files")
    if (
        not isinstance(bound_files, list)
        or len(bound_files) != len(required_roles)
        or len({x.get("role") for x in bound_files if isinstance(x, dict)}) != len(bound_files)
        or {x.get("role") for x in bound_files if isinstance(x, dict)} != required_roles
    ):
        raise AssertionError("execution amendment bound-file roles differ")
    verified_files: list[dict[str, Any]] = []
    for record in bound_files:
        spec = (Path(record["path"]), int(record["size_bytes"]), str(record["sha256"]))
        if spec[0].resolve() != required_paths[record["role"]]:
            raise AssertionError(f"execution amendment canonical path differs: {record['role']}")
        verified = _identity(spec)
        verified["role"] = record["role"]
        verified_files.append(verified)
    runner_record = next(x for x in bound_files if x["role"] == "runner")
    if Path(runner_record["path"]).resolve() != Path(__file__).resolve():
        raise AssertionError("execution amendment runner path differs")
    runtime = payload.get("runtime")
    expected_runtime = {
        "python": platform.python_version(),
        "packages": package_versions(("numpy", "pandas", "scikit-learn", "lightgbm", "pyarrow", "joblib")),
    }
    if runtime != expected_runtime:
        raise AssertionError("execution amendment runtime differs")
    zero = payload.get("prelaunch_zero_state")
    if zero != {
        "canonical_output_absent": True,
        "attempt_absent": True,
        "heavy_guard_absent": True,
    }:
        raise AssertionError("execution amendment zero-state declaration differs")
    if out.exists() or ATTEMPT.exists() or HEAVY_GUARD.exists():
        raise AssertionError("execution zero-state differs")
    if payload.get("required_independent_prelaunch_review") != str(PRELAUNCH_REVIEW.resolve()):
        raise AssertionError("execution amendment prelaunch review path differs")
    review_path = PRELAUNCH_REVIEW
    if not review_path.is_file():
        raise FileNotFoundError(f"independent prelaunch review is absent: {review_path}")
    review = json.loads(review_path.read_text(encoding="utf-8"))
    if review.get("experiment_id") != EXPERIMENT_ID or review.get("verdict") != "PASS" or review.get("blocking_defects") != 0:
        raise AssertionError("independent prelaunch review does not pass")
    if review.get("execution_amendment", {}).get("sha256") != observed["sha256"]:
        raise AssertionError("independent review is not bound to execution amendment")
    if review.get("bound_runner", {}).get("sha256") != runner_record["sha256"]:
        raise AssertionError("independent review is not bound to runner")
    return {
        "file": observed,
        "sidecar": describe_file(sidecar),
        "payload": payload,
        "verified_bound_files": verified_files,
        "independent_prelaunch_review": describe_file(review_path),
    }


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with tmp.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    frame.to_parquet(tmp, engine="pyarrow", compression="zstd", index=True)
    with tmp.open("r+b") as stream:
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def _atomic_joblib(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    joblib.dump(value, tmp, compress=3)
    with tmp.open("r+b") as stream:
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def _atomic_copy(source: Path, destination: Path) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    tmp = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    with source.open("rb") as reader, tmp.open("wb") as writer:
        shutil.copyfileobj(reader, writer)
        writer.flush()
        os.fsync(writer.fileno())
    os.replace(tmp, destination)
    if sha256_file(source) != sha256_file(destination):
        raise AssertionError(f"atomic copy differs: {source}")
    return describe_file(destination)


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, int(pid))
        if not handle:
            return False
        ctypes.windll.kernel32.CloseHandle(handle)
        return True
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


class HeavyGuard:
    def __init__(self) -> None:
        self.token = uuid.uuid4().hex
        self.payload: dict[str, Any] | None = None

    def __enter__(self) -> "HeavyGuard":
        HEAVY_GUARD.parent.mkdir(parents=True, exist_ok=True)
        if HEAVY_GUARD.exists():
            record = json.loads(HEAVY_GUARD.read_text(encoding="utf-8"))
            raise RuntimeError(f"heavy guard already exists (alive={_pid_alive(int(record.get('pid', -1)))}): {record}")
        self.payload = {"schema_version": 1, "experiment_id": EXPERIMENT_ID, "pid": os.getpid(), "token": self.token, "created_utc": utc_now()}
        fd = os.open(HEAVY_GUARD, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(self.payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        return self

    def verify(self) -> None:
        if self.payload is None or not HEAVY_GUARD.is_file():
            raise AssertionError("heavy guard missing")
        if json.loads(HEAVY_GUARD.read_text(encoding="utf-8")) != self.payload:
            raise AssertionError("heavy guard identity changed")

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.verify()
        HEAVY_GUARD.unlink()
        self.payload = None


def _verify_science() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    payloads = []
    for path, size, digest in SCIENCE_FILES:
        _identity((path, size, digest))
        sidecar = path.with_suffix(path.suffix + ".sha256")
        expected = f"{digest}  {path.name}\n"
        if sidecar.read_text(encoding="utf-8") != expected:
            raise AssertionError(f"science sidecar differs: {sidecar}")
        payloads.append(json.loads(path.read_text(encoding="utf-8")))
    v1, v2, v3 = payloads
    if any(x["experiment_id"] != EXPERIMENT_ID for x in payloads):
        raise AssertionError("experiment identity differs")
    if "gate2024_locked_v3_cf_fix" not in json.dumps(v2) or not v2["correction"]["incorrect_v1_stress_reference"]["must_not_be_used_by_this_experiment"]:
        raise AssertionError("wrong stress reference is not explicitly rejected")
    if float(v1["cap_evidence"]["cap_cf"]) != CAP_CF:
        raise AssertionError("cap differs")
    if v1["paired_model_contract"]["parameters"] != PARAMS:
        raise AssertionError("paired model parameters differ")
    if v1["paired_model_contract"]["stress_expected_eligible_counts"] != STRESS_COUNTS:
        raise AssertionError("stress eligible counts differ")
    if v1["paired_model_contract"]["final_expected_eligible_counts"] != FINAL_COUNTS:
        raise AssertionError("final eligible counts differ")
    if not v1["delta_transfer_formula"]["cap_applied_exactly_once"] or v1["delta_transfer_formula"]["per_group"][-1] != "q1_kwh = capacity_kwh * clip(canonical_q07_kwh/capacity_kwh + capped_delta_cf, 0.0, 1.02)":
        raise AssertionError("delta transfer formula differs")
    gates = v1["historical_veto_2024"]["pass_gates"]
    if gates != {"all_seven_mixed_score_deltas_strictly_positive": True, "full_one_minus_nmae_delta_min": 0.0, "full_ficr_delta_strictly_positive": True}:
        raise AssertionError("historical veto gates differ")
    output = v1["execution_and_output_contract"]
    if ROOT / output["output_root"] != DEFAULT_OUT or output["csv_name"] != "feature_exit_nonwind_cap025143_scale097_overlay_2025.csv":
        raise AssertionError("output contract differs")
    base = v1["final_base_contract"]["prediction"]
    if (ROOT / base["path"], int(base["bytes"]), base["sha256"]) != FINAL_BASE:
        raise AssertionError("final base contract differs")
    stress_reference = v2["correction"]["effective_original_gate_A0_reference"]
    if (ROOT / stress_reference["path"], int(stress_reference["bytes"]), stress_reference["sha256"]) != GATE_REFERENCE:
        raise AssertionError("corrected stress reference differs")
    gate_recipe = v2["correction"]["effective_original_gate_recipe"]
    if (ROOT / gate_recipe["path"], int(gate_recipe["bytes"]), gate_recipe["sha256"]) != GATE_RECIPE:
        raise AssertionError("corrected gate recipe differs")
    final_recipe = v1["canonical_component_contract"]["actual_generation_recipe_snapshot"]
    if (ROOT / final_recipe["path"], int(final_recipe["bytes"]), final_recipe["sha256"]) != FINAL_RECIPE:
        raise AssertionError("final recipe differs")
    if not v3["effective_science_head"]:
        raise AssertionError("v3 not effective")
    return v1, v2, v3


def _read_prefix_raw() -> tuple[bytes, dict[str, Any]]:
    with RAW_LABELS.open("rb", buffering=0) as stream:
        raw = stream.read(PREFIX_BYTES)
        position = stream.tell()
    if len(raw) != PREFIX_BYTES or position != PREFIX_BYTES or hashlib.sha256(raw).hexdigest() != PREFIX_SHA:
        raise AssertionError("bounded label prefix differs")
    return raw, {"bytes": len(raw), "sha256": PREFIX_SHA, "expected_rows": 17_520, "expected_end": str(STRESS_END), "parsed": False, "suffix_bytes_read": 0}


def _parse_prefix(raw: bytes) -> pd.DataFrame:
    if len(raw) != PREFIX_BYTES or hashlib.sha256(raw).hexdigest() != PREFIX_SHA:
        raise AssertionError("prefix changed between source lock and parse")
    frame = pd.read_csv(io.BytesIO(raw), encoding="utf-8-sig")
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.pop("kst_dtm"), errors="raise"), name="forecast_kst_dtm")
    frame = frame.astype(np.float64)
    expected = pd.date_range("2022-01-01 01:00:00", STRESS_END, freq="h", name="forecast_kst_dtm")
    if len(frame) != 17_520 or not frame.index.equals(expected):
        raise AssertionError("bounded label prefix rows/index differ")
    return frame


def _read_suffix_once(prefix_raw: bytes) -> tuple[pd.DataFrame, bytes, dict[str, Any]]:
    with RAW_LABELS.open("rb", buffering=0) as stream:
        stream.seek(PREFIX_BYTES, os.SEEK_SET)
        raw = stream.read(SUFFIX_BYTES)
        end = stream.tell()
        physical_size = os.fstat(stream.fileno()).st_size
    if len(raw) != SUFFIX_BYTES or end != FULL_LABEL_BYTES or physical_size != FULL_LABEL_BYTES or hashlib.sha256(raw).hexdigest() != SUFFIX_SHA:
        raise AssertionError("label suffix differs")
    if hashlib.sha256(prefix_raw + raw).hexdigest() != FULL_LABEL_SHA:
        raise AssertionError("in-memory full label identity differs")
    frame = pd.read_csv(io.BytesIO(raw), names=("kst_dtm", *TARGET_COLS), header=None)
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.pop("kst_dtm"), errors="raise"), name="forecast_kst_dtm")
    frame = frame.astype(np.float64)
    if len(frame) != 8_784 or not frame.index.equals(STRESS_INDEX):
        raise AssertionError("label suffix rows/index differ")
    return frame, raw, {"bytes": len(raw), "sha256": SUFFIX_SHA, "rows": len(frame), "start": str(frame.index.min()), "end": str(frame.index.max()), "disk_reads": 1, "physical_file_size_bytes": physical_size, "full_file_size_exact": physical_size == FULL_LABEL_BYTES, "full_sha_from_memory": FULL_LABEL_SHA}


def _read_cache(spec: tuple[Path, int, str], expected_index: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    _identity(spec)
    frame = pd.read_parquet(spec[0]).astype(np.float32, copy=False)
    frame.index = pd.DatetimeIndex(frame.index, name="forecast_kst_dtm")
    if expected_index is not None and not frame.index.equals(expected_index):
        raise AssertionError(f"cache index differs: {spec[0]}")
    if frame.shape[1] != 612 or not np.isfinite(frame.to_numpy()).all():
        raise AssertionError(f"cache values/schema differ: {spec[0]}")
    return frame


def _feature_contract(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    all_names = list(map(str, frame.columns))
    if _sha_lines(all_names) != EXPECTED_FEATURE_SHA:
        raise AssertionError("feature order differs")
    keep = list(kept_features(all_names, "nonwind_atmospheric"))
    if len(keep) != 398 or _sha_lines(keep) != EXPECTED_KEEP_SHA:
        raise AssertionError("non-wind feature subset differs")
    return all_names, keep


def _row_digest(index: pd.DatetimeIndex, target_cf: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(np.asarray(index.asi8, dtype="<i8").tobytes())
    h.update(np.asarray(target_cf, dtype="<f8").tobytes())
    return h.hexdigest()


def _fit_pair(group: str, features: pd.DataFrame, labels: pd.Series, apply: pd.DataFrame, all_names: list[str], keep: list[str], model_dir: Path, expected_count: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    capacity = float(CAPACITY_KWH[group])
    aligned = labels.reindex(features.index)
    mask = aligned.notna() & np.isfinite(aligned.to_numpy()) & (aligned >= 0.10 * capacity)
    index = features.index[mask]
    target_cf = aligned.loc[index].to_numpy(dtype=np.float64) / capacity
    if len(index) != expected_count:
        raise AssertionError(f"eligible count differs for {group}: {len(index)}")
    digest = _row_digest(index, target_cf)
    predictions: dict[str, np.ndarray] = {}
    model_records: dict[str, Any] = {}
    for kind, names in (("control", all_names), ("nonwind", keep)):
        model = LGBMRegressor(**PARAMS)
        model.fit(features.loc[index, names], target_cf)
        pred = np.asarray(model.predict(apply.loc[:, names]), dtype=np.float64)
        if not np.isfinite(pred).all():
            raise AssertionError("non-finite model prediction")
        path = model_dir / f"{group}__{kind}.joblib"
        feature_sha = _sha_lines(names)
        model_payload = {"group": group, "kind": kind, "features": names, "feature_names_sha256": feature_sha, "eligible_count": len(index), "eligible_index_target_digest": digest, "parameters": PARAMS, "model": model}
        _atomic_joblib(path, model_payload)
        r1 = joblib.load(path)
        p1 = np.asarray(r1["model"].predict(apply.loc[:, r1["features"]]), dtype=np.float64)
        r2 = joblib.load(path)
        p2 = np.asarray(r2["model"].predict(apply.loc[:, r2["features"]]), dtype=np.float64)
        if not np.array_equal(pred, p1) or not np.array_equal(pred, p2):
            raise AssertionError("model save/reload prediction differs")
        predictions[kind] = pred
        model_records[kind] = {"file": describe_file(path), "feature_count": len(names), "feature_names_sha256": feature_sha, "eligible_count": len(index), "eligible_index_target_digest": digest, "parameters": PARAMS, "reload_1_equal": True, "reload_2_equal": True}
    if model_records["control"]["eligible_index_target_digest"] != model_records["nonwind"]["eligible_index_target_digest"]:
        raise AssertionError("paired eligible rows differ")
    return predictions["control"], predictions["nonwind"], model_records


def _read_prediction(spec: tuple[Path, int, str], index: pd.DatetimeIndex) -> pd.DataFrame:
    _identity(spec)
    frame = pd.read_parquet(spec[0]).astype(np.float64)
    frame.index = pd.DatetimeIndex(frame.index, name="forecast_kst_dtm")
    if tuple(frame.columns) != tuple(TARGET_COLS) or not frame.index.equals(index) or not np.isfinite(frame.to_numpy()).all():
        raise AssertionError(f"prediction frame differs: {spec[0]}")
    return frame


def _prebin(components: Mapping[str, np.ndarray], group: str, recipe: Mapping[str, Any]) -> np.ndarray:
    ensemble = recipe["ensemble"]
    weights = ensemble["weights"][group]
    value = sum(float(weights[name]) * np.asarray(components[name], dtype=np.float64) for name in weights)
    affine = ensemble["affine"][group]
    value = float(affine["scale"]) * value + float(affine["bias_kwh"])
    clip = ensemble["clip"][group]
    return np.clip(value, float(clip["lower_capacity_fraction"]) * CAPACITY_KWH[group], float(clip["upper_capacity_fraction"]) * CAPACITY_KWH[group])


def _verify_A0_reconstruction(component_frames: Mapping[str, pd.DataFrame], recipe: Mapping[str, Any], reference: pd.DataFrame) -> dict[str, Any]:
    result = {}
    for group in TARGET_COLS:
        components = {name: frame[group].to_numpy(dtype=np.float64) for name, frame in component_frames.items()}
        rebuilt = assemble_locked_group(components, group=group, capacity_kwh=CAPACITY_KWH[group], ensemble=recipe["ensemble"])
        expected = reference[group].to_numpy(dtype=np.float64)
        max_abs = float(np.max(np.abs(rebuilt - expected)))
        six_decimal = bool(np.array_equal(np.round(rebuilt, 6), np.round(expected, 6)))
        if max_abs > 1e-9 or not six_decimal:
            raise AssertionError(f"pre-fit A0 reconstruction differs: {group} {max_abs}")
        result[group] = {"max_abs_kwh": max_abs, "six_decimal_exact": six_decimal}
    return result


def _assemble_delta(component_frames: Mapping[str, pd.DataFrame], control_cf: pd.DataFrame, exit_cf: pd.DataFrame, recipe: Mapping[str, Any], reference: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    delta = pd.DataFrame(index=reference.index, columns=TARGET_COLS, dtype=np.float64)
    candidate_q07 = pd.DataFrame(index=reference.index, columns=TARGET_COLS, dtype=np.float64)
    a0_frame = pd.DataFrame(index=reference.index, columns=TARGET_COLS, dtype=np.float64)
    a1_frame = pd.DataFrame(index=reference.index, columns=TARGET_COLS, dtype=np.float64)
    audit: dict[str, Any] = {}
    for group in TARGET_COLS:
        capacity = float(CAPACITY_KWH[group])
        canonical_q = component_frames["lgb_q07"][group].to_numpy(dtype=np.float64)
        control_vs_canonical = control_cf[group].to_numpy(dtype=np.float64) * capacity - canonical_q
        raw_dcf = exit_cf[group].to_numpy(dtype=np.float64) - control_cf[group].to_numpy(dtype=np.float64)
        capped = np.clip(raw_dcf, -CAP_CF, CAP_CF)
        q1 = capacity * np.clip(canonical_q / capacity + capped, 0.0, 1.02)
        candidate_q07[group] = q1
        c0 = {name: frame[group].to_numpy(dtype=np.float64) for name, frame in component_frames.items()}
        c1 = dict(c0)
        c1["lgb_q07"] = q1
        standard_a0 = assemble_locked_group(c0, group=group, capacity_kwh=capacity, ensemble=recipe["ensemble"])
        if group != "kpx_group_2":
            a1 = assemble_locked_group(c1, group=group, capacity_kwh=capacity, ensemble=recipe["ensemble"])
            crossings = 0
        else:
            z0 = _prebin(c0, group, recipe)
            z1 = _prebin(c1, group, recipe)
            bins = recipe["ensemble"]["power_bins"][group]
            edges = np.asarray(bins["edges_cf"], dtype=np.float64)
            offsets = np.asarray(bins["delta_kwh"], dtype=np.float64)
            b0 = np.searchsorted(edges[1:-1], z0 / capacity, side="right")
            b1 = np.searchsorted(edges[1:-1], z1 / capacity, side="right")
            lower = float(recipe["ensemble"]["clip"][group]["lower_capacity_fraction"]) * capacity
            upper = float(recipe["ensemble"]["clip"][group]["upper_capacity_fraction"]) * capacity
            rebuilt_a0 = np.clip(z0 + offsets[b0], lower, upper)
            if not np.array_equal(rebuilt_a0, standard_a0):
                raise AssertionError("G2 baseline-bin assembly differs")
            a1 = np.clip(z1 + offsets[b0], lower, upper)
            crossings = int(np.count_nonzero(b1 != b0))
        ref = reference[group].to_numpy(dtype=np.float64)
        max_error = float(np.max(np.abs(standard_a0 - ref)))
        if max_error > 1e-9 or not np.array_equal(np.round(standard_a0, 6), np.round(ref, 6)):
            raise AssertionError(f"A0 reference reconstruction differs: {group} {max_error}")
        delta[group] = a1 - standard_a0
        a0_frame[group] = standard_a0
        a1_frame[group] = a1
        audit[group] = {"cap_hits": int(np.count_nonzero(np.abs(raw_dcf) > CAP_CF)), "max_abs_raw_delta_cf": float(np.max(np.abs(raw_dcf))), "max_abs_capped_delta_cf": float(np.max(np.abs(capped))), "hypothetical_g2_bin_crossings": crossings, "A0_reference_max_abs_kwh": max_error, "A0_six_decimal_exact": True, "new_cf_control_vs_canonical_kwh_q07_report_only": {"mean_signed_kwh": float(np.mean(control_vs_canonical)), "mean_abs_kwh": float(np.mean(np.abs(control_vs_canonical))), "max_abs_kwh": float(np.max(np.abs(control_vs_canonical))), "used_by_gate_abort_formula_or_selection": False}}
    return delta, candidate_q07, a0_frame, a1_frame, audit


def _verify_candidate_durability(out: Path, lock_path: Path, output_paths: Mapping[str, Path], model_records: Mapping[str, Any]) -> dict[str, Any]:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    output_checks = {}
    for key, path in output_paths.items():
        observed = describe_file(path)
        expected = lock["outputs"][key]
        if observed["size_bytes"] != expected["size_bytes"] or observed["sha256"] != expected["sha256"]:
            raise AssertionError(f"post-lock output identity differs: {key}")
        frame = pd.read_parquet(path)
        if len(frame) != 8_784 or tuple(frame.columns) != tuple(TARGET_COLS):
            raise AssertionError(f"post-lock output schema differs: {key}")
        output_checks[key] = {"identity_equal": True, "reopened_rows": len(frame)}
    model_checks = {}
    for group, pair in model_records.items():
        model_checks[group] = {}
        for kind, record in pair.items():
            path = Path(record["file"]["path"])
            observed = describe_file(path)
            expected = record["file"]
            if observed["size_bytes"] != expected["size_bytes"] or observed["sha256"] != expected["sha256"]:
                raise AssertionError(f"post-lock model identity differs: {group}/{kind}")
            payload = joblib.load(path)
            if payload["group"] != group or payload["kind"] != kind or payload["eligible_count"] != record["eligible_count"] or payload["eligible_index_target_digest"] != record["eligible_index_target_digest"] or payload["feature_names_sha256"] != record["feature_names_sha256"] or payload["parameters"] != PARAMS:
                raise AssertionError("post-lock model metadata differs")
            observed_params = payload["model"].get_params()
            if any(observed_params.get(key) != value for key, value in PARAMS.items()):
                raise AssertionError("post-lock model parameters differ")
            model_checks[group][kind] = {"identity_equal": True, "joblib_reopened": True}
    result = {"created_utc": utc_now(), "candidate_lock": describe_file(lock_path), "outputs": output_checks, "models": model_checks, "all_reopened_and_rehashed": True}
    _atomic_json(out / "stress/candidate_durability_lock.json", result)
    return result


def _metric(frame: pd.DataFrame, labels: pd.DataFrame, mask: np.ndarray) -> dict[str, Any]:
    by_group = {}
    n = []
    f = []
    for group in TARGET_COLS:
        m = group_metrics(labels.loc[mask, group].to_numpy(np.float64), frame.loc[mask, group].to_numpy(np.float64), CAPACITY_KWH[group], group_name=group)
        by_group[group] = m.as_dict()
        n.append(m.one_minus_nmae)
        f.append(m.ficr)
    mean_n, mean_f = float(np.mean(n)), float(np.mean(f))
    return {"score": 0.5 * (mean_n + mean_f), "one_minus_nmae": mean_n, "ficr": mean_f, "by_group": by_group, "rows": int(np.count_nonzero(mask))}


def _veto_masks(index: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    return {
        "FULL": np.ones(len(index), dtype=bool),
        "H1": (index >= pd.Timestamp("2024-01-01 01:00:00")) & (index <= pd.Timestamp("2024-07-01 00:00:00")),
        "H2": (index >= pd.Timestamp("2024-07-01 01:00:00")) & (index <= pd.Timestamp("2025-01-01 00:00:00")),
        "Q1": (index >= pd.Timestamp("2024-01-01 01:00:00")) & (index <= pd.Timestamp("2024-04-01 00:00:00")),
        "Q2": (index >= pd.Timestamp("2024-04-01 01:00:00")) & (index <= pd.Timestamp("2024-07-01 00:00:00")),
        "Q3": (index >= pd.Timestamp("2024-07-01 01:00:00")) & (index <= pd.Timestamp("2024-10-01 00:00:00")),
        "Q4": (index >= pd.Timestamp("2024-10-01 01:00:00")) & (index <= pd.Timestamp("2025-01-01 00:00:00")),
    }


def _score_veto(base: pd.DataFrame, candidate: pd.DataFrame, labels: pd.DataFrame) -> dict[str, Any]:
    masks = _veto_masks(labels.index)
    slices = {}
    for name, mask in masks.items():
        b, c = _metric(base, labels, np.asarray(mask)), _metric(candidate, labels, np.asarray(mask))
        slices[name] = {"base": b, "candidate": c, "delta": {k: float(c[k] - b[k]) for k in ("score", "one_minus_nmae", "ficr")}}
    gates = {"all_seven_score_positive": all(slices[x]["delta"]["score"] > 0.0 for x in masks), "full_n_nonnegative": slices["FULL"]["delta"]["one_minus_nmae"] >= 0.0, "full_f_positive": slices["FULL"]["delta"]["ficr"] > 0.0}
    return {"slices": slices, "gates": gates, "passed": all(gates.values())}


def _copy_science(out: Path) -> list[dict[str, Any]]:
    records = []
    dst = out / "science"
    dst.mkdir(parents=True, exist_ok=False)
    for path, _, _ in SCIENCE_FILES:
        target = dst / path.name
        shutil.copyfile(path, target)
        records.append(describe_file(target))
    return records


def _copy_execution_lineage(out: Path, authorization: Mapping[str, Any]) -> dict[str, Any]:
    source = Path(authorization["file"]["path"])
    sidecar = Path(authorization["sidecar"]["path"])
    review = Path(authorization["independent_prelaunch_review"]["path"])
    destination = out / "provenance"
    records = {
        "execution_amendment": _atomic_copy(source, destination / source.name),
        "execution_amendment_sidecar": _atomic_copy(sidecar, destination / sidecar.name),
        "independent_prelaunch_review": _atomic_copy(review, destination / review.name),
    }
    if records["execution_amendment"]["sha256"] != authorization["file"]["sha256"]:
        raise AssertionError("copied execution amendment identity differs")
    return records


def _create_attempt(out: Path) -> dict[str, Any]:
    if out.exists() or ATTEMPT.exists():
        raise FileExistsError("single attempt/output already exists")
    ATTEMPT.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": 1, "experiment_id": EXPERIMENT_ID, "pid": os.getpid(), "token": uuid.uuid4().hex, "created_utc": utc_now(), "single_attempt": True}
    fd = os.open(ATTEMPT, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    out.mkdir(parents=True, exist_ok=False)
    return describe_file(ATTEMPT)


def _canonical_q07_replay(test_features: Mapping[str, pd.DataFrame], stored: pd.DataFrame) -> dict[str, Any]:
    _identity(FINAL_Q07_MODEL)
    payload = joblib.load(FINAL_Q07_MODEL[0])
    result = {}
    for group in TARGET_COLS:
        names = list(payload["feature_names"][group])
        pred = np.asarray(payload["models"][group].predict(test_features[group].loc[:, names]), dtype=np.float64)
        expected = stored[group].to_numpy(dtype=np.float64)
        if not np.array_equal(pred, expected):
            raise AssertionError(f"canonical q07 replay differs: {group}")
        result[group] = {"array_equal": True, "max_abs_kwh": 0.0}
    return result


def _write_csv(path: Path, sample: pd.DataFrame, prediction: pd.DataFrame) -> dict[str, Any]:
    values = prediction.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise AssertionError("final prediction contains non-finite values")
    for group in TARGET_COLS:
        group_values = prediction[group].to_numpy(dtype=np.float64)
        if np.any(group_values < 0.0) or np.any(group_values > 1.02 * CAPACITY_KWH[group]):
            raise AssertionError(f"final prediction is out of bounds: {group}")
    frame = sample.copy()
    frame.loc[:, list(TARGET_COLS)] = prediction.to_numpy(dtype=np.float64)
    text = frame.to_csv(index=False, float_format="%.6f", lineterminator="\n")
    raw = b"\xef\xbb\xbf" + text.encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with tmp.open("wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)
    observed = path.read_bytes()
    if observed != raw or not observed.startswith(b"\xef\xbb\xbf") or b"\r\n" in observed:
        raise AssertionError("CSV byte contract differs")
    reread = pd.read_csv(path, encoding="utf-8-sig", dtype={"forecast_id": "string", "forecast_kst_dtm": "string"})
    if tuple(reread.columns) != tuple(frame.columns) or len(reread) != 8760:
        raise AssertionError("CSV schema/rows differ")
    if not np.array_equal(reread.loc[:, list(TARGET_COLS)].to_numpy(np.float64), np.round(prediction.to_numpy(np.float64), 6)):
        raise AssertionError("CSV numeric roundtrip differs")
    if not reread["forecast_id"].astype("string").equals(frame["forecast_id"].astype("string").reset_index(drop=True)):
        raise AssertionError("CSV forecast_id roundtrip differs")
    if not reread["forecast_kst_dtm"].astype("string").equals(frame["forecast_kst_dtm"].astype("string").reset_index(drop=True)):
        raise AssertionError("CSV timestamp roundtrip differs")
    return {"file": describe_file(path), "bom": True, "lf": True, "rows": 8760, "decimal_places": 6, "text_roundtrip_exact": True}


def static_audit(out: Path) -> dict[str, Any]:
    v1, v2, v3 = _verify_science()
    if out.exists() or ATTEMPT.exists() or HEAVY_GUARD.exists():
        raise AssertionError("prelaunch zero-state differs")
    return {"verdict": "PASS", "experiment_id": EXPERIMENT_ID, "science_hashes": [x[2] for x in SCIENCE_FILES], "formula": v1["delta_transfer_formula"], "wrong_reference_rejected": v2["correction"]["incorrect_v1_stress_reference"]["must_not_be_used_by_this_experiment"], "access_order_steps": len(v3["effective_label_access_order"]), "output_absent": True, "attempt_absent": True, "guard_absent": True, "science_or_data_access": 0}


def run(out: Path, authorization: Mapping[str, Any]) -> int:
    v1, _, _ = _verify_science()
    attempt_record = _create_attempt(out)
    with HeavyGuard() as guard:
        events: list[dict[str, Any]] = []
        guard.verify()
        science_records = _copy_science(out)
        execution_lineage = _copy_execution_lineage(out, authorization)
        source_inputs: list[dict[str, Any]] = []
        source_inputs.extend(_identity(spec) for spec in SMALL_PROVENANCE)
        source_inputs.extend(_identity(spec) for spec in OFFICIAL_TRAIN_RAW)
        for spec in TRAIN_CACHES.values():
            source_inputs.append(_identity(spec))
        source_inputs.extend(_identity(x) for x in (*GATE_COMPONENTS.values(), GATE_RECIPE, FINAL_RECIPE, GATE_REFERENCE, STRESS_BASE))
        prefix_raw, prefix_audit = _read_prefix_raw()
        events.append({"sequence": 1, "event": "bounded_label_prefix_read", "created_utc": utc_now(), "bytes": PREFIX_BYTES, "suffix_reads_so_far": 0})
        deferred_test_raw = [
            {**_declared_identity(f"official_test_raw_{i}", spec), "hash_verified": False}
            for i, spec in enumerate(OFFICIAL_TEST_RAW, start=1)
        ]
        _atomic_json(out / "source_lock.json", {"created_utc": utc_now(), "science": science_records, "execution_lineage": execution_lineage, "attempt_lock": attempt_record, "verified_inputs": source_inputs, "label_prefix": prefix_audit, "deferred_full_label_sha": FULL_LABEL_SHA, "deferred_test_and_final_inputs": True, "deferred_official_test_raw": deferred_test_raw})
        events.append({"sequence": 2, "event": "source_lock_durable", "created_utc": utc_now(), "suffix_reads_so_far": 0})
        prefix_labels = _parse_prefix(prefix_raw)
        events.append({"sequence": 3, "event": "bounded_label_prefix_parsed_after_source_lock", "created_utc": utc_now(), "rows": len(prefix_labels), "suffix_reads_so_far": 0})
        train_features = {group: _read_cache(TRAIN_CACHES[group]) for group in TARGET_COLS}
        names, keep = _feature_contract(train_features[TARGET_COLS[0]])
        if any(list(map(str, train_features[g].columns)) != names for g in TARGET_COLS):
            raise AssertionError("group feature schemas differ")
        gate_recipe = json.loads(GATE_RECIPE[0].read_text(encoding="utf-8"))
        gate_components = {name: _read_prediction(spec, STRESS_INDEX) for name, spec in GATE_COMPONENTS.items()}
        gate_reference = _read_prediction(GATE_REFERENCE, STRESS_INDEX)
        stress_base = _read_prediction(STRESS_BASE, STRESS_INDEX)
        A0_preflight = _verify_A0_reconstruction(gate_components, gate_recipe, gate_reference)
        _atomic_json(out / "stress/A0_reconstruction_preflight.json", {"created_utc": utc_now(), "before_any_stress_fit": True, "groups": A0_preflight})
        events.append({"sequence": 4, "event": "original_A0_reconstruction_verified_before_fit", "created_utc": utc_now(), "stress_model_fits_so_far": 0, "suffix_reads_so_far": 0})
        stress_control = pd.DataFrame(index=STRESS_INDEX, columns=TARGET_COLS, dtype=np.float64)
        stress_exit = stress_control.copy()
        stress_models: dict[str, Any] = {}
        for group in TARGET_COLS:
            control, exit_, records = _fit_pair(group, train_features[group].loc[:STRESS_END], prefix_labels[group], train_features[group].loc[STRESS_INDEX], names, keep, out / "stress/models", STRESS_COUNTS[group])
            stress_control[group], stress_exit[group], stress_models[group] = control, exit_, records
        events.append({"sequence": 5, "event": "six_stress_models_double_reload_verified", "created_utc": utc_now(), "stress_model_fits": 6, "suffix_reads_so_far": 0})
        stress_delta, stress_q1, stress_a0, stress_a1, stress_assembly = _assemble_delta(gate_components, stress_control, stress_exit, gate_recipe, gate_reference)
        stress_candidate = pd.DataFrame(index=STRESS_INDEX, columns=TARGET_COLS, dtype=np.float64)
        for group in TARGET_COLS:
            stress_candidate[group] = np.clip(stress_base[group].to_numpy(np.float64) + stress_delta[group].to_numpy(np.float64), 0.0, 1.02 * CAPACITY_KWH[group])
        stress_paths = {
            "control_cf": out / "stress/predictions/paired_control_cf_2024.parquet",
            "exit_cf": out / "stress/predictions/paired_nonwind_cf_2024.parquet",
            "candidate_q07": out / "stress/predictions/candidate_q07_kwh_2024.parquet",
            "A0": out / "stress/predictions/A0_locked_v3_kwh_2024.parquet",
            "A1_safe": out / "stress/predictions/A1_safe_locked_v3_kwh_2024.parquet",
            "increment": out / "stress/predictions/component_increment_kwh_2024.parquet",
            "candidate": out / "stress/predictions/candidate_scale097_kwh_2024.parquet",
        }
        for key, frame in (("control_cf", stress_control), ("exit_cf", stress_exit), ("candidate_q07", stress_q1), ("A0", stress_a0), ("A1_safe", stress_a1), ("increment", stress_delta), ("candidate", stress_candidate)):
            _atomic_parquet(stress_paths[key], frame)
        candidate_lock = {"created_utc": utc_now(), "models": stress_models, "assembly": stress_assembly, "outputs": {k: describe_file(p) for k, p in stress_paths.items()}, "label_suffix_reads_before_lock": 0, "formula_sha": hashlib.sha256(json.dumps(v1["delta_transfer_formula"], sort_keys=True).encode()).hexdigest()}
        _atomic_json(out / "stress/candidate_lock.json", candidate_lock)
        events.append({"sequence": 6, "event": "stress_candidate_lock_durable", "created_utc": utc_now(), "suffix_reads_so_far": 0})
        durability = _verify_candidate_durability(out, out / "stress/candidate_lock.json", stress_paths, stress_models)
        events.append({"sequence": 7, "event": "candidate_models_predictions_and_lock_reopened_rehashed", "created_utc": utc_now(), "suffix_reads_so_far": 0, "durability_lock": describe_file(out / "stress/candidate_durability_lock.json")})
        suffix_labels, suffix_raw, suffix_audit = _read_suffix_once(prefix_raw)
        events.append({"sequence": 8, "event": "label_suffix_single_read_and_in_memory_full_hash", "created_utc": utc_now(), "bytes": SUFFIX_BYTES, "suffix_reads_total": 1})
        _atomic_json(out / "stress/score_label_access.json", suffix_audit)
        stress_result = _score_veto(stress_base, stress_candidate, suffix_labels)
        _atomic_json(out / "stress/results.json", stress_result)
        events.append({"sequence": 9, "event": "historical_veto_scored", "created_utc": utc_now(), "passed": bool(stress_result["passed"]), "suffix_reads_total": 1})
        if not stress_result["passed"]:
            _atomic_json(out / "stress/rejection.json", {"verdict": "REJECT_NO_FINAL_CSV", "created_utc": utc_now(), "gates": stress_result["gates"], "no_rescue": True})
            rejection_stress_manifest = {"artifact_type": f"{EXPERIMENT_ID}_stress", "created_utc": utc_now(), "verdict": "REJECT", "candidate_lock": describe_file(out / "stress/candidate_lock.json"), "durability_lock": describe_file(out / "stress/candidate_durability_lock.json"), "results": describe_file(out / "stress/results.json"), "rejection": describe_file(out / "stress/rejection.json"), "outputs": [describe_file(p) for p in (out / "stress").rglob("*") if p.is_file()]}
            _atomic_json(out / "stress/manifest.json", rejection_stress_manifest)
            access = {"events": events, "prefix_range_disk_reads": 1, "suffix_range_disk_reads": 1, "whole_file_single_stream_reads": 0, "full_file_identity_computed_from_two_in_memory_ranges": True, "stress_model_fits": 6, "final_model_fits": 0, "test_cache_files_opened": 0, "final_component_files_opened": 0, "sample_files_opened": 0, "csv_files_written": 0, "suffix_buffer_reused_for_final_fit": False}
            _atomic_json(out / "access_ledger.json", access)
            _atomic_json(out / "manifest.json", {"artifact_type": EXPERIMENT_ID, "verdict": "PASS_PROTOCOL_PERFORMANCE_REJECT", "created_utc": utc_now(), "execution_lineage": execution_lineage, "attempt_lock": attempt_record, "stress": stress_result, "access": access, "final_access": {"test_caches": 0, "final_components": 0, "sample": 0, "models": 0, "csv": 0}, "outputs": [describe_file(p) for p in out.rglob("*") if p.is_file()]})
            return 2
        promotion = {"verdict": "PASS", "created_utc": utc_now(), "candidate_lock": describe_file(out / "stress/candidate_lock.json"), "results": describe_file(out / "stress/results.json"), "suffix_access": describe_file(out / "stress/score_label_access.json"), "no_rescue": True}
        _atomic_json(out / "stress/promotion_lock.json", promotion)
        stress_manifest = {"artifact_type": f"{EXPERIMENT_ID}_stress", "created_utc": utc_now(), "verdict": "PASS", "candidate_lock": describe_file(out / "stress/candidate_lock.json"), "durability_lock": describe_file(out / "stress/candidate_durability_lock.json"), "results": describe_file(out / "stress/results.json"), "promotion_lock": describe_file(out / "stress/promotion_lock.json"), "outputs": [describe_file(p) for p in (out / "stress").rglob("*") if p.is_file()]}
        _atomic_json(out / "stress/manifest.json", stress_manifest)
        events.append({"sequence": 10, "event": "promotion_and_stress_manifest_durable", "created_utc": utc_now(), "suffix_reads_total": 1})
        full_labels = pd.concat([prefix_labels, suffix_labels])
        expected_full_index = pd.date_range("2022-01-01 01:00:00", "2025-01-01 00:00:00", freq="h", name="forecast_kst_dtm")
        if len(full_labels) != 26_304 or tuple(full_labels.columns) != tuple(TARGET_COLS) or not full_labels.index.equals(expected_full_index) or not full_labels.index.is_unique:
            raise AssertionError("full labels assembled from buffers differ")
        official_test_raw = [_identity(spec) for spec in OFFICIAL_TEST_RAW]
        test_features = {group: _read_cache(TEST_CACHES[group], TEST_INDEX) for group in TARGET_COLS}
        final_components = {name: _read_prediction(spec, TEST_INDEX) for name, spec in FINAL_COMPONENTS.items()}
        final_reference = _read_prediction(FINAL_REFERENCE, TEST_INDEX)
        final_base = _read_prediction(FINAL_BASE, TEST_INDEX)
        final_recipe = json.loads(FINAL_RECIPE[0].read_text(encoding="utf-8"))
        events.append({"sequence": 11, "event": "conditional_official_test_raw_test_caches_and_final_inputs_verified_and_read", "created_utc": utc_now(), "suffix_reads_total": 1, "official_test_raw": official_test_raw})
        canonical_replay = _canonical_q07_replay(test_features, final_components["lgb_q07"])
        final_control = pd.DataFrame(index=TEST_INDEX, columns=TARGET_COLS, dtype=np.float64)
        final_exit = final_control.copy()
        final_models: dict[str, Any] = {}
        for group in TARGET_COLS:
            control, exit_, records = _fit_pair(group, train_features[group], full_labels[group], test_features[group], names, keep, out / "final/models", FINAL_COUNTS[group])
            final_control[group], final_exit[group], final_models[group] = control, exit_, records
        events.append({"sequence": 12, "event": "six_final_models_double_reload_verified", "created_utc": utc_now(), "final_model_fits": 6, "suffix_reads_total": 1})
        final_delta, final_q1, final_a0, final_a1, final_assembly = _assemble_delta(final_components, final_control, final_exit, final_recipe, final_reference)
        final_prediction = pd.DataFrame(index=TEST_INDEX, columns=TARGET_COLS, dtype=np.float64)
        for group in TARGET_COLS:
            final_prediction[group] = np.clip(final_base[group].to_numpy(np.float64) + final_delta[group].to_numpy(np.float64), 0.0, 1.02 * CAPACITY_KWH[group])
        final_frames = {"paired_control_cf_2025.parquet": final_control, "paired_nonwind_cf_2025.parquet": final_exit, "candidate_q07_kwh_2025.parquet": final_q1, "A0_locked_v3_kwh_2025.parquet": final_a0, "A1_safe_locked_v3_kwh_2025.parquet": final_a1, "feature_exit_component_increment_kwh_2025.parquet": final_delta, "final_prediction_kwh_2025.parquet": final_prediction}
        for name, frame in final_frames.items():
            _atomic_parquet(out / "final/predictions" / name, frame)
        _identity(SAMPLE_IDENTITY)
        sample = pd.read_csv(SAMPLE, encoding="utf-8-sig", dtype={"forecast_id": "string", "forecast_kst_dtm": "string"})
        sample_times = pd.DatetimeIndex(pd.to_datetime(sample["forecast_kst_dtm"], errors="raise"), name="forecast_kst_dtm")
        if tuple(sample.columns) != ("forecast_id", "forecast_kst_dtm", *TARGET_COLS) or not sample_times.equals(TEST_INDEX):
            raise AssertionError("sample contract differs")
        csv_audit = _write_csv(out / "feature_exit_nonwind_cap025143_scale097_overlay_2025.csv", sample, final_prediction)
        events.append({"sequence": 13, "event": "final_csv_byte_roundtrip_verified", "created_utc": utc_now(), "suffix_reads_total": 1})
        guard.verify()
        access = {"events": events, "prefix_range_disk_reads": 1, "suffix_range_disk_reads": 1, "whole_file_single_stream_reads": 0, "full_file_identity_computed_from_two_in_memory_ranges": True, "stress_model_fits": 6, "final_model_fits": 6, "test_cache_files_opened": 3, "final_component_files_opened": 6, "sample_files_opened": 1, "csv_files_written": 1, "suffix_buffer_reused_for_final_fit": True}
        _atomic_json(out / "access_ledger.json", access)
        manifest = {"artifact_type": EXPERIMENT_ID, "verdict": "PASS_PROTOCOL_AND_HISTORICAL_VETO_CSV_READY", "created_utc": utc_now(), "execution_lineage": execution_lineage, "attempt_lock": attempt_record, "stress": stress_result, "promotion": promotion, "final": {"models": final_models, "canonical_q07_replay": canonical_replay, "assembly": final_assembly, "csv": csv_audit, "mean_abs_increment_kwh": {g: float(np.mean(np.abs(final_delta[g]))) for g in TARGET_COLS}, "max_abs_increment_kwh": {g: float(np.max(np.abs(final_delta[g]))) for g in TARGET_COLS}}, "access": access, "outputs": [describe_file(p) for p in out.rglob("*") if p.is_file()]}
        _atomic_json(out / "manifest.json", manifest)
        return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static-audit", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--execution-amendment", type=Path)
    args = parser.parse_args(argv)
    if args.static_audit == args.run:
        parser.error("choose exactly one of --static-audit or --run")
    if args.run and args.execution_amendment is None:
        parser.error("--run requires --execution-amendment")
    if args.static_audit and args.execution_amendment is not None:
        parser.error("--execution-amendment is valid only with --run")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.static_audit:
        print(json.dumps(static_audit(args.out_dir.resolve()), sort_keys=True))
        return 0
    if args.out_dir.resolve() != DEFAULT_OUT.resolve():
        raise AssertionError("--run requires the canonical frozen output directory")
    authorization = _verify_execution_authorization(args.execution_amendment.resolve(), args.out_dir.resolve())
    return run(args.out_dir.resolve(), authorization)


if __name__ == "__main__":
    raise SystemExit(main())
