from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from scripts import run_full_weather_ordinal_bayes as runner
from src.full_weather_ordinal_bayes import MODEL_PARAMETERS, TRANSFER_WEIGHT


ROOT = Path(__file__).resolve().parents[1]


def test_preregister_and_feasibility_hashes_are_authoritative() -> None:
    preregister = ROOT / "configs/full_weather_ordinal_bayes_preregister_v1.json"
    feasibility = ROOT / "artifacts/audits/full_weather_ordinal_bayes_feasibility_v1.json"
    payload = runner._verify_preregister(preregister, feasibility)
    assert payload["classifier"]["parameters"] == MODEL_PARAMETERS
    assert payload["official_utility_action"]["fixed_transfer_weight"] == TRANSFER_WEIGHT
    assert payload["claims"]["candidate_fit_count_before_this_freeze"] == 0
    assert payload["claims"]["official_metric_count_before_this_freeze"] == 0


def test_default_namespace_is_new_and_stage1_only() -> None:
    args = runner.parse_args([])
    assert args.stage == "stage1"
    assert args.out_dir.as_posix().endswith(
        "artifacts/postgate/full_weather_ordinal_bayes_strict_v1"
    )
    assert args.preregister.name == "full_weather_ordinal_bayes_preregister_v1.json"


def test_label_specs_physically_omit_unneeded_targets() -> None:
    payload = json.loads(
        (ROOT / "configs/full_weather_ordinal_bayes_preregister_v1.json").read_text(
            encoding="utf-8"
        )
    )
    g12 = runner._label_spec(payload, "labels_g12_fit")
    g3 = runner._label_spec(payload, "labels_g3_fit")
    score = runner._label_spec(
        payload, "labels_stage1_application_after_prescore_lock"
    )
    assert g12["usecols"] == ["kst_dtm", "kpx_group_1", "kpx_group_2"]
    assert g12["next_row_first_field_only"] == "2023-01-01 01:00:00"
    assert g3["usecols"] == ["kst_dtm", "kpx_group_3"]
    assert g3["next_row_first_field_only"] == "2023-07-01 01:00:00"
    assert score["usecols"] == ["kst_dtm", "kpx_group_1", "kpx_group_2", "kpx_group_3"]
    assert score["next_row_first_field_only"] == "2024-01-01 01:00:00"


def test_reader_order_locks_candidate_before_application_label_decode() -> None:
    source = inspect.getsource(runner.run_stage1)
    assert source.index('global_lock = out_dir / "stage1_global_prescore_lock.json"') < source.index(
        "score_labels, score_evidence = bounded._bounded_label_prefix"
    )
    before_score = source[: source.index("score_labels, score_evidence")]
    assert '"application_target_value_cells_materialized": 0' in before_score
    assert '"public_or_scale_artifact_bytes_read": 0' in before_score
    assert "_fit_predict_group(" in before_score


def test_runner_contains_no_stage2_final_or_csv_execution() -> None:
    source = inspect.getsource(runner.run_stage1)
    assert "read_csv(" not in source
    assert "to_csv(" not in source
    assert '"stage2_opened": False' in source
    assert '"year_2024_value_bytes_read": 0' in source
    assert '"year_2025_value_bytes_read": 0' in source
    assert '"csv_files_created": 0' in source


def test_heavy_guard_is_exclusive_and_owner_only_release(tmp_path: Path) -> None:
    path = tmp_path / "heavy_cpu_fit.pid.json"
    owner = runner.acquire_heavy_guard(path)
    assert path.exists()
    with pytest.raises(RuntimeError, match="held by live PID"):
        runner.acquire_heavy_guard(path)
    runner.release_heavy_guard(path, {**owner, "token": "wrong"})
    assert path.exists()
    runner.release_heavy_guard(path, owner)
    assert not path.exists()


def test_static_import_closure_contains_core_runner_and_protocol() -> None:
    closure = runner.protocol._static_repo_import_closure((Path(runner.__file__).resolve(),))
    relative = {path.relative_to(ROOT).as_posix() for path in closure}
    assert "scripts/run_full_weather_ordinal_bayes.py" in relative
    assert "src/full_weather_ordinal_bayes.py" in relative
    assert "scripts/run_catboost_multiquantile_bayes.py" in relative
    assert "scripts/run_direct_interval_probability.py" in relative
    assert "src/features.py" in relative


def test_feasibility_records_median_support_and_mapping_smoke() -> None:
    payload = json.loads(
        (ROOT / "artifacts/audits/full_weather_ordinal_bayes_feasibility_v1.json").read_text(
            encoding="utf-8"
        )
    )
    support = payload["fixed_bin_support_from_permitted_fit_prefixes"]
    assert support["kpx_group_1"]["median_positive_class_count"] == 164.0
    assert support["kpx_group_2"]["median_positive_class_count"] == 139.0
    assert support["kpx_group_3"]["median_positive_class_count"] == 64.0
    smoke = payload["fixed_42_column_probability_mapping_smoke"]
    assert smoke["output_shape"] == [2, 42]
    assert smoke["row_sums"] == [1.0, 1.0]
    assert smoke["all_unobserved_columns_exact_zero"] is True
    assert smoke["candidate_fit_count_increment"] == 0
