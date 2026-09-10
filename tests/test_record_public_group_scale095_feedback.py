from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

from scripts import record_public_group_scale095_feedback as recorder


def _metric(delta: str) -> dict[str, str]:
    shift = Decimal(delta)
    return {
        "score": format(recorder.BASE["score"] + shift, "f"),
        "one_minus_nmae": format(recorder.BASE["one_minus_nmae"] + shift, "f"),
        "ficr": format(recorder.BASE["ficr"] + shift, "f"),
    }


def test_decimal_third_group_inference_is_exact_and_positive_rule_is_strict() -> None:
    decision = recorder.infer_group_deltas(
        {
            "kpx_group_1": _metric("0.001"),
            "kpx_group_2": _metric("-0.0002"),
        }
    )
    assert decision["inferred_group"] == "kpx_group_3"
    assert decision["positive_groups_strict_total_delta"] == ["kpx_group_1"]
    deltas = decision["group_deltas_decimal"]
    for component in recorder.COMPONENTS:
        observed_sum = sum((Decimal(deltas[group][component]) for group in recorder.TARGET_COLS), Decimal(0))
        assert observed_sum == Decimal(decision["global_delta_decimal"][component])
    assert decision["decimal_additivity_exact"] is True
    assert decision["automatic_final_recommendation"] is False


@pytest.mark.parametrize(
    ("groups", "missing"),
    [
        (("kpx_group_1", "kpx_group_3"), "kpx_group_2"),
        (("kpx_group_2", "kpx_group_3"), "kpx_group_1"),
    ],
)
def test_any_two_distinct_groups_can_identify_the_third(groups: tuple[str, str], missing: str) -> None:
    decision = recorder.infer_group_deltas({groups[0]: _metric("0.0001"), groups[1]: _metric("0.0002")})
    assert decision["inferred_group"] == missing


def test_recorder_rejects_any_count_other_than_two() -> None:
    with pytest.raises(ValueError, match="exactly two"):
        recorder.infer_group_deltas({"kpx_group_1": _metric("0.001")})
    with pytest.raises(ValueError, match="exactly two"):
        recorder.infer_group_deltas({group: _metric("0.001") for group in recorder.TARGET_COLS})


def test_metric_identity_is_validated() -> None:
    bad = _metric("0.001")
    bad["score"] = "0.9"
    with pytest.raises(ValueError, match="score identity"):
        recorder.validate_metric_triplet(bad, label="bad")


def test_recorder_is_bound_to_frozen_config_and_default_pack_namespace() -> None:
    assert recorder.CONFIG_PATH.name == "public_group_scale095_probe_20260808.json"
    assert recorder.PACK_MANIFEST.as_posix().endswith("artifacts/postgate/public_group_scale095_probe/manifest.json")
    config = recorder._verify_config()
    assert config["future_feedback_recorder"]["automatic_final_recommendation_even_after_hybrid"] is False
    assert config["future_feedback_recorder"]["no_hybrid_before_two_feedback_results"] is True

