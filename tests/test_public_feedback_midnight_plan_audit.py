from __future__ import annotations

from decimal import Decimal
import json
from pathlib import Path

import pandas as pd

from scripts import record_public_group_scale095_feedback as recorder
from src.manifest import sha256_file


def _triplet(delta: str) -> dict[str, str]:
    shift = Decimal(delta)
    return {
        name: format(recorder.BASE[name] + shift, "f")
        for name in recorder.COMPONENTS
    }


def test_two_feedback_rows_infer_third_exactly_and_build_expected_hybrid(
    tmp_path: Path,
) -> None:
    config = recorder._verify_config()
    known = recorder._known_group_csvs(config, recorder.PACK_MANIFEST)
    feedback = {
        "schema_version": 1,
        "group_only_results": [
            {
                "group": "kpx_group_1",
                "file": known["kpx_group_1"]["path"],
                "sha256": known["kpx_group_1"]["sha256"],
                **_triplet("0.001"),
            },
            {
                "group": "kpx_group_2",
                "file": known["kpx_group_2"]["path"],
                "sha256": known["kpx_group_2"]["sha256"],
                **_triplet("-0.001"),
            },
        ],
    }
    feedback_path = tmp_path / "synthetic_feedback.json"
    feedback_path.write_text(json.dumps(feedback, indent=2), encoding="utf-8")
    out_dir = tmp_path / "synthetic_hybrid"
    assert recorder.main(
        ["--feedback", str(feedback_path), "--out-dir", str(out_dir)]
    ) == 0

    decision = json.loads((out_dir / "decision.json").read_text(encoding="utf-8"))[
        "decision"
    ]
    assert decision["inferred_group"] == "kpx_group_3"
    assert decision["positive_groups_strict_total_delta"] == [
        "kpx_group_1",
        "kpx_group_3",
    ]
    for component in recorder.COMPONENTS:
        deltas = decision["group_deltas_decimal"]
        observed = sum(
            (Decimal(deltas[group][component]) for group in recorder.TARGET_COLS),
            Decimal(0),
        )
        assert observed == Decimal(decision["global_delta_decimal"][component])

    base_spec = config["input_identities"]["base_csv"]
    base = pd.read_csv(
        recorder.PROJECT_ROOT / base_spec["path"], encoding="utf-8-sig", dtype="string"
    )
    hybrid = pd.read_csv(
        out_dir / "positive_group_scale095_hybrid_2025.csv",
        encoding="utf-8-sig",
        dtype="string",
    )
    for group in recorder.TARGET_COLS:
        expected = (
            pd.read_csv(known[group]["path"], encoding="utf-8-sig", dtype="string")
            if group in decision["positive_groups_strict_total_delta"]
            else base
        )
        assert hybrid[group].equals(expected[group])
    assert (out_dir / "manifest.json").is_file()
    assert decision["decimal_additivity_exact"] is True
    assert decision["automatic_final_recommendation"] is False
    assert sha256_file(out_dir / "positive_group_scale095_hybrid_2025.csv")
