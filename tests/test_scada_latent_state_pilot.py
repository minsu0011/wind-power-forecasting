from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.run_scada_latent_state_causal_student_pilot import read_bounded_csv
from src.scada_latent_state_pilot import (
    STATES,
    assign_proxy_states,
    fit_upper_envelope,
    jensen_shannon,
    pava_nondecreasing,
    state_entropy,
)


def test_pava_is_nondecreasing_and_preserves_block_means() -> None:
    fitted = pava_nondecreasing([0.0, 0.6, 0.4, 0.9])
    np.testing.assert_allclose(fitted, [0.0, 0.5, 0.5, 0.9])
    assert np.all(np.diff(fitted) >= 0)


def test_upper_envelope_is_monotone() -> None:
    wind = np.repeat(np.arange(0.25, 10.0, 0.5), 40)
    power = np.clip((wind / 10.0) + 0.03 * np.sin(np.arange(len(wind))), 0, 1.05)
    curve = fit_upper_envelope(wind, power, minimum_bin_rows=30)
    assert curve.populated_bins == 20
    assert np.all(np.diff(curve.values) >= 0)
    assert np.all((curve.predict([0, 5, 40]) >= 0) & (curve.predict([0, 5, 40]) <= 1.05))


def test_proxy_states_distinguish_peer_outage_and_common_low() -> None:
    potential = np.full((3, 4), 0.8)
    wind = np.full((3, 4), 8.0)
    power = np.asarray([
        [0.8, 0.8, 0.8, 0.0],
        [0.32, 0.32, 0.32, 0.32],
        [0.8, 0.56, 0.8, 0.8],
    ])
    states, ratio, peer = assign_proxy_states(power, wind, potential)
    assert states[0, 3] == "OUTAGE_LIKE"
    assert set(states[1]) == {"CURTAILMENT_LIKE"}
    assert states[2, 1] == "PARTIAL_DERATE"
    assert states[2, 0] == "NORMAL"
    assert np.isclose(ratio[0, 3], 0.0)
    assert peer[0, 3] >= 0.75
    assert set(np.unique(states)).issubset(STATES)


def test_distribution_and_entropy_identities() -> None:
    assert jensen_shannon([1, 1, 0], [1, 1, 0]) == 0.0
    assert jensen_shannon([1, 0], [0, 1]) > 0.6
    assert np.isclose(state_entropy([1, 1]), np.log(2.0))


def test_bounded_csv_does_not_parse_boundary_measurements(tmp_path: Path) -> None:
    path = tmp_path / "bounded.csv"
    path.write_bytes(
        b"kst_dtm,x\n"
        b"2023-01-01 00:00:00,1\n"
        b"2023-01-01 01:00:00,THIS_MUST_NOT_BE_PARSED\n"
        b"2024-01-01 00:00:00,SECRET\n"
    )
    frame, record = read_bounded_csv(path, pd.Timestamp("2023-01-01 01:00:00"), ["kst_dtm", "x"])
    assert frame["x"].tolist() == [1]
    assert record["next_boundary_timestamp_only"] == "2023-01-01T01:00:00"
    assert record["next_boundary_measurement_cells_read"] == 0
    assert record["suffix_rows_parsed"] == 0


def test_runner_has_no_submission_or_test_reader() -> None:
    source = Path("scripts/run_scada_latent_state_causal_student_pilot.py").read_text(encoding="utf-8")
    assert "sample_submission" not in source
    assert "Downloads/open/test" not in source
    assert "submission_csv_files_created\": 0" in source
    assert "post_validation_rescues\": 0" in source
    assert "to_pandas(ignore_metadata=True)" in source
