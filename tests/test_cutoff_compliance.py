from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.external_provenance import (
    GROUP_ORDER,
    SOURCE_ORDER,
    assess_cutoff_evidence,
    build_champion_provenance_rows,
    selected_previous_day,
)


ROOT = Path(__file__).resolve().parents[1]


def test_champion_hour_mask_is_exact_but_does_not_itself_prove_cutoff() -> None:
    hours = pd.date_range("2025-03-02 00:00", periods=24, freq="h")
    observed = [selected_previous_day(timestamp) for timestamp in hours]

    assert observed == [2] + [1] * 13 + [2] * 10
    missing_publication = assess_cutoff_evidence(
        "2025-03-02 13:00",
        1,
        exact_run_identifier=None,
        issue_time=None,
        publication_time=None,
        immutable_raw_sha_verified=True,
    )
    assert missing_publication.nominal_reference_before_cutoff is True
    assert missing_publication.cutoff_compliant_proven is False
    assert "exact_run_identifier" in missing_publication.reason
    assert "publication_time" in missing_publication.reason


def test_exact_run_issue_publication_and_raw_sha_are_all_required() -> None:
    passed = assess_cutoff_evidence(
        "2025-03-02 13:00",
        1,
        exact_run_identifier="ecmwf_20250301_00z",
        issue_time="2025-03-01T09:00:00+09:00",
        publication_time="2025-03-01T10:00:00+09:00",
        immutable_raw_sha_verified=True,
    )
    assert passed.cutoff_compliant_proven is True

    late = assess_cutoff_evidence(
        "2025-03-02 13:00",
        1,
        exact_run_identifier="ecmwf_20250301_00z",
        issue_time="2025-03-01T09:00:00+09:00",
        publication_time="2025-03-01T14:00:01+09:00",
        immutable_raw_sha_verified=True,
    )
    assert late.cutoff_compliant_proven is False
    assert late.publication_before_cutoff is False

    unbound_raw = assess_cutoff_evidence(
        "2025-03-02 13:00",
        1,
        exact_run_identifier="ecmwf_20250301_00z",
        issue_time="2025-03-01T09:00:00+09:00",
        publication_time="2025-03-01T10:00:00+09:00",
        immutable_raw_sha_verified=False,
    )
    assert unbound_raw.cutoff_compliant_proven is False


def test_bound_champion_sources_stop_when_publication_evidence_is_absent() -> None:
    frame, metadata = build_champion_provenance_rows(ROOT)

    assert metadata["rows"] == 18
    assert list(frame["source_id"].drop_duplicates()) == list(SOURCE_ORDER)
    assert list(frame["group"].drop_duplicates()) == list(GROUP_ORDER)
    assert not frame["cutoff_compliant_proven"].any()
    assert frame["cutoff_proven_rows"].sum() == 0
    assert not frame["exact_run_identifier_present"].any()
    assert not frame["exact_issue_time_present"].any()
    assert not frame["exact_publication_time_present"].any()
    assert set(frame["row_verdict"]) == {"STOP_PROVENANCE_FAILURE"}

    history = frame["archive_period"].eq("historical_training_2024")
    missing_history_raw = frame.loc[history & ~frame["raw_file_present"]]
    assert set(missing_history_raw["source_id"]) == {"ecmwf_ifs025", "icon_global"}
    assert len(missing_history_raw) == 6
    gfs_history = frame.loc[history & frame["source_id"].eq("gfs_global")]
    assert gfs_history["raw_sha256_verified"].all()
    assert gfs_history["download_time_utc"].isna().all()


def test_same_three_source_pipeline_is_not_present_for_2022_or_2023() -> None:
    frame, _ = build_champion_provenance_rows(ROOT)

    assert sorted(frame["calendar_year"].unique().tolist()) == [2024, 2025]
    assert not frame["same_pipeline_2022_through_2025_proven"].any()
    assert not frame["post_cutoff_or_reanalysis_exclusion_proven"].any()
