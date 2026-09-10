from pathlib import Path

from src.champion_reconstruction import assess_champion_reconstruction


ROOT = Path(__file__).resolve().parents[1]


def test_static_champion_reconstruction_stops_without_pre2024_three_source_history():
    result = assess_champion_reconstruction(ROOT)
    assert result["recipe_present"] is True
    assert result["source_manifest_identities_match"] is True
    assert result["source_evidence"]["icon_2022_probe_total_non_null"] == 0
    assert result["source_evidence"]["icon_2023_probe_total_non_null"] == 0
    assert result["source_evidence"]["ecmwf_request_years_in_manifest"] == ["2024"]
    assert result["source_evidence"]["gfs_request_years_in_manifest"] == ["2024"]
    assert result["causal_rolling_origin_oof_estimable"] is False
    assert result["decision"] == "STOP"


def test_scaffold_has_no_array_parser_or_model_fit_surface():
    source = (ROOT / "src/champion_reconstruction.py").read_text(encoding="utf-8")
    for forbidden in (
        "read_parquet(",
        "read_csv(",
        ".fit(",
        ".predict(",
        "import pandas",
        "import lightgbm",
        "import sklearn",
    ):
        assert forbidden not in source
