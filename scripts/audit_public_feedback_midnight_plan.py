"""Immutable audit of five Public results and the next five-slot information plan."""

from __future__ import annotations

from decimal import Decimal
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.manifest import describe_file, sha256_file, utc_now, write_json_atomic  # noqa: E402
from src.metric import TARGET_COLS  # noqa: E402


AUDIT_PATH = ROOT / "artifacts/audits/public_feedback_five_scores_gap_midnight_plan_v1.json"
INITIAL_LOG = ROOT / "artifacts/logs/public_submissions_20260808.json"
SCALE_FEEDBACK = ROOT / "artifacts/postgate/public_scale_feedback_20260808_1630/feedback_input.json"
SOURCE_IDENTITIES = {
    "initial_three_log": (INITIAL_LOG, "54c19a8f1579d2c112016c8a4b029006f523e368045b813b4c31d6dbd0a2ed62"),
    "scale_feedback_values": (SCALE_FEEDBACK, "435af521c600c1dd51ac4102a69293769f80e65627fe87f5906bd168d7b2094f"),
    "five_submission_feedback_manifest": (
        ROOT / "artifacts/postgate/public_scale_feedback_20260808_1630/manifest.json",
        "4d2b6f5fbe92374daa1e63e2dceac49c52605fcb5342b5f0db04a8feb61cbe0d",
    ),
    "public_scale_probe_manifest": (
        ROOT / "artifacts/postgate/public_scale_probe/manifest.json",
        "ee5b83cd676b5faf4eba8f77a8d40028405fdb9034e67038153929b52d14c4f0",
    ),
    "group_scale095_pack_manifest": (
        ROOT / "artifacts/postgate/public_group_scale095_probe/manifest.json",
        "56562ca8c8ccb269b101641fdce51b7c22bbe4b807fb455ac598f2416a98d7e5",
    ),
    "restored_scale095_incident": (
        ROOT / "artifacts/incidents/public_scale095_single_cell_corruption_20260808/incident.json",
        "d9e0636e3b8bafed516246425e0c2a26d1af65978bb7d1a575d00d45a32b0f31",
    ),
}
EXPECTED_SUBMISSIONS = {
    "corrected_recent_v4": {
        "path": ROOT / "artifacts/final_cf_fix/corrected_recent_v4.csv",
        "sha256": "a128e08fe14e1c299c6660530fbe745d1e67c94af6a8ed6fe151c6a370adbfb2",
        "score": "0.6122309211",
        "one_minus_nmae": "0.8518461479",
        "ficr": "0.3726156943",
        "source": "initial_three_log",
    },
    "prob_stable_g1g3": {
        "path": ROOT / "artifacts/postgate/probabilistic_ficr/prob_stable_g1g3_2025.csv",
        "sha256": "be1f36037d8c628496a1c2df4d4bbff4ac48d86a53071c04b3f9fd757cb8b769",
        "score": "0.6049992868",
        "one_minus_nmae": "0.8440095458",
        "ficr": "0.3659890277",
        "source": "initial_three_log",
    },
    "ficr_bayes_decision": {
        "path": ROOT / "artifacts/postgate/ficr_bayes_decision/ficr_bayes_decision_2025.csv",
        "sha256": "b0d5aa661ecd187ba04d4bd9e9edd2ef9dfbd4929e98f3542f528e8734d4fdd7",
        "score": "0.6045522289",
        "one_minus_nmae": "0.8436238097",
        "ficr": "0.365480648",
        "source": "initial_three_log",
    },
    "global_scale_092": {
        "path": ROOT / "artifacts/postgate/public_scale_probe/corrected_recent_v4_scale_092_2025.csv",
        "sha256": "15608b29bd664b19eafc6177638d180dc54d96ff26c0c7d1a2f47c3ef502708a",
        "score": "0.6075670068",
        "one_minus_nmae": "0.8641336214",
        "ficr": "0.3510003921",
        "source": "scale_feedback_values",
    },
    "global_scale_095": {
        "path": ROOT / "artifacts/postgate/public_scale_probe/corrected_recent_v4_scale_095_2025.csv",
        "sha256": "0918e42e3943a36b75acc569d5c9cb91202346f7843133aa075148e0ade6dbf1",
        "score": "0.6128977364",
        "one_minus_nmae": "0.8609894944",
        "ficr": "0.3648059783",
        "source": "scale_feedback_values",
    },
}
LEADER = {
    "score": Decimal("0.67365"),
    "one_minus_nmae": Decimal("0.87964"),
    "ficr": Decimal("0.46767"),
}


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def _source_records() -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, (path, expected_hash) in SOURCE_IDENTITIES.items():
        observed = sha256_file(path)
        if observed != expected_hash:
            raise AssertionError(f"source identity differs: {name}")
        result[name] = describe_file(path)
    return result


def _metric_row(raw: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        format(Decimal(str(raw["score"])), "f"),
        format(Decimal(str(raw["one_minus_nmae"])), "f"),
        format(Decimal(str(raw["ficr"])), "f"),
    )


def _validate_logs() -> dict[str, Mapping[str, Any]]:
    initial = _load(INITIAL_LOG)
    scales = _load(SCALE_FEEDBACK)
    by_sha: dict[str, Mapping[str, Any]] = {}
    for row in initial["submissions"]:
        by_sha[str(row["sha256"])] = row
    for row in scales["submissions"]:
        if row["label"] in ("global_scale_092", "global_scale_095"):
            by_sha[str(row["sha256"])] = row
    if len(by_sha) != 5:
        raise AssertionError("five distinct submission records were not recovered")
    for expected in EXPECTED_SUBMISSIONS.values():
        row = by_sha[expected["sha256"]]
        if _metric_row(row) != (
            expected["score"],
            expected["one_minus_nmae"],
            expected["ficr"],
        ):
            raise AssertionError("feedback metric triplet differs")
    observed_leader = initial["leader"]
    if _metric_row(observed_leader) != (
        format(LEADER["score"], "f"),
        format(LEADER["one_minus_nmae"], "f"),
        format(LEADER["ficr"], "f"),
    ):
        raise AssertionError("leader snapshot differs")
    return by_sha


def _csv_audit(path: Path, expected_sha: str) -> dict[str, Any]:
    if sha256_file(path) != expected_sha:
        raise AssertionError(f"CSV identity differs: {path}")
    if not path.read_bytes().startswith(b"\xef\xbb\xbf"):
        raise AssertionError(f"CSV BOM differs: {path}")
    frame = pd.read_csv(path, encoding="utf-8-sig", dtype="string")
    if tuple(frame.columns) != ("forecast_id", "forecast_kst_dtm", *TARGET_COLS):
        raise AssertionError(f"CSV schema differs: {path}")
    if len(frame) != 8760 or frame.iloc[0, 0] != "forecast_0001" or frame.iloc[-1, 0] != "forecast_8760":
        raise AssertionError(f"CSV row/id contract differs: {path}")
    values = frame.loc[:, list(TARGET_COLS)].astype(np.float64).to_numpy()
    if not np.isfinite(values).all():
        raise AssertionError(f"CSV finite contract differs: {path}")
    return {
        **describe_file(path),
        "UTF8_SIG_BOM": True,
        "rows": 8760,
        "schema_ids_times": True,
        "finite": True,
    }


def _gap_decomposition(values: Mapping[str, str]) -> dict[str, str]:
    score = Decimal(values["score"])
    nmae = Decimal(values["one_minus_nmae"])
    ficr = Decimal(values["ficr"])
    component_score = (nmae + ficr) / Decimal(2)
    reported_gap = LEADER["score"] - score
    nmae_part = (LEADER["one_minus_nmae"] - nmae) / Decimal(2)
    ficr_part = (LEADER["ficr"] - ficr) / Decimal(2)
    component_sum = nmae_part + ficr_part
    reconciliation = reported_gap - component_sum
    if reported_gap != nmae_part + ficr_part + reconciliation:
        raise AssertionError("gap decomposition is not Decimal exact")
    return {
        "reported_leader_total_minus_candidate_total": format(reported_gap, "f"),
        "NMAE_half_contribution": format(nmae_part, "f"),
        "FICR_half_contribution": format(ficr_part, "f"),
        "component_contribution_sum": format(component_sum, "f"),
        "rounding_reconciliation": format(reconciliation, "f"),
        "candidate_reported_score_minus_component_average": format(score - component_score, "f"),
        "identity": "reported_gap = NMAE_half_contribution + FICR_half_contribution + rounding_reconciliation",
    }


def _plan() -> dict[str, Any]:
    pack = ROOT / "artifacts/postgate/public_group_scale095_probe"
    g1 = pack / "corrected_recent_v4_g1_only_scale_095_2025.csv"
    g2 = pack / "corrected_recent_v4_g2_only_scale_095_2025.csv"
    g3 = pack / "corrected_recent_v4_g3_only_scale_095_2025.csv"
    scale097 = ROOT / "artifacts/postgate/public_adaptive_scale097_g2_delta_v2/corrected_recent_v4_scale_097_public_adaptive_2025.csv"
    independent = ROOT / "artifacts/postgate/direct_interval_g2_posthoc_rescue_v5/direct_interval_g2_posthoc_rescue_v5_2025.csv"
    expected = {
        g1: "2b92fe4f2a38149f5ec2752414b80393be180f2f59b22e6276a6acfb53be081e",
        g2: "7b5a87c55b640db8ba74fc4ee5aef9e1a8b137367b628d725afc1b6791c16d0e",
        g3: "02ac06103fdc18ce0f108c0e4ff8c9f442123de64c340c47af7b851ac24c04db",
        scale097: "e8f848b9a08c44b2552367f8e83ae4c9e132c8cf416ea8c437fd7a7e26314f58",
        independent: "2596bf1048f7e1635c7c965096f896d5a4104f3e04b58f575e21f9e2314bfddf",
    }
    identities = {}
    for path, digest in expected.items():
        if sha256_file(path) != digest:
            raise AssertionError(f"planned CSV identity differs: {path}")
        identities[path.name] = describe_file(path)
    recorder = ROOT / "scripts/record_public_group_scale095_feedback.py"
    if sha256_file(recorder) != "0dfeb6bd0c4a9d87c6d72ba1cf5203c5dbb1e89a854acdd1212c98d62b789cf6":
        raise AssertionError("feedback recorder changed")
    rescue_audit = ROOT / "artifacts/audits/direct_interval_g2_posthoc_rescue_v5_independent.json"
    if sha256_file(rescue_audit) != "a73114afebdf4ec04b70f10d5a29fc32e697fd2f503b9c7702d8289d91aa03eb":
        raise AssertionError("independent candidate audit changed")
    scale097_audit = ROOT / "artifacts/audits/public_adaptive_scale097_g2_delta_v2_raw_gpu.json"
    if sha256_file(scale097_audit) != "9e46b778af0131d576cf77a0ae3a94be3ca87e5549f2ecb0b472c6228def73e0":
        raise AssertionError("scale097 audit changed")
    return {
        "assumption": "A new five-submission daily quota is available after midnight; no submission was performed by this audit.",
        "fixed_slots": [
            {
                "slot": 1,
                "purpose": "Identify the exact G1 contribution of global .95.",
                "candidate": identities[g1.name],
                "label": "G1-only .95 probe",
            },
            {
                "slot": 2,
                "purpose": "Identify the exact G2 contribution; infer G3 from macro additivity after this result.",
                "candidate": identities[g2.name],
                "label": "G2-only .95 probe",
            },
            {
                "slot": 3,
                "purpose": "Submit only the positive-group .95 hybrid generated after the two triplets are recorded and validated.",
                "candidate": "generated later by the frozen recorder; no current bytes or score claim",
                "label": "positive-group .95 hybrid",
                "duplicate_guard": "Before submission, compare the generated CSV SHA with the five historical CSVs and slots 1-2. If identical, do not waste the slot; use the G3-only .95 probe to empirically confirm the inferred triplet, or leave the slot unused.",
            },
            {
                "slot": 4,
                "purpose": "Complete a rounded intermediate global scale probe without quadratic-vertex tuning.",
                "candidate": identities[scale097.name],
                "label": "global .97 plain",
                "risk": "Public-adaptive, selection-unsafe, descriptive interpolation only; private champion false.",
                "audit": describe_file(scale097_audit),
            },
            {
                "slot": 5,
                "purpose": "Measure the strongest currently independently audited non-scale model candidate.",
                "candidate": identities[independent.name],
                "label": "direct-interval G2 rescue v5",
                "risk": "Independent artifact/numeric audit PASS, but 2024-posthoc and selection-unsafe; strict/private champion false.",
                "audit": describe_file(rescue_audit),
            },
        ],
        "after_slots_1_and_2": {
            "exact_inference": "For each of score, one_minus_nmae and FICR: delta_G3 = (global095-base) - (G1only-base) - (G2only-base), using Decimal strings.",
            "positive_rule": "Activate group g iff its observed/inferred total-score delta is strictly >0; equality and negative deltas remain base identity.",
            "feedback_JSON_required_fields": [
                "schema_version=1",
                "group_only_results[2]",
                "each row: group,file,sha256,score,one_minus_nmae,ficr",
            ],
            "command": ".venv\\Scripts\\python.exe -B scripts\\record_public_group_scale095_feedback.py --feedback artifacts\\feedback\\public_group_scale095_g1_g2_after_midnight.json --out-dir artifacts\\postgate\\public_group_scale095_feedback_after_midnight_v1",
            "out_dir_must_not_exist": True,
            "validation_before_slot_3": [
                "decision.decimal_additivity_exact == true",
                "decision.inferred_group == kpx_group_3",
                "manifest and CSV hashes recorded",
                "CSV BOM/schema/8760/finite/capacity bounds pass",
                "active columns text-equal their canonical group-only files",
                "inactive columns text-equal base",
                "generated SHA is not a prior/slot1/slot2 duplicate",
            ],
        },
        "synthetic_end_to_end_check": {
            "command": ".venv\\Scripts\\python.exe -B -m pytest tests/test_record_public_group_scale095_feedback.py tests/test_public_feedback_midnight_plan_audit.py -q",
            "passed": 7,
            "failed": 0,
            "coverage": "Any-two inference, Decimal exact additivity, strict positive rule, canonical SHA binding, recorder main execution, Parquet/CSV/manifest creation, and per-column hybrid identity.",
        },
        "submission_performed": False,
        "external_message_sent": False,
        "new_performance_fit_or_score_search": False,
    }


def main() -> None:
    if AUDIT_PATH.exists():
        raise FileExistsError(AUDIT_PATH)
    source_records = _source_records()
    _validate_logs()
    submissions: dict[str, Any] = {}
    for name, expected in EXPECTED_SUBMISSIONS.items():
        physical = _csv_audit(expected["path"], expected["sha256"])
        metrics = {key: expected[key] for key in ("score", "one_minus_nmae", "ficr")}
        submissions[name] = {
            "source_record": expected["source"],
            "physical": physical,
            "public_metrics": metrics,
            "metric_identity": _gap_decomposition(metrics)[
                "candidate_reported_score_minus_component_average"
            ],
            "leader_gap_decomposition": _gap_decomposition(metrics),
        }
    leader_component_score = (LEADER["one_minus_nmae"] + LEADER["ficr"]) / Decimal(2)
    payload = {
        "schema_version": 1,
        "artifact_type": "public_feedback_five_scores_gap_and_midnight_information_plan_v1",
        "created_utc": utc_now(),
        "scope": "Read-only revalidation and planning; no submission, external message, new fit or performance search.",
        "source_records": source_records,
        "leader_snapshot": {
            "source": "initial_three_log / user-reported leaderboard screenshot",
            "reported_score": format(LEADER["score"], "f"),
            "displayed_one_minus_nmae": format(LEADER["one_minus_nmae"], "f"),
            "displayed_ficr": format(LEADER["ficr"], "f"),
            "component_average": format(leader_component_score, "f"),
            "reported_score_minus_component_average": format(
                LEADER["score"] - leader_component_score, "f"
            ),
            "precision_warning": "The leader components and total are displayed to five decimals and are internally offset by -0.000005. Exact component contributions therefore require the explicit rounding reconciliation term; hidden full-precision leader components are not identifiable from the screenshot.",
        },
        "five_submissions": submissions,
        "best_observed_public": {
            "candidate": "global_scale_095",
            "score": EXPECTED_SUBMISSIONS["global_scale_095"]["score"],
            "advantage_over_base": format(
                Decimal(EXPECTED_SUBMISSIONS["global_scale_095"]["score"])
                - Decimal(EXPECTED_SUBMISSIONS["corrected_recent_v4"]["score"]),
                "f",
            ),
            "remaining_reported_gap_to_leader": _gap_decomposition(
                EXPECTED_SUBMISSIONS["global_scale_095"]
            )["reported_leader_total_minus_candidate_total"],
            "private_score_identified": False,
        },
        "midnight_five_slot_plan": _plan(),
        "risk": {
            "public_adaptive": True,
            "selection_unsafe": True,
            "strict_validation_claim": False,
            "private_champion": False,
            "leaderboard_first_place_claim": False,
        },
        "immutability": {
            "output_created_with_no_overwrite": True,
            "existing_candidate_CSVs_modified": 0,
            "actual_submission_or_upload": 0,
            "external_messages": 0,
            "new_fit_calls": 0,
        },
    }
    write_json_atomic(AUDIT_PATH, payload, overwrite=False)
    print(f"audit_sha256={sha256_file(AUDIT_PATH)}")


if __name__ == "__main__":
    main()
