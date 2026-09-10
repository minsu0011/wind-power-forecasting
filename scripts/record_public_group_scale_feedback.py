"""Validate Public feedback and record the frozen group-scale decision state.

This command never uploads a submission and never mutates prediction artifacts.
It accepts a JSON feedback snapshot and atomically creates a new decision
directory.  The output is explicitly Public-adaptive and selection-unsafe.

Input schema::

    {
      "schema_version": 1,
      "submissions": [
        {
          "label": "base",
          "file": "artifacts/final_cf_fix/corrected_recent_v4.csv",
          "sha256": "...",
          "score": 0.6122309211,
          "one_minus_nmae": 0.8518461479,
          "ficr": 0.3726156943
        },
        {"label": "global092", "file": "...", "sha256": "...", "score": 0.0,
         "one_minus_nmae": 0.0, "ficr": 0.0}
      ]
    }

``base`` and ``global092`` are required.  ``g3_only`` and then ``g1_only``
may be appended only in that frozen order after the global activation gate.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs/public_group_scale_probe_20260808.json"
CONFIG_SHA256 = "db6011f63ab18461c69df5de79e956bfe137001b7bb58b048f3652d33723b65e"
GROUP_MANIFEST_PATH = (
    PROJECT_ROOT / "artifacts/postgate/public_group_scale_probe/manifest.json"
)
GROUP_MANIFEST_SHA256 = (
    "62a0fc5561020d986f169cc68f83fc48d51750134c746eae5ef7d31b7a69e28c"
)
METRIC_IDENTITY_ATOL = 5e-10
BASE_PUBLIC = {
    "score": 0.6122309211,
    "one_minus_nmae": 0.8518461479,
    "ficr": 0.3726156943,
}
LABEL_ORDER = ("base", "global092", "g3_only", "g1_only")
COMPONENTS = ("score", "one_minus_nmae", "ficr")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feedback", type=Path, required=True)
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="A new, nonexistent directory; no overwrite option is provided.",
    )
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "size_bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def read_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"{path}: JSON root must be an object")
    return value


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_known_submissions() -> tuple[dict[str, dict[str, str]], dict[str, Any]]:
    if sha256_file(CONFIG_PATH) != CONFIG_SHA256:
        raise AssertionError("frozen group-scale config hash changed")
    if sha256_file(GROUP_MANIFEST_PATH) != GROUP_MANIFEST_SHA256:
        raise AssertionError("canonical group-scale manifest hash changed")
    config = read_object(CONFIG_PATH)
    manifest = read_object(GROUP_MANIFEST_PATH)
    if not config.get("public_adaptive") or not config.get("selection_unsafe"):
        raise AssertionError("frozen adaptive/unsafe flags changed")
    if config.get("private_champion") is not False:
        raise AssertionError("frozen private_champion flag changed")
    protocol = config["sequential_protocol"]
    if protocol["if_gate_passes_group_only_submission_order"] != [
        "kpx_group_3",
        "kpx_group_1",
    ]:
        raise AssertionError("frozen group-only order changed")
    if protocol["inferred_unsubmitted_group"] != "kpx_group_2":
        raise AssertionError("frozen inferred group changed")

    output_hashes = {
        Path(item["path"]).name: str(item["sha256"])
        for item in manifest["outputs"]
    }
    output_dir = resolve_project_path(config["output_directory"])
    g3_name = config["output_names"]["kpx_group_3"] + ".csv"
    g1_name = config["output_names"]["kpx_group_1"] + ".csv"
    known = {
        "base": {
            "file": str(resolve_project_path(config["base_submission"])),
            "sha256": str(config["base_submission_sha256"]),
        },
        "global092": {
            "file": str(resolve_project_path(protocol["global_probe"])),
            "sha256": str(protocol["global_probe_sha256"]),
        },
        "g3_only": {
            "file": str((output_dir / g3_name).resolve()),
            "sha256": output_hashes[g3_name],
        },
        "g1_only": {
            "file": str((output_dir / g1_name).resolve()),
            "sha256": output_hashes[g1_name],
        },
    }
    return known, config


def validate_triplet(record: Mapping[str, Any], *, label: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for component in COMPONENTS:
        raw = record[component]
        if isinstance(raw, bool):
            raise TypeError(f"{label}: {component} must be numeric, not boolean")
        value = float(raw)
        if not math.isfinite(value):
            raise ValueError(f"{label}: {component} must be finite")
        if value < 0.0 or value > 1.0:
            raise ValueError(f"{label}: {component} must be within [0, 1]")
        values[component] = value
    expected = 0.5 * (values["one_minus_nmae"] + values["ficr"])
    residual = values["score"] - expected
    if abs(residual) > METRIC_IDENTITY_ATOL:
        raise ValueError(
            f"{label}: score is inconsistent with 0.5*(1-NMAE+FICR); "
            f"residual={residual:.17g}"
        )
    values["identity_residual"] = float(residual)
    return values


def validate_feedback(
    payload: Mapping[str, Any], known: Mapping[str, Mapping[str, str]]
) -> dict[str, dict[str, Any]]:
    allowed_root = {"schema_version", "submissions", "reported_at_kst", "source"}
    unknown_root = set(payload) - allowed_root
    if unknown_root:
        raise ValueError(f"unknown feedback fields: {sorted(unknown_root)}")
    if payload.get("schema_version") != 1:
        raise ValueError("feedback schema_version must equal 1")
    submissions = payload.get("submissions")
    if not isinstance(submissions, list):
        raise TypeError("submissions must be a list")
    if not 2 <= len(submissions) <= len(LABEL_ORDER):
        raise ValueError("feedback must contain two to four submissions")

    records: dict[str, dict[str, Any]] = {}
    observed_labels: list[str] = []
    seen_files: set[Path] = set()
    seen_hashes: set[str] = set()
    expected_fields = {"label", "file", "sha256", *COMPONENTS}
    for raw in submissions:
        if not isinstance(raw, dict):
            raise TypeError("each submission must be an object")
        if set(raw) != expected_fields:
            raise ValueError(
                "submission fields must be exactly " + ", ".join(sorted(expected_fields))
            )
        label = str(raw["label"])
        if label not in known:
            raise ValueError(f"unknown submission label: {label}")
        if label in records:
            raise ValueError(f"duplicate submission label: {label}")
        source = resolve_project_path(str(raw["file"]))
        supplied_hash = str(raw["sha256"]).lower()
        if source in seen_files or supplied_hash in seen_hashes:
            raise ValueError(f"duplicate submission artifact: {label}")
        expected = known[label]
        if source != Path(expected["file"]):
            raise ValueError(f"{label}: file does not match frozen artifact")
        if supplied_hash != expected["sha256"]:
            raise ValueError(f"{label}: supplied SHA does not match frozen artifact")
        if not source.is_file():
            raise FileNotFoundError(source)
        observed_hash = sha256_file(source)
        if observed_hash != expected["sha256"]:
            raise AssertionError(f"{label}: on-disk artifact SHA changed")
        metrics = validate_triplet(raw, label=label)
        records[label] = {
            "label": label,
            "file": str(source),
            "sha256": observed_hash,
            **metrics,
        }
        observed_labels.append(label)
        seen_files.add(source)
        seen_hashes.add(supplied_hash)

    labels = set(records)
    if not {"base", "global092"}.issubset(labels):
        raise ValueError("base and global092 feedback are required")
    if "g1_only" in labels and "g3_only" not in labels:
        raise ValueError("g1_only feedback cannot precede g3_only feedback")
    expected_prefix = list(LABEL_ORDER[: len(observed_labels)])
    if observed_labels != expected_prefix:
        raise ValueError(
            f"submission order must be the frozen prefix {expected_prefix}; "
            f"got {observed_labels}"
        )
    for component, expected in BASE_PUBLIC.items():
        if abs(records["base"][component] - expected) > METRIC_IDENTITY_ATOL:
            raise ValueError(f"base: {component} differs from the frozen Public result")
    return records


def component_delta(candidate: Mapping[str, Any], base: Mapping[str, Any]) -> dict[str, float]:
    return {
        component: float(candidate[component] - base[component])
        for component in COMPONENTS
    }


def make_decision(
    records: Mapping[str, Mapping[str, Any]],
    known: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    base = records["base"]
    global_record = records["global092"]
    global_delta = component_delta(global_record, base)
    frozen_base_score = float(BASE_PUBLIC["score"])
    gate_passed = bool(global_record["score"] > frozen_base_score)
    optional_present = {"g3_only", "g1_only"}.intersection(records)
    if not gate_passed and optional_present:
        raise ValueError("group-only feedback is forbidden when the global092 gate fails")

    group_deltas: dict[str, Any] = {}
    if "g3_only" in records:
        group_deltas["kpx_group_3"] = {
            "source": "observed_g3_only_minus_base",
            **component_delta(records["g3_only"], base),
        }
    if "g1_only" in records:
        group_deltas["kpx_group_1"] = {
            "source": "observed_g1_only_minus_base",
            **component_delta(records["g1_only"], base),
        }

    inferred = False
    if gate_passed and "g3_only" in records and "g1_only" in records:
        group_deltas["kpx_group_2"] = {
            "source": "inferred_by_macro_separability",
            **{
                component: float(
                    global_delta[component]
                    - group_deltas["kpx_group_3"][component]
                    - group_deltas["kpx_group_1"][component]
                )
                for component in COMPONENTS
            },
        }
        inferred = True

    if not gate_passed:
        stage = "global_gate_failed"
        next_label = None
    elif "g3_only" not in records:
        stage = "global_gate_passed"
        next_label = "g3_only"
    elif "g1_only" not in records:
        stage = "g3_observed"
        next_label = "g1_only"
    else:
        stage = "two_group_probes_observed_g2_inferred"
        next_label = None

    next_submission = None
    if next_label is not None:
        next_submission = {
            "label": next_label,
            "file": known[next_label]["file"],
            "sha256": known[next_label]["sha256"],
        }
    return {
        "stage": stage,
        "activation_gate": {
            "rule": "global092.score > frozen BASE_PUBLIC.score (strict)",
            "base_score": frozen_base_score,
            "global092_score": float(global_record["score"]),
            "passed": gate_passed,
        },
        "global_delta": global_delta,
        "next_submission": next_submission,
        "group_deltas": group_deltas,
        "g2_delta_inferred": inferred,
        "separability_identity": (
            "delta_global = delta_g1_only + delta_g2_only + delta_g3_only "
            "for score, one_minus_nmae, and ficr separately"
        ),
        "automatic_final_scale_choice": None,
        "warning": "This diagnostic must not automatically select a final scale or Private champion.",
    }


def create_output_directory(
    out_dir: Path,
    feedback_path: Path,
    decision_payload: Mapping[str, Any],
    provenance_inputs: Sequence[Path],
) -> Path:
    if out_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {out_dir}")
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{out_dir.name}.tmp-", dir=str(out_dir.parent))
    )
    try:
        feedback_copy = staging / "feedback_input.json"
        shutil.copyfile(feedback_path, feedback_copy)
        decision_path = staging / "decision.json"
        atomic_json(decision_path, decision_payload)
        feedback_copy_record = file_record(feedback_copy)
        feedback_copy_record["path"] = str((out_dir / feedback_copy.name).resolve())
        decision_record = file_record(decision_path)
        decision_record["path"] = str((out_dir / decision_path.name).resolve())
        manifest = {
            "schema_version": 1,
            "artifact_type": "baram_public_group_scale_feedback_decision_manifest",
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "public_adaptive": True,
            "selection_unsafe": True,
            "private_champion": False,
            "submission_performed": False,
            "upload_capability": False,
            "overwrite_guard": "output directory must not already exist",
            "inputs": [file_record(path) for path in provenance_inputs],
            "outputs": [feedback_copy_record, decision_record],
        }
        atomic_json(staging / "manifest.json", manifest)
        if out_dir.exists():
            raise FileExistsError(f"refusing race-time overwrite: {out_dir}")
        staging.rename(out_dir)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return out_dir / "manifest.json"


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    feedback_path = args.feedback.resolve()
    out_dir = args.out_dir.resolve()
    if not feedback_path.is_file():
        raise FileNotFoundError(feedback_path)
    if out_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {out_dir}")

    known, config = load_known_submissions()
    feedback = read_object(feedback_path)
    records = validate_feedback(feedback, known)
    decision = make_decision(records, known)
    payload = {
        "schema_version": 1,
        "artifact_type": "baram_public_group_scale_feedback_decision",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "public_adaptive": True,
        "selection_unsafe": True,
        "private_champion": False,
        "submission_performed": False,
        "upload_capability": False,
        "feedback_input": file_record(feedback_path),
        "frozen_protocol": {
            "config": file_record(CONFIG_PATH),
            "canonical_group_probe_manifest": file_record(GROUP_MANIFEST_PATH),
            "factor": float(config["factor"]),
            "group_only_order": ["g3_only", "g1_only"],
            "inferred_group": "kpx_group_2",
        },
        "validated_feedback": [records[label] for label in LABEL_ORDER if label in records],
        "decision": decision,
    }
    provenance = [
        feedback_path,
        CONFIG_PATH,
        GROUP_MANIFEST_PATH,
        Path(__file__).resolve(),
        *[Path(known[label]["file"]) for label in LABEL_ORDER if label in records],
    ]
    manifest_path = create_output_directory(out_dir, feedback_path, payload, provenance)
    print(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "stage": decision["stage"],
                "next_submission": decision["next_submission"],
                "submission_performed": False,
                "manifest_sha256": sha256_file(manifest_path),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
