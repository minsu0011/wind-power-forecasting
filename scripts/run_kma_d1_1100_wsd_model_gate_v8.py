"""V8/V9 path-and-availability wrapper for the frozen KMA model gate.

The only source transformation is the V9-authorized 0.995 -> 0.98 coverage
threshold and the distinct V8 gate-pass filename. Recipes, features, folds,
weights, gates and selector execute from the frozen runner unchanged.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT / "scripts/run_kma_d1_1100_wsd_model_gate.py"
V8_ROOT = PROJECT / "artifacts/final_submission_sprint_20260812/kma_d1_1100_v8"
MATERIALIZED = V8_ROOT / "wsd_typ01/materialized/KMA_D1_1100_WSD_HOURLY_2022_2024_V8.parquet"
MANIFEST = MATERIALIZED.parent / "MANIFEST_V8.json"
OUTPUT = V8_ROOT / "model_gate"
EXECUTION_CONFIG = PROJECT / "configs/kma_d1_1100_wsd_model_gate_execution_v8.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(path: Path) -> dict[str, object]:
    return {"path": path.relative_to(PROJECT).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def load_frozen():
    spec = importlib.util.spec_from_file_location("_kma_model_gate_v8_base", SOURCE)
    if spec is None:
        raise RuntimeError("cannot create V8 frozen-runner module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    text = SOURCE.read_text(encoding="utf-8")
    if text.count("0.995") != 2 or text.count('"MODEL_GATE_PASS.json"') != 1:
        raise RuntimeError("frozen model runner authorized replacement sites drifted")
    text = text.replace("0.995", "0.98").replace('"MODEL_GATE_PASS.json"', '"MODEL_GATE_PASS_V8.json"')
    exec(compile(text, str(SOURCE) + "::V8_THRESHOLD_AND_NAMESPACE", "exec"), module.__dict__)
    module.MATERIALIZED = MATERIALIZED
    module.MATERIALIZED_MANIFEST = MANIFEST
    module.DEFAULT_OUTPUT = OUTPUT
    return module


base = load_frozen()


def verify_execution_config() -> None:
    payload = json.loads(EXECUTION_CONFIG.read_text(encoding="ascii"))
    if payload.get("schema_version") != 8 or payload.get("runner") != identity(Path(__file__).resolve()):
        raise RuntimeError("V8 model execution config does not bind current runner")
    for declared in payload.get("bound_inputs", []):
        path = PROJECT / declared["path"]
        if identity(path) != declared:
            raise RuntimeError(f"V8 model execution input drift: {path.name}")
    frozen = payload.get("frozen_science", {})
    if not all(frozen.get(key) is True for key in ("recipes", "features", "parameters", "weights", "folds", "baselines", "metrics", "gates", "selector")):
        raise RuntimeError("V8 frozen science declaration incomplete")


def main() -> None:
    verify_execution_config()
    base.main()


if __name__ == "__main__":
    main()
