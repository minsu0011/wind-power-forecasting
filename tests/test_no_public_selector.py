from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_phase_a_runners_have_no_fit_or_public_selection_api() -> None:
    for relative in (
        "scripts/run_evidence_closure_v2.py",
        "scripts/run_submission_lineage_v2.py",
        "src/evidence_claims.py",
        "src/submission_lineage.py",
    ):
        source = (ROOT / relative).read_text(encoding="utf-8")
        tree = ast.parse(source)
        called = {
            node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, (ast.Attribute, ast.Name))
        }
        assert "fit" not in called
        assert "predict" not in called
        assert "train" not in called
    assert "selector_eligible\": true" not in (
        ROOT / "configs" / "baram2026_evidence_first_v2.json"
    ).read_text(encoding="utf-8").lower()

