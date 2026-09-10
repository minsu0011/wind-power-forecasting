"""Portable path discovery for the breakthrough experiment."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    repo: Path
    data: Path
    artifacts: Path
    experiment: Path


def _is_data_root(path: Path) -> bool:
    required = (
        path / "train" / "train_labels.csv",
        path / "train" / "ldaps_train.csv",
        path / "train" / "gfs_train.csv",
        path / "train" / "scada_vestas_train.csv",
        path / "train" / "scada_unison_train.csv",
        path / "test" / "ldaps_test.csv",
        path / "test" / "gfs_test.csv",
        path / "sample_submission.csv",
        path / "info.xlsx",
    )
    return all(item.is_file() for item in required)


def discover_paths(
    repo: str | Path | None = None,
    data: str | Path | None = None,
) -> ProjectPaths:
    repo_path = (
        Path(repo).expanduser().resolve()
        if repo is not None
        else Path(__file__).resolve().parents[2]
    )
    candidates: list[Path] = []
    if data is not None:
        candidates.append(Path(data).expanduser())
    env_data = os.environ.get("BARAM_DATA_ROOT")
    if env_data:
        candidates.append(Path(env_data).expanduser())
    candidates.extend(
        (
            repo_path / "data",
            repo_path.parent / "open",
            Path.home() / "Downloads" / "open",
        )
    )
    for candidate in candidates:
        resolved = candidate.resolve()
        if _is_data_root(resolved):
            return ProjectPaths(
                repo=repo_path,
                data=resolved,
                artifacts=repo_path / "artifacts",
                experiment=repo_path / "artifacts" / "breakthrough_v1",
            )
    rendered = "\n".join(f"- {item}" for item in candidates)
    raise FileNotFoundError(f"Could not discover BARAM data root. Checked:\n{rendered}")
