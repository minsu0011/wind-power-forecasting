"""Train a locked full-data recipe and create a validated DACON submission.

Examples:
  python scripts/train_final.py --raw-dir C:/data/open --cache-dir artifacts/cache
      --out-dir artifacts/final --config configs/train_final.example.json --dry-run

  python scripts/train_final.py --raw-dir C:/data/open --cache-dir artifacts/cache
      --out-dir artifacts/final --config configs/train_final.v2.locked.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.final_training import (  # noqa: E402
    load_final_data,
    read_recipe,
    schema_summary,
    train_final,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir",
        type=Path,
        required=True,
        help="official data root containing train_labels.csv and sample_submission.csv",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        required=True,
        help="weather cache directory produced by scripts/build_features.py",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="destination for fitted models, predictions, submission, and manifest",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="locked final-recipe JSON; use the example only with --dry-run",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate config/raw/cache/submission schemas without fitting or writing",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="atomically replace outputs with the same recipe name",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    raw_dir = args.raw_dir.expanduser().resolve()
    cache_dir = args.cache_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    config = args.config.expanduser().resolve()

    if args.dry_run:
        recipe = read_recipe(config)
        data = load_final_data(raw_dir, cache_dir)
        print(
            json.dumps(
                schema_summary(recipe, data),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    result = train_final(
        raw_dir=raw_dir,
        cache_dir=cache_dir,
        out_dir=out_dir,
        recipe_path=config,
        overwrite=bool(args.overwrite),
        project_dir=PROJECT_DIR,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
