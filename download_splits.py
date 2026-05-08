"""Download the CoLOR data_splits/ from Hugging Face Hub.

The shift simulation for cifar100, amazon_reviews, and sun397 is materialized as
CSV files (one directory per (seed, ood_class, ood_class_ratio, fraction_ood_class)
combination). Because the SUN397 splits alone are ~235MB, they are hosted on HF
rather than bundled in the git repo.

Default repo: ``Shravan25C/CoLOR-data-splits`` (dataset). Override via --repo-id.

Examples:
    python download_splits.py                              # all 3 datasets
    python download_splits.py --datasets cifar100 sun397
    python download_splits.py --output-dir /tmp/CoLOR/data_splits
"""
from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_REPO_ID = "Shravan25C/CoLOR-data-splits"
DEFAULT_DATASETS = ("cifar100", "amazon_reviews", "sun397")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help=f"Hugging Face dataset repo id (default: {DEFAULT_REPO_ID})",
    )
    p.add_argument(
        "--datasets",
        nargs="+",
        default=list(DEFAULT_DATASETS),
        choices=list(DEFAULT_DATASETS),
        help="Which dataset's splits to fetch (default: all three).",
    )
    p.add_argument(
        "--output-dir",
        default="./data_splits",
        help="Local directory to place splits under (default: ./data_splits).",
    )
    p.add_argument(
        "--revision",
        default=None,
        help="Optional HF Hub revision/branch/tag.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is required. Install with: pip install huggingface-hub"
        ) from exc

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    allow_patterns = [f"{ds}/**" for ds in args.datasets]
    print(f"Downloading {args.datasets} from {args.repo_id} → {output_dir}")
    local_path = snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        revision=args.revision,
        local_dir=str(output_dir),
        allow_patterns=allow_patterns,
    )
    print(f"Done. Splits placed under: {local_path}")


if __name__ == "__main__":
    main()
