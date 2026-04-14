#!/usr/bin/env python
"""One-time sequential preprocessing for MovieLens-1M."""

import argparse
import sys
import json
import hashlib
from pathlib import Path
import yaml

# Add parent directory to path to enable recsys imports.
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from recsys.config import load_config
from recsys.data import (
    load_data_with_fallback,
    preprocess_sequential_ratings,
    save_sequential_preprocessing_artifacts,
    write_preprocessing_manifest,
)
from recsys.utils import print_environment_info


def _contract_hash(contract: dict) -> str:
    """Create a deterministic SHA256 hash for contract content."""
    canonical = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _load_contract(contract_path: Path) -> dict:
    """Load frozen experiment contract YAML."""
    if not contract_path.exists():
        raise FileNotFoundError(f"Contract file not found: {contract_path}")

    with open(contract_path, "r") as f:
        contract = yaml.safe_load(f)

    required_top = {"protocol_name", "protocol_version", "data", "sampling", "metrics"}
    missing = sorted(required_top - set(contract.keys()))
    if missing:
        raise ValueError(f"Contract missing required fields: {missing}")

    return contract


def _apply_contract_to_config(config, contract: dict) -> None:
    """Apply frozen contract values to runtime config."""
    data_cfg = contract["data"]
    sampling_cfg = contract["sampling"]

    config.data.dataset_name = data_cfg["dataset_name"]
    config.data.min_interactions_per_user = int(data_cfg["min_interactions_per_user"])
    config.data.min_interactions_per_item = int(data_cfg["min_interactions_per_item"])
    config.data.time_gap_num_buckets = int(data_cfg["time_gap_num_buckets"])
    if "positive_rating_threshold" in data_cfg:
        config.data.positive_rating_threshold = float(data_cfg["positive_rating_threshold"])
    config.model.history_length = int(data_cfg["sequence_length"])
    config.data.negative_samples_train = int(sampling_cfg["train_negatives_per_positive"])
    config.data.negative_samples_eval = int(sampling_cfg["eval_negatives_per_positive"])


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Preprocess MovieLens-1M into train/val/test sequential artifacts"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="local",
        choices=["local", "kaggle"],
        help="Configuration profile to load",
    )
    parser.add_argument(
        "--contract-path",
        type=str,
        default=str(Path(__file__).parent.parent / "config" / "experiment_contract.yaml"),
        help="Path to frozen experiment contract YAML",
    )
    parser.add_argument(
        "--allow-contract-overrides",
        action="store_true",
        help="Allow overriding contract values for debug runs",
    )
    parser.add_argument(
        "--subset",
        type=float,
        default=None,
        help="Optional data fraction for quick debug runs (e.g., 0.1)",
    )
    parser.add_argument(
        "--min-user-interactions",
        type=int,
        default=None,
        help="Minimum interactions per user for k-core filtering",
    )
    parser.add_argument(
        "--min-item-interactions",
        type=int,
        default=None,
        help="Minimum interactions per item for k-core filtering",
    )
    parser.add_argument(
        "--time-gap-buckets",
        type=int,
        default=None,
        help="Number of log-scale time-gap buckets",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save preprocessing artifacts",
    )
    return parser.parse_args()


def main() -> None:
    """Run one-time ML-1M preprocessing."""
    args = parse_args()

    config = load_config(args.config)
    contract_path = Path(args.contract_path)
    contract = _load_contract(contract_path)
    contract_sha = _contract_hash(contract)
    _apply_contract_to_config(config, contract)

    if not args.allow_contract_overrides and (
        args.min_user_interactions is not None
        or args.min_item_interactions is not None
        or args.time_gap_buckets is not None
    ):
        raise ValueError(
            "Contract is frozen. Use --allow-contract-overrides if you need debug overrides."
        )

    if args.subset is not None:
        config.data.data_subset = args.subset
    if args.allow_contract_overrides and args.min_user_interactions is not None:
        config.data.min_interactions_per_user = args.min_user_interactions
    if args.allow_contract_overrides and args.min_item_interactions is not None:
        config.data.min_interactions_per_item = args.min_item_interactions
    if args.allow_contract_overrides and args.time_gap_buckets is not None:
        config.data.time_gap_num_buckets = args.time_gap_buckets

    print(f"Using contract: {contract['protocol_name']} ({contract['protocol_version']})")
    print(f"Contract hash: {contract_sha}")

    print_environment_info(dataset_name=config.data.dataset_name)

    print("\nLoading MovieLens-1M ratings...")
    ratings, _ = load_data_with_fallback(
        data_path=config.paths.raw_data,
        subset=config.data.data_subset,
        auto_download=False,
        dataset_name=config.data.dataset_name,
    )

    train, val, test, encoder, gap_metadata = preprocess_sequential_ratings(
        ratings=ratings,
        min_user_interactions=config.data.min_interactions_per_user,
        min_item_interactions=config.data.min_interactions_per_item,
        time_gap_num_buckets=config.data.time_gap_num_buckets,
    )

    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else config.paths.processed_data / "ml-1m-sequential"
    )

    artifacts = save_sequential_preprocessing_artifacts(
        train=train,
        val=val,
        test=test,
        encoder=encoder,
        gap_metadata=gap_metadata,
        output_dir=output_dir,
    )

    preprocessing_params = {
        "config_profile": args.config,
        "dataset_name": config.data.dataset_name,
        "data_subset": config.data.data_subset,
        "min_interactions_per_user": config.data.min_interactions_per_user,
        "min_interactions_per_item": config.data.min_interactions_per_item,
        "time_gap_num_buckets": config.data.time_gap_num_buckets,
        "history_length": config.model.history_length,
        "negative_samples_train": config.data.negative_samples_train,
        "negative_samples_eval": config.data.negative_samples_eval,
        "contract_path": str(contract_path),
    }

    manifest_path = write_preprocessing_manifest(
        artifacts=artifacts,
        output_dir=output_dir,
        preprocessing_params=preprocessing_params,
        contract=contract,
        contract_hash=contract_sha,
    )
    artifacts["manifest"] = manifest_path

    print("\n" + "=" * 60)
    print("ML-1M PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"Users: {encoder.num_users:,}")
    print(f"Items: {encoder.num_items:,}")
    print(f"Train interactions: {len(train):,}")
    print(f"Val interactions: {len(val):,}")
    print(f"Test interactions: {len(test):,}")
    print("\nSaved artifacts:")
    for key, path in artifacts.items():
        print(f"  {key:14s}: {path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
