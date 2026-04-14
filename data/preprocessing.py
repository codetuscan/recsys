"""
Data preprocessing pipeline for MovieLens data.

Handles:
- ID encoding (user and movie IDs to dense indices)
- Train/test splitting (temporal)
- Data filtering
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional
from pathlib import Path
import pickle
import json
import hashlib
from datetime import datetime, timezone


class IDEncoder:
    """Encode sparse IDs to dense indices consistently across train/test."""

    def __init__(self):
        self.user_encoder = {}
        self.item_encoder = {}
        self.user_decoder = {}
        self.item_decoder = {}
        self.num_users = 0
        self.num_items = 0

    def fit(self, user_ids: np.ndarray, item_ids: np.ndarray):
        """
        Fit encoders on training data.

        Args:
            user_ids: Array of user IDs
            item_ids: Array of item IDs
        """
        # Create user encoding
        unique_users = sorted(np.unique(user_ids))
        self.user_encoder = {uid: idx for idx, uid in enumerate(unique_users)}
        self.user_decoder = {idx: uid for uid, idx in self.user_encoder.items()}
        self.num_users = len(unique_users)

        # Create item encoding
        unique_items = sorted(np.unique(item_ids))
        self.item_encoder = {iid: idx for idx, iid in enumerate(unique_items)}
        self.item_decoder = {idx: iid for iid, idx in self.item_encoder.items()}
        self.num_items = len(unique_items)

        print(f"IDEncoder fitted: {self.num_users} users, {self.num_items} items")

    def transform_users(self, user_ids: np.ndarray) -> np.ndarray:
        """Transform user IDs to indices."""
        return np.array([self.user_encoder.get(uid, -1) for uid in user_ids])

    def transform_items(self, item_ids: np.ndarray) -> np.ndarray:
        """Transform item IDs to indices."""
        return np.array([self.item_encoder.get(iid, -1) for iid in item_ids])

    def fit_transform(
        self, user_ids: np.ndarray, item_ids: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fit encoders and transform IDs."""
        self.fit(user_ids, item_ids)
        return self.transform_users(user_ids), self.transform_items(item_ids)

    def inverse_transform_users(self, user_indices: np.ndarray) -> np.ndarray:
        """Transform user indices back to original IDs."""
        return np.array([self.user_decoder.get(idx, -1) for idx in user_indices])

    def inverse_transform_items(self, item_indices: np.ndarray) -> np.ndarray:
        """Transform item indices back to original IDs."""
        return np.array([self.item_decoder.get(idx, -1) for idx in item_indices])

    def save(self, path: Path):
        """Save encoder to file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "user_encoder": self.user_encoder,
                    "item_encoder": self.item_encoder,
                    "user_decoder": self.user_decoder,
                    "item_decoder": self.item_decoder,
                    "num_users": self.num_users,
                    "num_items": self.num_items,
                },
                f,
            )
        print(f"IDEncoder saved to {path}")

    @classmethod
    def load(cls, path: Path):
        """Load encoder from file."""
        with open(path, "rb") as f:
            data = pickle.load(f)

        encoder = cls()
        encoder.user_encoder = data["user_encoder"]
        encoder.item_encoder = data["item_encoder"]
        encoder.user_decoder = data["user_decoder"]
        encoder.item_decoder = data["item_decoder"]
        encoder.num_users = data["num_users"]
        encoder.num_items = data["num_items"]

        print(f"IDEncoder loaded from {path}")
        return encoder


def temporal_train_test_split(
    ratings: pd.DataFrame, user_col: str = "userId", timestamp_col: str = "timestamp"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Temporal leave-one-out split: last interaction per user for test.

    Args:
        ratings: Ratings DataFrame
        user_col: Column name for user IDs
        timestamp_col: Column name for timestamps

    Returns:
        Tuple of (train DataFrame, test DataFrame)
    """
    print("Performing temporal train/test split (leave-one-out)...")

    # Sort by timestamp
    ratings = ratings.sort_values(timestamp_col).reset_index(drop=True)

    # Get last interaction per user for test
    test = ratings.groupby(user_col).tail(1).reset_index(drop=True)

    # Get all but last interaction per user for train
    train = ratings.groupby(user_col).head(-1).reset_index(drop=True)

    print(f"Train: {len(train):,} interactions")
    print(f"Test: {len(test):,} interactions")

    return train, test


def temporal_train_val_test_split(
    ratings: pd.DataFrame,
    user_col: str = "userId",
    timestamp_col: str = "timestamp",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Temporal split for sequential recommendation:
    - last interaction per user -> test
    - second-to-last interaction per user -> validation
    - remaining interactions -> train

    Args:
        ratings: Ratings DataFrame
        user_col: Column name for user IDs
        timestamp_col: Column name for timestamps

    Returns:
        Tuple of (train DataFrame, val DataFrame, test DataFrame)
    """
    print("Performing temporal train/val/test split...")

    sort_cols = [user_col, timestamp_col]
    if "movieId" in ratings.columns:
        sort_cols.append("movieId")

    ratings_sorted = ratings.sort_values(sort_cols).reset_index(drop=True)

    test = ratings_sorted.groupby(user_col, sort=False).tail(1)
    remainder = ratings_sorted.drop(index=test.index)
    val = remainder.groupby(user_col, sort=False).tail(1)
    train = remainder.drop(index=val.index)

    train = train.reset_index(drop=True)
    val = val.reset_index(drop=True)
    test = test.reset_index(drop=True)

    print(f"Train: {len(train):,} interactions")
    print(f"Val:   {len(val):,} interactions")
    print(f"Test:  {len(test):,} interactions")

    return train, val, test


def add_log_time_gap_buckets(
    ratings: pd.DataFrame,
    user_col: str = "userId",
    timestamp_col: str = "timestamp",
    num_buckets: int = 20,
    gap_col: str = "time_gap_seconds",
    bucket_col: str = "time_gap_bucket",
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """
    Compute consecutive time gaps per user and bucket them on a log scale.

    The first interaction for each user is assigned gap=0 and bucket=0.

    Args:
        ratings: Input ratings DataFrame
        user_col: Column name for user IDs
        timestamp_col: Column name for timestamps (unix seconds)
        num_buckets: Maximum number of positive-gap buckets
        gap_col: Output column name for raw gap in seconds
        bucket_col: Output column name for log-bucket index

    Returns:
        Tuple of:
        - ratings DataFrame with added time-gap columns
        - metadata dictionary with bucket edges and stats
    """
    if num_buckets < 1:
        raise ValueError(f"num_buckets must be >= 1, got {num_buckets}")

    ratings = ratings.sort_values([user_col, timestamp_col]).reset_index(drop=True).copy()

    gaps = ratings.groupby(user_col)[timestamp_col].diff().fillna(0).clip(lower=0)
    gaps = gaps.astype(np.int64)

    bucket_values = np.zeros(len(ratings), dtype=np.int16)
    positive_mask = gaps.values > 0

    effective_buckets = 0
    log_bin_edges = []

    if positive_mask.any():
        log_gaps = np.log1p(gaps.values[positive_mask].astype(np.float64))
        min_log = float(np.min(log_gaps))
        max_log = float(np.max(log_gaps))

        if np.isclose(min_log, max_log):
            bucket_values[positive_mask] = 1
            effective_buckets = 1
            log_bin_edges = [min_log, max_log]
        else:
            effective_buckets = min(num_buckets, int(np.sum(positive_mask)))
            edges = np.linspace(min_log, max_log, effective_buckets + 1)
            # Buckets are 1..effective_buckets for positive gaps.
            bucket_values[positive_mask] = (
                np.digitize(log_gaps, bins=edges[1:-1], right=False) + 1
            )
            log_bin_edges = edges.tolist()

    ratings[gap_col] = gaps
    ratings[bucket_col] = bucket_values

    metadata = {
        "requested_num_buckets": int(num_buckets),
        "effective_num_buckets": int(effective_buckets),
        "gap_column": gap_col,
        "bucket_column": bucket_col,
        "zero_gap_bucket": 0,
        "max_gap_seconds": int(gaps.max()) if len(gaps) > 0 else 0,
        "log_bin_edges": log_bin_edges,
    }

    print("Computed log-scale time-gap buckets")
    print(f"  Positive-gap interactions: {int(np.sum(positive_mask)):,}")
    print(f"  Effective buckets: {effective_buckets}")

    return ratings, metadata


def preprocess_sequential_ratings(
    ratings: pd.DataFrame,
    min_user_interactions: int = 5,
    min_item_interactions: int = 5,
    time_gap_num_buckets: int = 20,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, IDEncoder, Dict[str, object]]:
    """
    Sequential preprocessing pipeline for MovieLens-1M style data.

    Steps:
    1. k-core filtering (min interactions for users and items)
    2. Sort interactions by user and timestamp
    3. Compute log-scale time-gap buckets
    4. Temporal split into train/val/test
    5. Re-index users/items using train split

    Args:
        ratings: Raw ratings DataFrame
        min_user_interactions: k-core threshold for users
        min_item_interactions: k-core threshold for items
        time_gap_num_buckets: Number of log-scale time-gap buckets

    Returns:
        Tuple of:
        - train DataFrame
        - val DataFrame
        - test DataFrame
        - IDEncoder
        - time-gap metadata dict
    """
    print("\n" + "=" * 60)
    print("SEQUENTIAL PREPROCESSING PIPELINE")
    print("=" * 60)

    ratings = filter_sparse_users_items(
        ratings,
        min_user_interactions=min_user_interactions,
        min_item_interactions=min_item_interactions,
    )

    if ratings.empty:
        raise ValueError(
            "No ratings remain after k-core filtering for sequential preprocessing. "
            "Increase data subset size or lower min_interactions thresholds."
        )

    ratings, gap_metadata = add_log_time_gap_buckets(
        ratings,
        user_col="userId",
        timestamp_col="timestamp",
        num_buckets=time_gap_num_buckets,
    )

    print("\nEncoding IDs on full filtered dataset...")
    encoder = IDEncoder()

    user_idx, item_idx = encoder.fit_transform(
        ratings["userId"].values,
        ratings["movieId"].values,
    )
    ratings = ratings.copy()
    ratings["user_idx"] = user_idx
    ratings["item_idx"] = item_idx

    train, val, test = temporal_train_val_test_split(
        ratings,
        user_col="userId",
        timestamp_col="timestamp",
    )

    print("\n✓ Sequential preprocessing complete")
    print(f"  Train: {len(train):,} ratings")
    print(f"  Val:   {len(val):,} ratings")
    print(f"  Test:  {len(test):,} ratings")
    print(f"  Users: {encoder.num_users:,}")
    print(f"  Items: {encoder.num_items:,}")
    print("=" * 60 + "\n")

    return train, val, test, encoder, gap_metadata


def save_sequential_preprocessing_artifacts(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    encoder: IDEncoder,
    gap_metadata: Dict[str, object],
    output_dir: Path,
) -> Dict[str, Path]:
    """
    Persist sequential preprocessing outputs to disk.

    Saved artifacts:
    - train.csv.gz, val.csv.gz, test.csv.gz
    - id_encoder.pkl
    - train_user_histories.pkl
    - preprocessing_metadata.json
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train.csv.gz"
    val_path = output_dir / "val.csv.gz"
    test_path = output_dir / "test.csv.gz"
    encoder_path = output_dir / "id_encoder.pkl"
    histories_path = output_dir / "train_user_histories.pkl"
    metadata_path = output_dir / "preprocessing_metadata.json"

    train.to_csv(train_path, index=False, compression="gzip")
    val.to_csv(val_path, index=False, compression="gzip")
    test.to_csv(test_path, index=False, compression="gzip")
    encoder.save(encoder_path)

    train_histories = (
        train.sort_values(["user_idx", "timestamp"])
        .groupby("user_idx", sort=False)["item_idx"]
        .apply(list)
        .to_dict()
    )
    with open(histories_path, "wb") as f:
        pickle.dump(train_histories, f)

    metadata = {
        "num_users": int(encoder.num_users),
        "num_items": int(encoder.num_items),
        "num_train_interactions": int(len(train)),
        "num_val_interactions": int(len(val)),
        "num_test_interactions": int(len(test)),
        "time_gap": gap_metadata,
        "columns": {
            "user_id": "userId",
            "item_id": "movieId",
            "rating": "rating",
            "timestamp": "timestamp",
            "user_idx": "user_idx",
            "item_idx": "item_idx",
            "time_gap_seconds": "time_gap_seconds",
            "time_gap_bucket": "time_gap_bucket",
        },
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved preprocessing artifacts to {output_dir}")

    return {
        "train": train_path,
        "val": val_path,
        "test": test_path,
        "encoder": encoder_path,
        "train_histories": histories_path,
        "metadata": metadata_path,
    }


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute SHA256 hash for a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def write_preprocessing_manifest(
    artifacts: Dict[str, Path],
    output_dir: Path,
    preprocessing_params: Dict[str, object],
    contract: Optional[Dict[str, object]] = None,
    contract_hash: Optional[str] = None,
) -> Path:
    """
    Write reproducibility manifest for preprocessing artifacts.

    The manifest records:
    - preprocessing parameters
    - optional frozen contract and hash
    - file size and SHA256 per artifact
    """
    manifest_path = output_dir / "artifact_manifest.json"

    artifact_entries = {}
    for key, path in artifacts.items():
        path_obj = Path(path)
        artifact_entries[key] = {
            "path": str(path_obj),
            "size_bytes": int(path_obj.stat().st_size),
            "sha256": _sha256_file(path_obj),
        }

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "preprocessing_params": preprocessing_params,
        "contract": contract,
        "contract_hash": contract_hash,
        "artifacts": artifact_entries,
    }

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Saved artifact manifest to {manifest_path}")

    return manifest_path


def filter_sparse_users_items(
    ratings: pd.DataFrame,
    min_user_interactions: int = 5,
    min_item_interactions: int = 5,
    user_col: str = "userId",
    item_col: str = "movieId",
) -> pd.DataFrame:
    """
    Filter out users and items with too few interactions.

    Args:
        ratings: Ratings DataFrame
        min_user_interactions: Minimum interactions per user
        min_item_interactions: Minimum interactions per item
        user_col: Column name for user IDs
        item_col: Column name for item IDs

    Returns:
        Filtered ratings DataFrame
    """
    print(f"Filtering sparse users/items...")
    print(f"  Before: {len(ratings):,} ratings")

    # Iteratively filter until no more changes
    prev_len = len(ratings)
    iteration = 0

    while True:
        iteration += 1

        # Filter users
        user_counts = ratings[user_col].value_counts()
        valid_users = user_counts[user_counts >= min_user_interactions].index
        ratings = ratings[ratings[user_col].isin(valid_users)]

        # Filter items
        item_counts = ratings[item_col].value_counts()
        valid_items = item_counts[item_counts >= min_item_interactions].index
        ratings = ratings[ratings[item_col].isin(valid_items)]

        # Check if converged
        curr_len = len(ratings)
        if curr_len == prev_len:
            break

        prev_len = curr_len

    print(f"  After {iteration} iterations: {len(ratings):,} ratings")
    print(f"  Users: {ratings[user_col].nunique():,}")
    print(f"  Items: {ratings[item_col].nunique():,}")

    return ratings.reset_index(drop=True)


def preprocess_ratings(
    ratings: pd.DataFrame,
    min_user_interactions: int = 5,
    min_item_interactions: int = 5,
    use_temporal_split: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, IDEncoder]:
    """
    Complete preprocessing pipeline:
    1. Filter sparse users/items
    2. Train/test split
    3. Encode IDs consistently

    Args:
        ratings: Raw ratings DataFrame
        min_user_interactions: Minimum interactions per user
        min_item_interactions: Minimum interactions per item
        use_temporal_split: If True, use temporal split; otherwise random

    Returns:
        Tuple of (train DataFrame, test DataFrame, ID encoder)
    """
    print("\n" + "=" * 60)
    print("PREPROCESSING PIPELINE")
    print("=" * 60)

    # Step 1: Filter sparse users/items
    ratings = filter_sparse_users_items(
        ratings, min_user_interactions, min_item_interactions
    )

    if ratings.empty:
        raise ValueError(
            "No ratings remain after k-core filtering. "
            "Increase data subset size or lower min_interactions thresholds."
        )

    # Step 2: Train/test split
    if use_temporal_split:
        train, test = temporal_train_test_split(ratings)
    else:
        # Random split (not recommended for temporal data)
        from sklearn.model_selection import train_test_split as sklearn_split

        train, test = sklearn_split(ratings, test_size=0.2, random_state=42)

    # Step 3: Encode IDs (fit on train, transform both)
    print("\nEncoding IDs...")
    encoder = IDEncoder()
    train_user_idx, train_item_idx = encoder.fit_transform(
        train["userId"].values, train["movieId"].values
    )

    # Add encoded columns to train
    train = train.copy()
    train["user_idx"] = train_user_idx
    train["item_idx"] = train_item_idx

    # Transform test using same encoder
    test = test.copy()
    test_user_idx = encoder.transform_users(test["userId"].values)
    test_item_idx = encoder.transform_items(test["movieId"].values)

    # Filter out test samples with unknown users/items
    valid_mask = (test_user_idx >= 0) & (test_item_idx >= 0)
    n_filtered = (~valid_mask).sum()

    if n_filtered > 0:
        print(f"Warning: Filtered {n_filtered} test samples with unknown users/items")
        test = test[valid_mask].reset_index(drop=True)
        test_user_idx = test_user_idx[valid_mask]
        test_item_idx = test_item_idx[valid_mask]

    test["user_idx"] = test_user_idx
    test["item_idx"] = test_item_idx

    print(f"\n✓ Preprocessing complete")
    print(f"  Train: {len(train):,} ratings")
    print(f"  Test: {len(test):,} ratings")
    print(f"  Users: {encoder.num_users:,}")
    print(f"  Items: {encoder.num_items:,}")
    denominator = encoder.num_users * encoder.num_items
    sparsity = (100 * len(train) / denominator) if denominator > 0 else 0.0
    print(f"  Sparsity: {sparsity:.4f}%")
    print("=" * 60 + "\n")

    return train, test, encoder
