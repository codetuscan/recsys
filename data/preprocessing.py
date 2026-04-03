"""
Data preprocessing pipeline for MovieLens data.

Handles:
- ID encoding (user and movie IDs to dense indices)
- Train/test splitting (temporal)
- Data filtering
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict
from pathlib import Path
import pickle


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
    print(f"  Sparsity: {100 * len(train) / (encoder.num_users * encoder.num_items):.4f}%")
    print("=" * 60 + "\n")

    return train, test, encoder
