"""
PyTorch Dataset classes for recommendation training and evaluation.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, Set
from collections import defaultdict


class PairwiseTrainingDataset(Dataset):
    """
    PyTorch Dataset for pairwise ranking training.

    Generates triplets (user, positive_item, negative_item) where:
    - positive_item is in the user's interaction history
    - negative_item is NOT in the user's interaction history

    Optionally includes user history sequences for sequential models like PURS.
    """

    def __init__(
        self,
        user_items: Dict[int, Set[int]],
        num_items: int,
        num_negatives: int = 1,
        num_samples_per_epoch: int = None,
        user_history: Dict[int, list] = None,
        history_length: int = 10,
        pad_value: int = 0,
    ):
        """
        Initialize pairwise training dataset.

        Args:
            user_items: Dictionary mapping user_id -> set of item_ids
            num_items: Total number of items
            num_negatives: Number of negative samples per positive
            num_samples_per_epoch: Total samples per epoch. If None, uses total interactions.
            user_history: Dictionary mapping user_id -> ordered list of item_ids (for sequential models)
            history_length: Fixed history sequence length (for padding/truncation)
        """
        self.user_items = user_items
        self.num_items = num_items
        self.num_negatives = num_negatives
        self.user_history = user_history or {}
        self.history_length = history_length
        self.pad_value = pad_value

        # Create list of all (user, positive_item) pairs
        self.user_positive_pairs = []
        for user, items in user_items.items():
            for item in items:
                self.user_positive_pairs.append((user, item))

        # Number of samples per epoch
        if num_samples_per_epoch is None:
            self.num_samples = len(self.user_positive_pairs)
        else:
            self.num_samples = min(num_samples_per_epoch, len(self.user_positive_pairs))

        print(
            f"PairwiseTrainingDataset initialized: {len(user_items)} users, "
            f"{num_items} items, {len(self.user_positive_pairs)} interactions, "
            f"{self.num_samples} samples per epoch"
            + (f", history_length={history_length}" if user_history else "")
        )

    def __len__(self) -> int:
        """Return number of samples in the dataset."""
        return self.num_samples

    def __getitem__(self, idx: int):
        """
        Get a training triplet (user, positive_item, negative_item) and optional history.

        Args:
            idx: Sample index

        Returns:
            If history provided: (user_id_tensor, history_seq_tensor, pos_item_tensor, neg_item_tensor)
            Otherwise: (user_id_tensor, pos_item_tensor, neg_item_tensor)
        """
        # Sample a random user-positive pair
        pair_idx = np.random.randint(len(self.user_positive_pairs))
        user, pos_item = self.user_positive_pairs[pair_idx]

        # Sample negative item(s)
        neg_items = self._sample_negative_items(user, self.num_negatives)

        # Build return tuple
        result = [
            torch.LongTensor([user]),
            torch.LongTensor([pos_item]),
            torch.LongTensor(neg_items) if self.num_negatives > 1 else torch.LongTensor([neg_items[0]]),
        ]

        # Add history if available
        if self.user_history:
            history = self.user_history.get(user, [])
            history_padded = pad_sequence(history, self.history_length, pad_value=self.pad_value)
            result.insert(1, torch.LongTensor(history_padded))  # Insert after user_id

        return tuple(result)

    def _sample_negative_items(self, user: int, num_negatives: int) -> list[int]:
        """
        Sample negative items for a user (items NOT in user's history).

        Args:
            user: User ID
            num_negatives: Number of negative samples

        Returns:
            List of negative item IDs
        """
        user_positive_items = self.user_items[user]
        negative_items = []

        # Rejection sampling: keep sampling until we get enough negatives
        while len(negative_items) < num_negatives:
            neg_item = np.random.randint(self.num_items)
            if neg_item not in user_positive_items:
                negative_items.append(neg_item)

        return negative_items


class PointwiseTrainingDataset(Dataset):
    """
    Pointwise binary training dataset for explicit ratings.

    Each sample is built from an observed user-item interaction and a binary label:
    - label = 1 if rating >= positive_threshold
    - label = 0 otherwise

    Optionally includes user history sequences for sequential models like PURS.
    """

    def __init__(
        self,
        ratings_df,
        user_col: str = "user_idx",
        item_col: str = "item_idx",
        rating_col: str = "rating",
        positive_threshold: float = 3.5,
        user_history: Dict[int, list] = None,
        history_length: int = 10,
        pad_value: int = 0,
    ):
        if rating_col not in ratings_df.columns:
            raise ValueError(
                f"PointwiseTrainingDataset requires '{rating_col}' column for label binarization."
            )

        self.user_history = user_history or {}
        self.history_length = history_length
        self.pad_value = pad_value

        self.users = ratings_df[user_col].astype(np.int64).values
        self.items = ratings_df[item_col].astype(np.int64).values
        raw_ratings = ratings_df[rating_col].astype(np.float32).values
        self.labels = (raw_ratings >= float(positive_threshold)).astype(np.float32)

        positive_count = int(np.sum(self.labels))
        negative_count = int(len(self.labels) - positive_count)

        print(
            f"PointwiseTrainingDataset initialized: {len(self.labels)} samples, "
            f"positives={positive_count}, negatives={negative_count}, "
            f"threshold={positive_threshold}"
            + (f", history_length={history_length}" if user_history else "")
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        user = int(self.users[idx])
        item = int(self.items[idx])
        label = float(self.labels[idx])

        result = [
            torch.LongTensor([user]),
            torch.LongTensor([item]),
            torch.FloatTensor([label]),
        ]

        if self.user_history:
            history = self.user_history.get(user, [])
            history_padded = pad_sequence(history, self.history_length, pad_value=self.pad_value)
            result.insert(1, torch.LongTensor(history_padded))

        return tuple(result)


class EvaluationDataset(Dataset):
    """
    Dataset for evaluation with negative sampling.

    For each test sample, creates a candidate set of:
    - 1 positive item (ground truth)
    - K negative items

    Optionally includes user history sequences for sequential models like PURS.
    """

    def __init__(
        self,
        test_interactions: list[tuple[int, int]],
        user_train_items: Dict[int, Set[int]],
        num_items: int,
        num_negatives: int = 99,
        user_history: Dict[int, list] = None,
        history_length: int = 10,
        pad_value: int = 0,
    ):
        """
        Initialize evaluation dataset.

        Args:
            test_interactions: List of (user_id, item_id) test pairs
            user_train_items: Dictionary of training items per user (for negative sampling)
            num_items: Total number of items
            num_negatives: Number of negative samples per positive
            user_history: Dictionary mapping user_id -> ordered list of item_ids (for sequential models)
            history_length: Fixed history sequence length (for padding/truncation)
        """
        self.test_interactions = test_interactions
        self.user_train_items = user_train_items
        self.num_items = num_items
        self.num_negatives = num_negatives
        self.user_history = user_history or {}
        self.history_length = history_length
        self.pad_value = pad_value

        print(
            f"EvaluationDataset initialized: {len(test_interactions)} test samples, "
            f"{num_negatives} negatives per sample"
            + (f", history_length={history_length}" if user_history else "")
        )

    def __len__(self) -> int:
        """Return number of test samples."""
        return len(self.test_interactions)

    def __getitem__(self, idx: int):
        """
        Get evaluation sample with candidate items and optional history.

        Args:
            idx: Sample index

        Returns:
            If history provided: (user_id_tensor, history_seq_tensor, candidate_items_tensor, gt_position_tensor)
            Otherwise: (user_id_tensor, candidate_items_tensor, gt_position_tensor)
        """
        user, pos_item = self.test_interactions[idx]

        # Sample negative items
        train_items = self.user_train_items.get(user, set())
        neg_items = []

        while len(neg_items) < self.num_negatives:
            neg_item = np.random.randint(self.num_items)
            # Exclude both training and test items
            if neg_item not in train_items and neg_item != pos_item:
                neg_items.append(neg_item)

        # Create candidate set: [positive_item, negative_items...]
        # Randomly place positive item in the candidate set
        gt_position = np.random.randint(self.num_negatives + 1)
        candidates = neg_items[:gt_position] + [pos_item] + neg_items[gt_position:]

        # Build return tuple
        result = [
            torch.LongTensor([user]),
            torch.LongTensor(candidates),
            torch.LongTensor([gt_position]),
        ]

        # Add history if available
        if self.user_history:
            history = self.user_history.get(user, [])
            history_padded = pad_sequence(history, self.history_length, pad_value=self.pad_value)
            result.insert(1, torch.LongTensor(history_padded))  # Insert after user_id

        return tuple(result)


class SequentialPairwiseDataset(Dataset):
    """
    Sequential pairwise training dataset for next-item recommendation.

    Creates samples from chronological user sequences:
    - history: interactions before time t
    - positive: interaction at time t
    - negative: sampled item not seen by the user
    """

    def __init__(
        self,
        ratings_df,
        num_items: int,
        num_negatives: int = 1,
        history_length: int = 50,
        user_col: str = "user_idx",
        item_col: str = "item_idx",
        time_col: str = "timestamp",
        pad_value: int = 0,
    ):
        self.num_items = num_items
        self.num_negatives = num_negatives
        self.history_length = history_length
        self.pad_value = pad_value

        self.user_positive_items: Dict[int, Set[int]] = {}
        self.samples: list[tuple[int, list[int], int]] = []

        ratings_df_sorted = ratings_df.sort_values(by=[user_col, time_col])

        for user, group in ratings_df_sorted.groupby(user_col, sort=False):
            items = [int(x) for x in group[item_col].tolist()]

            if len(items) < 2:
                continue

            self.user_positive_items[int(user)] = set(items)

            for idx in range(1, len(items)):
                start = max(0, idx - history_length)
                history = items[start:idx]
                pos_item = items[idx]
                self.samples.append((int(user), history, int(pos_item)))

        print(
            f"SequentialPairwiseDataset initialized: {len(self.user_positive_items)} users, "
            f"{num_items} items, {len(self.samples)} sequence samples, "
            f"history_length={history_length}, pad_value={pad_value}"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        user, history, pos_item = self.samples[idx]
        neg_items = self._sample_negative_items(user, self.num_negatives)

        history_padded = pad_sequence(history, self.history_length, pad_value=self.pad_value)

        result = [
            torch.LongTensor([user]),
            torch.LongTensor(history_padded),
            torch.LongTensor([pos_item]),
            torch.LongTensor(neg_items) if self.num_negatives > 1 else torch.LongTensor([neg_items[0]]),
        ]

        return tuple(result)

    def _sample_negative_items(self, user: int, num_negatives: int) -> list[int]:
        user_positive_items = self.user_positive_items[user]
        negative_items = []

        while len(negative_items) < num_negatives:
            neg_item = np.random.randint(self.num_items)
            if neg_item not in user_positive_items:
                negative_items.append(neg_item)

        return negative_items


def build_user_items_dict(ratings_df, user_col="user_idx", item_col="item_idx") -> Dict[int, Set[int]]:
    """
    Build dictionary mapping user_id -> set of item_ids efficiently.

    Args:
        ratings_df: DataFrame with ratings
        user_col: Column name for user IDs
        item_col: Column name for item IDs

    Returns:
        Dictionary mapping user_id -> set of item_ids
    """
    user_items = defaultdict(set)

    # Use groupby for efficient aggregation (much faster than iterrows)
    for user, group in ratings_df.groupby(user_col):
        user_items[user] = set(group[item_col].values)

    return dict(user_items)


def build_user_history_dict(
    ratings_df, user_col="user_idx", item_col="item_idx", time_col="timestamp", max_history_length: int = None
) -> Dict[int, list]:
    """
    Build dictionary mapping user_id -> ordered list of item_ids (chronological).

    Used for sequential models like PURS that need user interaction history.

    Args:
        ratings_df: DataFrame with ratings (must be sorted by timestamp)
        user_col: Column name for user IDs
        item_col: Column name for item IDs
        time_col: Column name for timestamps
        max_history_length: Maximum history length per user. If None, uses full history.

    Returns:
        Dictionary mapping user_id -> ordered list of item_ids
    """
    user_history = defaultdict(list)

    # Ensure data is sorted by timestamp
    ratings_df_sorted = ratings_df.sort_values(by=time_col)

    # Build history in chronological order
    for user, group in ratings_df_sorted.groupby(user_col):
        items = group[item_col].values.tolist()

        # Truncate to max length if specified
        if max_history_length is not None and len(items) > max_history_length:
            items = items[-max_history_length:]  # Keep most recent items

        user_history[user] = items

    return dict(user_history)


def pad_sequence(seq: list, max_length: int, pad_value: int = 0) -> np.ndarray:
    """
    Pad or truncate a sequence to fixed length.

    Args:
        seq: Input sequence (list of integers)
        max_length: Target length
        pad_value: Value to pad with (usually 0 or -1)

    Returns:
        Padded numpy array of shape (max_length,)
    """
    if len(seq) >= max_length:
        return np.array(seq[-max_length:], dtype=np.int64)
    else:
        padded = np.full(max_length, pad_value, dtype=np.int64)
        padded[-len(seq):] = seq  # Right-align (most recent at the end)
        return padded
