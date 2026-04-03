"""
Data loading and preprocessing modules.
"""

from .loaders import (
    load_movielens_ratings,
    load_movielens_movies,
    download_movielens_32m,
    load_data_with_fallback,
    validate_ratings_dataframe,
)
from .preprocessing import (
    IDEncoder,
    temporal_train_test_split,
    filter_sparse_users_items,
    preprocess_ratings,
)
from .dataset import (
    BPRDataset,
    EvaluationDataset,
    build_user_items_dict,
    build_user_history_dict,
    pad_sequence,
)

__all__ = [
    "load_movielens_ratings",
    "load_movielens_movies",
    "download_movielens_32m",
    "load_data_with_fallback",
    "validate_ratings_dataframe",
    "IDEncoder",
    "temporal_train_test_split",
    "filter_sparse_users_items",
    "preprocess_ratings",
    "BPRDataset",
    "EvaluationDataset",
    "build_user_items_dict",
    "build_user_history_dict",
    "pad_sequence",
]
