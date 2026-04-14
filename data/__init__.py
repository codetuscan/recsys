"""
Data loading and preprocessing modules.
"""

from .loaders import (
    load_movielens_ratings,
    load_movielens_1m_ratings,
    load_movielens_movies,
    load_movielens_1m_movies,
    download_movielens_32m,
    load_data_with_fallback,
    validate_ratings_dataframe,
)
from .preprocessing import (
    IDEncoder,
    temporal_train_test_split,
    temporal_train_val_test_split,
    add_log_time_gap_buckets,
    filter_sparse_users_items,
    preprocess_ratings,
    preprocess_sequential_ratings,
    save_sequential_preprocessing_artifacts,
    write_preprocessing_manifest,
)
from .dataset import (
    PairwiseTrainingDataset,
    PointwiseTrainingDataset,
    SequentialPairwiseDataset,
    EvaluationDataset,
    build_user_items_dict,
    build_user_history_dict,
    pad_sequence,
)

__all__ = [
    "load_movielens_ratings",
    "load_movielens_1m_ratings",
    "load_movielens_movies",
    "load_movielens_1m_movies",
    "download_movielens_32m",
    "load_data_with_fallback",
    "validate_ratings_dataframe",
    "IDEncoder",
    "temporal_train_test_split",
    "temporal_train_val_test_split",
    "add_log_time_gap_buckets",
    "filter_sparse_users_items",
    "preprocess_ratings",
    "preprocess_sequential_ratings",
    "save_sequential_preprocessing_artifacts",
    "write_preprocessing_manifest",
    "PairwiseTrainingDataset",
    "PointwiseTrainingDataset",
    "SequentialPairwiseDataset",
    "EvaluationDataset",
    "build_user_items_dict",
    "build_user_history_dict",
    "pad_sequence",
]
