"""
Data loading utilities for MovieLens dataset.
"""

import pandas as pd
from pathlib import Path
from typing import Tuple, Optional
import urllib.request
import zipfile
import os


def load_movielens_ratings(
    data_path: Path, subset: Optional[float] = None
) -> pd.DataFrame:
    """
    Load MovieLens ratings from CSV file.

    Args:
        data_path: Path to the data directory (containing ratings.csv)
        subset: If specified, use only this fraction of data (for testing)

    Returns:
        DataFrame with ratings
    """
    ratings_file = data_path / "ratings.csv"

    if not ratings_file.exists():
        raise FileNotFoundError(
            f"ratings.csv not found at {ratings_file}. "
            f"Please ensure MovieLens data is available."
        )

    print(f"Loading ratings from {ratings_file}...")

    # Load ratings
    ratings = pd.read_csv(ratings_file)

    print(f"Loaded {len(ratings):,} ratings")

    # Apply subset if specified
    if subset is not None and 0 < subset < 1:
        n_samples = int(len(ratings) * subset)
        ratings = ratings.sample(n=n_samples, random_state=42).reset_index(drop=True)
        print(f"Using subset: {len(ratings):,} ratings ({subset*100:.1f}%)")

    return ratings


def load_movielens_movies(data_path: Path) -> pd.DataFrame:
    """
    Load MovieLens movies metadata.

    Args:
        data_path: Path to the data directory (containing movies.csv)

    Returns:
        DataFrame with movie information
    """
    movies_file = data_path / "movies.csv"

    if not movies_file.exists():
        print(f"Warning: movies.csv not found at {movies_file}")
        return pd.DataFrame()

    print(f"Loading movies from {movies_file}...")
    movies = pd.read_csv(movies_file)
    print(f"Loaded {len(movies):,} movies")

    return movies


def download_movielens_32m(output_dir: Path) -> None:
    """
    Download MovieLens 32M dataset from GroupLens.

    Args:
        output_dir: Directory to save the dataset
    """
    url = "https://files.grouplens.org/datasets/movielens/ml-32m.zip"
    zip_path = output_dir / "ml-32m.zip"
    extract_path = output_dir

    print(f"Downloading MovieLens 32M from {url}...")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Download
    urllib.request.urlretrieve(url, zip_path)
    print(f"Downloaded to {zip_path}")

    # Extract
    print("Extracting...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_path)

    # Clean up zip file
    os.remove(zip_path)
    print(f"Extracted to {extract_path / 'ml-32m'}")


def load_data_with_fallback(
    data_path: Path, subset: Optional[float] = None, auto_download: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load MovieLens data with fallback to download if not found.

    Args:
        data_path: Path to the data directory
        subset: Optional fraction of data to use
        auto_download: If True, download data if not found

    Returns:
        Tuple of (ratings DataFrame, movies DataFrame)
    """
    ratings_file = data_path / "ratings.csv"

    # Check if data exists
    if not ratings_file.exists():
        if auto_download:
            print(f"Data not found at {data_path}. Downloading...")
            parent_dir = data_path.parent
            download_movielens_32m(parent_dir)
        else:
            raise FileNotFoundError(
                f"Data not found at {data_path}. "
                f"Set auto_download=True to download automatically."
            )

    # Load ratings and movies
    ratings = load_movielens_ratings(data_path, subset=subset)
    movies = load_movielens_movies(data_path)

    return ratings, movies


def validate_ratings_dataframe(ratings: pd.DataFrame) -> None:
    """
    Validate that ratings DataFrame has required columns.

    Args:
        ratings: Ratings DataFrame to validate

    Raises:
        ValueError: If required columns are missing
    """
    required_cols = ["userId", "movieId", "rating", "timestamp"]
    missing_cols = [col for col in required_cols if col not in ratings.columns]

    if missing_cols:
        raise ValueError(
            f"Missing required columns: {missing_cols}. "
            f"Found columns: {list(ratings.columns)}"
        )

    print(f"✓ Ratings DataFrame validation passed")
    print(f"  Shape: {ratings.shape}")
    print(f"  Users: {ratings['userId'].nunique():,}")
    print(f"  Movies: {ratings['movieId'].nunique():,}")
    print(f"  Rating range: [{ratings['rating'].min()}, {ratings['rating'].max()}]")
    print(f"  Date range: {pd.to_datetime(ratings['timestamp'], unit='s').min()} "
          f"to {pd.to_datetime(ratings['timestamp'], unit='s').max()}")
