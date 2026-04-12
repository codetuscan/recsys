"""
Data loading utilities for MovieLens dataset.
"""

import pandas as pd
from pathlib import Path
from typing import Tuple, Optional
import urllib.request
import zipfile
import os


def _normalize_dataset_name(dataset_name: str) -> str:
    """Normalize dataset aliases into canonical names."""
    normalized = dataset_name.strip().lower()

    if normalized in {"movielens-1m", "ml-1m", "1m", "movielens1m"}:
        return "movielens-1m"
    if normalized in {"movielens-32m", "ml-32m", "32m", "movielens32m"}:
        return "movielens-32m"

    raise ValueError(
        f"Unsupported dataset_name='{dataset_name}'. "
        f"Supported values are movielens-1m or movielens-32m."
    )


def _resolve_dataset_path(data_path: Path, dataset_name: str) -> Path:
    """
    Resolve the actual dataset directory.

    This supports both direct dataset paths and parent directories that contain
    a dataset subfolder (e.g., ml-1m/ml-1m or ml-32m/ml-32m on some platforms).
    """
    dataset_name = _normalize_dataset_name(dataset_name)
    ratings_file = "ratings.dat" if dataset_name == "movielens-1m" else "ratings.csv"

    candidate_dirs = [
        data_path,
        data_path / "ml-1m",
        data_path / "ml-32m",
        data_path / "movielens-1m",
        data_path / "movielens-32m",
    ]

    for candidate in candidate_dirs:
        if (candidate / ratings_file).exists():
            return candidate

    return data_path


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


def load_movielens_1m_ratings(
    data_path: Path, subset: Optional[float] = None
) -> pd.DataFrame:
    """
    Load MovieLens-1M ratings from ratings.dat.

    Args:
        data_path: Path to the data directory (containing ratings.dat)
        subset: If specified, use only this fraction of data (for testing)

    Returns:
        DataFrame with columns: userId, movieId, rating, timestamp
    """
    ratings_file = data_path / "ratings.dat"

    if not ratings_file.exists():
        raise FileNotFoundError(
            f"ratings.dat not found at {ratings_file}. "
            f"Please ensure MovieLens-1M data is available."
        )

    print(f"Loading ratings from {ratings_file}...")

    ratings = pd.read_csv(
        ratings_file,
        sep="::",
        engine="python",
        names=["userId", "movieId", "rating", "timestamp"],
        dtype={"userId": "int32", "movieId": "int32", "rating": "float32", "timestamp": "int64"},
    )

    print(f"Loaded {len(ratings):,} ratings")

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


def load_movielens_1m_movies(data_path: Path) -> pd.DataFrame:
    """
    Load MovieLens-1M movie metadata from movies.dat.

    Args:
        data_path: Path to the data directory (containing movies.dat)

    Returns:
        DataFrame with columns: movieId, title, genres
    """
    movies_file = data_path / "movies.dat"

    if not movies_file.exists():
        print(f"Warning: movies.dat not found at {movies_file}")
        return pd.DataFrame()

    print(f"Loading movies from {movies_file}...")

    # MovieLens-1M movies.dat commonly uses latin-1 encoded titles (for example, accented chars).
    # Try utf-8 first, then fallback to latin-1/cp1252 for portability across Kaggle datasets.
    encodings_to_try = ["utf-8", "latin-1", "cp1252"]
    last_error = None
    movies = None

    for encoding in encodings_to_try:
        try:
            movies = pd.read_csv(
                movies_file,
                sep="::",
                engine="python",
                names=["movieId", "title", "genres"],
                dtype={"movieId": "int32", "title": "string", "genres": "string"},
                encoding=encoding,
            )
            if encoding != "utf-8":
                print(f"Read movies.dat with fallback encoding: {encoding}")
            break
        except UnicodeDecodeError as exc:
            last_error = exc

    if movies is None:
        raise UnicodeDecodeError(
            last_error.encoding if last_error is not None else "utf-8",
            last_error.object if last_error is not None else b"",
            last_error.start if last_error is not None else 0,
            last_error.end if last_error is not None else 1,
            "Failed to decode movies.dat with tried encodings: "
            + ", ".join(encodings_to_try),
        )

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
    data_path: Path,
    subset: Optional[float] = None,
    auto_download: bool = False,
    dataset_name: str = "movielens-32m",
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
    dataset_name = _normalize_dataset_name(dataset_name)
    data_path = _resolve_dataset_path(data_path, dataset_name)

    ratings_file = data_path / ("ratings.dat" if dataset_name == "movielens-1m" else "ratings.csv")

    # Check if data exists
    if not ratings_file.exists():
        if auto_download:
            if dataset_name == "movielens-1m":
                raise FileNotFoundError(
                    "Auto-download is only implemented for MovieLens-32M. "
                    "Please add the MovieLens-1M dataset manually in Kaggle input."
                )
            print(f"Data not found at {data_path}. Downloading...")
            parent_dir = data_path.parent
            download_movielens_32m(parent_dir)
        else:
            raise FileNotFoundError(
                f"Data not found at {data_path}. "
                f"Set auto_download=True to download automatically."
            )

    # Load ratings and movies
    if dataset_name == "movielens-1m":
        ratings = load_movielens_1m_ratings(data_path, subset=subset)
        movies = load_movielens_1m_movies(data_path)
    else:
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
