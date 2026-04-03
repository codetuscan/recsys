"""
Environment detection and path management for local vs Kaggle execution.
"""

import os
from pathlib import Path
from typing import Dict, Literal

EnvType = Literal["local", "kaggle"]


def detect_environment() -> EnvType:
    """
    Detect the current execution environment.

    Returns:
        "kaggle" if running in Kaggle environment, "local" otherwise
    """
    if "KAGGLE_KERNEL_RUN_TYPE" in os.environ:
        return "kaggle"
    return "local"


def get_data_paths(env: EnvType = None) -> Dict[str, Path]:
    """
    Get environment-specific data and output paths.

    Args:
        env: Environment type. If None, auto-detects.

    Returns:
        Dictionary with paths for:
        - raw: Raw data directory
        - processed: Processed data directory
        - outputs: Outputs directory
        - models: Model checkpoints directory
        - logs: Logging directory
        - results: Results directory
    """
    if env is None:
        env = detect_environment()

    if env == "kaggle":
        base_input = Path("/kaggle/input")
        base_working = Path("/kaggle/working")

        return {
            "raw": base_input / "movielens-32m",
            "processed": base_working / "data" / "processed",
            "outputs": base_working / "outputs",
            "models": base_working / "outputs" / "models",
            "logs": base_working / "outputs" / "logs",
            "results": base_working / "outputs" / "results",
        }
    else:
        project_root = Path(__file__).parent.parent

        return {
            "raw": project_root / "data" / "raw" / "ml-32m",
            "processed": project_root / "data" / "processed",
            "outputs": project_root / "outputs",
            "models": project_root / "outputs" / "models",
            "logs": project_root / "outputs" / "logs",
            "results": project_root / "outputs" / "results",
        }


def ensure_directories(env: EnvType = None) -> None:
    """
    Create necessary directories if they don't exist.

    Args:
        env: Environment type. If None, auto-detects.
    """
    paths = get_data_paths(env)

    # Only create output directories (don't create input directories)
    for key in ["processed", "outputs", "models", "logs", "results"]:
        paths[key].mkdir(parents=True, exist_ok=True)


def get_device_str(prefer_gpu: bool = True) -> str:
    """
    Get the appropriate device string for PyTorch.

    Args:
        prefer_gpu: If True, use GPU if available

    Returns:
        "cuda" if GPU available and preferred, "cpu" otherwise
    """
    import torch

    if prefer_gpu and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def print_environment_info() -> None:
    """
    Print information about the current environment.
    """
    import sys
    import torch

    env = detect_environment()
    paths = get_data_paths(env)

    print("=" * 60)
    print("ENVIRONMENT INFORMATION")
    print("=" * 60)
    print(f"Environment: {env}")
    print(f"Python version: {sys.version.split()[0]}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
        print(f"GPU count: {torch.cuda.device_count()}")

    print("\nPaths:")
    for key, path in paths.items():
        exists = "✓" if path.exists() else "✗"
        print(f"  {key:12s}: {path} [{exists}]")

    print("=" * 60)
