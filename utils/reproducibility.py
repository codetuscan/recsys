"""
Reproducibility utilities for setting random seeds and logging environment.
"""

import random
import sys
import os
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Make cuDNN deterministic (may reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set environment variables for full reproducibility
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # For deterministic CUDA operations

    print(f"✓ Random seed set to {seed} (reproducible mode enabled)")


def log_environment() -> dict:
    """
    Log and return environment details for reproducibility.

    Returns:
        Dictionary with environment information
    """
    info = {
        "python_version": sys.version,
        "python_executable": sys.executable,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }

    if torch.cuda.is_available():
        info["gpu_devices"] = [
            {
                "id": i,
                "name": torch.cuda.get_device_name(i),
                "memory_total": torch.cuda.get_device_properties(i).total_memory,
                "memory_allocated": torch.cuda.memory_allocated(i),
            }
            for i in range(torch.cuda.device_count())
        ]

    return info


def print_reproducibility_info(seed: int) -> None:
    """
    Print reproducibility and environment information.

    Args:
        seed: The random seed being used
    """
    env_info = log_environment()

    print("\n" + "=" * 60)
    print("REPRODUCIBILITY INFORMATION")
    print("=" * 60)
    print(f"Random seed: {seed}")
    print(f"Python: {env_info['python_version'].split()[0]}")
    print(f"PyTorch: {env_info['torch_version']}")
    print(f"NumPy: {env_info['numpy_version']}")
    print(f"CUDA available: {env_info['cuda_available']}")

    if env_info["cuda_available"]:
        print(f"CUDA version: {env_info['cuda_version']}")
        print(f"cuDNN version: {env_info['cudnn_version']}")
        print(f"GPU device(s): {env_info['device_count']}")
        for gpu in env_info["gpu_devices"]:
            memory_gb = gpu["memory_total"] / (1024**3)
            print(f"  - {gpu['name']}: {memory_gb:.1f} GB")

    print("Deterministic mode: ENABLED")
    print("=" * 60 + "\n")


def setup_reproducibility(seed: int = 42, verbose: bool = True) -> dict:
    """
    Complete reproducibility setup: set seeds and log environment.

    Args:
        seed: Random seed value
        verbose: If True, print reproducibility information

    Returns:
        Dictionary with environment information
    """
    set_seed(seed)
    env_info = log_environment()

    if verbose:
        print_reproducibility_info(seed)

    return env_info
