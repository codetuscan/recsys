"""
Recommender System for Breaking Filter Bubbles through Unexpectedness.

This package implements a GPU-accelerated recommender system that optimizes
for both relevance and unexpectedness to break filter bubbles.
"""

__version__ = "0.1.0"

from .config import Config, load_config
from .experiments import ExperimentRunner

__all__ = [
    "Config",
    "load_config",
    "ExperimentRunner",
]
