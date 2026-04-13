"""
Minimal PURS baseline for Kaggle.
Run directly in notebook - delayed torch import to avoid CUDA conflicts.
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

print("Setting up environment...")

# Import lightweight modules first
import numpy as np
import pandas as pd
from pathlib import Path

print("✓ Core modules imported")
print("="*60)
print("PURS Baseline Setup")
print("="*60)

# Minimal config
class Config:
    # Check multiple paths for MovieLens data
    data_path = None
    for p in [
        Path('/kaggle/input/datasets/justsahil/movielens-32m/ml-32m'),  # Added dataset
        Path('/kaggle/input/movielens-datasets'),
        Path('/kaggle/input/movielens32m'),
        Path('/kaggle/working/ml-32m'),
    ]:
        if (p / 'ratings.csv').exists():
            data_path = p
            break

    if data_path is None:
        data_path = Path('/kaggle/input/datasets/justsahil/movielens-32m/ml-32m')

    model_name = 'purs'
    epochs = 5
    batch_size = 64
    history_length = 50
    gru_hidden_dim = 64
    num_clusters = 5
    unexpectedness_weight = 0.1
    learning_rate = 0.001
    device = 'cpu'

config = Config()

# Delayed torch import
try:
    import torch
    import torch.nn as nn
    config.device = 'cpu'
    print(f"Device: {config.device}")
except Exception as e:
    print(f"Warning: torch import had issues: {e}")

print(f"Data path: {config.data_path}")

# Quick test - can we import recsys modules now?
print("\nAttempting to import recsys...")
try:
    sys.path.insert(0, '/kaggle/working/recsys')
    from recsys.models import PURS
    from recsys.data import load_movielens_ratings
    print("✓ Successfully imported recsys modules!")

    print("\nLoading MovieLens data (subset 1%)...")
    ratings_df = load_movielens_ratings(config.data_path, subset=0.01)
    print(f"✓ Loaded {len(ratings_df)} ratings")

    print("\nInitializing PURS model...")
    model = PURS(
        num_users=ratings_df['userId'].max() + 1,
        num_items=ratings_df['movieId'].max() + 1,
        embedding_dim=32,
        gru_hidden_dim=config.gru_hidden_dim,
        num_clusters=config.num_clusters,
        unexpectedness_weight=config.unexpectedness_weight,
        history_length=config.history_length,
    ).to(config.device)
    print(f"✓ Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")

    print("\n" + "="*60)
    print("SUCCESS! Ready to train PURS")
    print("="*60)

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

