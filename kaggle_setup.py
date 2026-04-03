"""
Minimal PURS baseline for Kaggle.
Run directly in notebook - no package imports needed.
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Download and setup data
print("Setting up environment...")
os.system("cd /kaggle/working && wget -q https://files.grouplens.org/datasets/movielens/ml-32m.zip 2>/dev/null || echo 'Using existing data'")

# Import after warnings suppressed
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path

print("✓ Environment ready")
print("="*60)
print("PURS Baseline Setup")
print("="*60)

# Minimal config
class Config:
    data_path = Path('/kaggle/input/movielens-datasets') if Path('/kaggle/input/movielens-datasets').exists() else Path('/kaggle/working/ml-32m')
    model_name = 'purs'
    epochs = 5
    batch_size = 64
    history_length = 10
    gru_hidden_dim = 64
    num_clusters = 5
    unexpectedness_weight = 0.1
    learning_rate = 0.001
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

config = Config()
print(f"Device: {config.device}")
print(f"Data path: {config.data_path}")

# Quick test - can we import recsys modules now?
print("\nAttempting to import recsys...")
try:
    sys.path.insert(0, '/kaggle/working/recsys')
    from recsys.models import PURS
    from recsys.data import load_movielens_ratings
    print("✓ Successfully imported recsys modules!")

    print("\nLoading MovieLens data...")
    ratings_df = load_movielens_ratings(config.data_path, subset=0.01)
    print(f"✓ Loaded {len(ratings_df)} ratings")

    print("\nInitializing PURS model...")
    model = PURS(
        num_items=ratings_df['movieId'].max() + 1,
        embedding_dim=32,
        gru_hidden_dim=config.gru_hidden_dim,
        num_clusters=config.num_clusters,
    ).to(config.device)
    print(f"✓ Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")

    print("\n" + "="*60)
    print("SUCCESS! Ready to train PURS")
    print("="*60)

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
