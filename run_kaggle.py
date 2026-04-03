#!/usr/bin/env python
"""Standalone PURS runner for Kaggle - avoids package import issues."""

import sys
import os
from pathlib import Path

# Set up paths
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

def main():
    # Lazy imports to avoid numpy conflicts at startup
    import warnings
    warnings.filterwarnings('ignore')

    try:
        from recsys.experiments.experiment_runner import ExperimentRunner
        from recsys.config import load_config
    except ImportError as e:
        print(f"Import error: {e}")
        print("Trying alternative import...")
        # Fallback
        sys.exit(1)

    print("Loading config...")
    config = load_config('kaggle')
    config.model.model_name = 'purs'
    config.train.epochs = 10
    config.train.batch_size = 64

    print(f"Config: {config}")
    print("\nStarting experiment runner...")

    runner = ExperimentRunner(config)
    results = runner.run()

    print("\n" + "="*60)
    print("PURS BASELINE RESULTS")
    print("="*60)
    if isinstance(results, dict):
        for key, val in results.items():
            print(f"{key}: {val}")
    else:
        print(results)

if __name__ == '__main__':
    main()
