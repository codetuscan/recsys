"""
Logging utilities for training and evaluation.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
import json


def setup_logger(
    name: str,
    log_dir: Optional[Path] = None,
    level: int = logging.INFO,
    console_output: bool = True,
) -> logging.Logger:
    """
    Setup a logger with file and console handlers.

    Args:
        name: Logger name
        log_dir: Directory for log files. If None, only console logging.
        level: Logging level
        console_output: If True, also log to console

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers = []  # Remove existing handlers

    # Create formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"{name}_{timestamp}.log"

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        logger.info(f"Logging to {log_file}")

    return logger


class MetricsLogger:
    """Logger for tracking training and evaluation metrics."""

    def __init__(self, log_dir: Path, experiment_name: str):
        """
        Initialize metrics logger.

        Args:
            log_dir: Directory for saving metric logs
            experiment_name: Name of the experiment
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.experiment_name = experiment_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.train_metrics = []
        self.eval_metrics = []

        # Create experiment directory
        self.experiment_dir = self.log_dir / f"{experiment_name}_{self.timestamp}"
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

    def log_train_metrics(self, epoch: int, metrics: dict):
        """
        Log training metrics for an epoch.

        Args:
            epoch: Epoch number
            metrics: Dictionary of metric name -> value
        """
        entry = {"epoch": epoch, "timestamp": datetime.now().isoformat(), **metrics}
        self.train_metrics.append(entry)

        # Save incrementally
        self._save_json(self.train_metrics, "train_metrics.json")

    def log_eval_metrics(self, epoch: int, metrics: dict):
        """
        Log evaluation metrics for an epoch.

        Args:
            epoch: Epoch number
            metrics: Dictionary of metric name -> value
        """
        entry = {"epoch": epoch, "timestamp": datetime.now().isoformat(), **metrics}
        self.eval_metrics.append(entry)

        # Save incrementally
        self._save_json(self.eval_metrics, "eval_metrics.json")

    def log_final_results(self, results: dict):
        """
        Log final experiment results.

        Args:
            results: Dictionary with final results
        """
        results_with_metadata = {
            "experiment_name": self.experiment_name,
            "timestamp": self.timestamp,
            "completed_at": datetime.now().isoformat(),
            **results,
        }

        self._save_json(results_with_metadata, "final_results.json")

    def _save_json(self, data: dict | list, filename: str):
        """Save data to JSON file."""
        filepath = self.experiment_dir / filename
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def get_experiment_dir(self) -> Path:
        """Get the experiment directory path."""
        return self.experiment_dir


class ProgressTracker:
    """Track and display training progress."""

    def __init__(self, total_epochs: int, total_batches: int):
        """
        Initialize progress tracker.

        Args:
            total_epochs: Total number of epochs
            total_batches: Total number of batches per epoch
        """
        self.total_epochs = total_epochs
        self.total_batches = total_batches
        self.current_epoch = 0
        self.current_batch = 0
        self.epoch_start_time = None
        self.training_start_time = None

    def start_training(self):
        """Mark training start."""
        from datetime import datetime

        self.training_start_time = datetime.now()

    def start_epoch(self, epoch: int):
        """Mark epoch start."""
        from datetime import datetime

        self.current_epoch = epoch
        self.current_batch = 0
        self.epoch_start_time = datetime.now()

    def update_batch(self, batch: int, loss: float):
        """
        Update batch progress.

        Args:
            batch: Current batch number
            loss: Current loss value
        """
        from datetime import datetime

        self.current_batch = batch

        if batch % 100 == 0:  # Print every 100 batches
            progress = (batch / self.total_batches) * 100
            elapsed = (datetime.now() - self.epoch_start_time).total_seconds()
            batches_per_sec = batch / elapsed if elapsed > 0 else 0

            print(
                f"  Epoch {self.current_epoch}/{self.total_epochs} | "
                f"Batch {batch}/{self.total_batches} ({progress:.1f}%) | "
                f"Loss: {loss:.4f} | "
                f"Speed: {batches_per_sec:.1f} batches/s",
                flush=True,
            )

    def end_epoch(self, metrics: dict):
        """
        Mark epoch end and print summary.

        Args:
            metrics: Dictionary of metrics for the epoch
        """
        from datetime import datetime

        elapsed = (datetime.now() - self.epoch_start_time).total_seconds()

        metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])

        print(
            f"Epoch {self.current_epoch} completed in {elapsed:.1f}s | {metrics_str}",
            flush=True,
        )

    def end_training(self):
        """Mark training end and print summary."""
        from datetime import datetime

        if self.training_start_time:
            total_time = (datetime.now() - self.training_start_time).total_seconds()
            hours = int(total_time // 3600)
            minutes = int((total_time % 3600) // 60)
            seconds = int(total_time % 60)

            print("\n" + "=" * 60)
            print(f"Training completed in {hours}h {minutes}m {seconds}s")
            print("=" * 60 + "\n")
