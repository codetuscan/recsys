"""
Experiment runner that orchestrates training and evaluation.
"""

import sys
from pathlib import Path

# Add parent directory to path to enable recsys imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
from torch.utils.data import DataLoader
import json
from datetime import datetime

from recsys.config import Config, load_config
from recsys.utils import (
    detect_environment,
    ensure_directories,
    setup_reproducibility,
    MetricsLogger,
    print_environment_info,
)
from recsys.data import (
    load_data_with_fallback,
    preprocess_ratings,
    BPRDataset,
    EvaluationDataset,
    build_user_items_dict,
    build_user_history_dict,
)
from recsys.models import BPR_MF_PyTorch, train_bpr, evaluate_bpr, PURS, train_purs, evaluate_purs


class ExperimentRunner:
    """
    Main experiment orchestrator for running BPR training and evaluation.
    """

    def __init__(self, config: Config = None, config_name: str = None):
        """
        Initialize experiment runner.

        Args:
            config: Configuration object. If None, loads from config_name.
            config_name: Name of config to load ("local", "kaggle")
        """
        if config is None:
            config = load_config(config_name)

        self.config = config
        self.env = detect_environment()
        self.device = config.experiment.device

        # Setup directories
        ensure_directories(self.env)

        # Setup reproducibility
        setup_reproducibility(config.experiment.seed, verbose=config.experiment.verbose)

        # Setup logging
        self.metrics_logger = MetricsLogger(
            log_dir=config.paths.logs,
            experiment_name=config.experiment.experiment_name,
        )

        # Placeholders
        self.train_data = None
        self.test_data = None
        self.encoder = None
        self.model = None
        self.optimizer = None

    def load_data(self):
        """Load and preprocess data."""
        print("\n" + "=" * 60)
        print("LOADING DATA")
        print("=" * 60)

        # Load raw ratings
        ratings, movies = load_data_with_fallback(
            data_path=self.config.paths.raw_data,
            subset=self.config.data.data_subset,
            auto_download=(self.env == "kaggle"),  # Auto-download on Kaggle if needed
        )

        # Preprocess
        self.train_data, self.test_data, self.encoder = preprocess_ratings(
            ratings,
            min_user_interactions=self.config.data.min_interactions_per_user,
            min_item_interactions=self.config.data.min_interactions_per_item,
            use_temporal_split=self.config.data.use_temporal_split,
        )

        print(f"✓ Data loaded successfully")
        print(f"  Environment: {self.env}")
        print(f"  Device: {self.device}")

    def create_dataloaders(self):
        """Create PyTorch DataLoaders for training and evaluation."""
        print("\n" + "=" * 60)
        print("CREATING DATALOADERS")
        print("=" * 60)

        # Build user-items dictionary
        user_items_train = build_user_items_dict(self.train_data)

        # Build user history if using PURS
        user_history_train = None
        if self.config.model.model_name == "purs":
            print("Building user history sequences for PURS...")
            user_history_train = build_user_history_dict(
                self.train_data,
                user_col="user_idx",
                item_col="item_idx",
                time_col="timestamp",
                max_history_length=self.config.model.history_length,
            )
            print(f"  Sample history lengths: {[len(h) for h in list(user_history_train.values())[:5]]}")

        # Create training dataset
        train_dataset = BPRDataset(
            user_items=user_items_train,
            num_items=self.encoder.num_items,
            num_negatives=self.config.data.negative_samples_train,
            user_history=user_history_train,
            history_length=self.config.model.history_length if user_history_train else 10,
        )

        # Create training DataLoader
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.model.batch_size,
            shuffle=True,
            num_workers=self.config.model.num_workers,
            pin_memory=self.config.model.pin_memory,
        )

        # Create evaluation dataset
        test_interactions = list(
            zip(self.test_data["user_idx"].values, self.test_data["item_idx"].values)
        )

        eval_dataset = EvaluationDataset(
            test_interactions=test_interactions,
            user_train_items=user_items_train,
            num_items=self.encoder.num_items,
            num_negatives=self.config.data.negative_samples_eval,
            user_history=user_history_train,
            history_length=self.config.model.history_length if user_history_train else 10,
        )

        # Create evaluation DataLoader
        self.eval_loader = DataLoader(
            eval_dataset,
            batch_size=self.config.model.batch_size,
            shuffle=False,
            num_workers=self.config.model.num_workers,
            pin_memory=self.config.model.pin_memory,
        )

        print(f"✓ DataLoaders created")
        print(f"  Train batches: {len(self.train_loader)}")
        print(f"  Eval batches: {len(self.eval_loader)}")
        print(f"  Batch size: {self.config.model.batch_size}")
        print("=" * 60)

    def create_model(self):
        """Create and initialize model."""
        print("\n" + "=" * 60)
        print("CREATING MODEL")
        print("=" * 60)

        if self.config.model.model_name == "purs":
            # Create PURS model
            self.model = PURS(
                num_users=self.encoder.num_users,
                num_items=self.encoder.num_items,
                embedding_dim=self.config.model.embedding_dim,
                gru_hidden_dim=self.config.model.gru_hidden_dim,
                num_clusters=self.config.model.num_clusters,
                unexpectedness_weight=self.config.model.unexpectedness_weight,
                history_length=self.config.model.history_length,
            )
            model_type = "PURS"
        else:
            # Create BPR-MF model (default)
            self.model = BPR_MF_PyTorch(
                num_users=self.encoder.num_users,
                num_items=self.encoder.num_items,
                embedding_dim=self.config.model.embedding_dim,
                reg_lambda=self.config.model.regularization,
            )
            model_type = "BPR-MF"

        # Move model to device
        self.model = self.model.to(self.device)

        # Create optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config.model.learning_rate
        )

        # Count parameters
        num_params = sum(p.numel() for p in self.model.parameters())

        print(f"✓ Model created: {model_type}")
        print(f"  Users: {self.encoder.num_users:,}")
        print(f"  Items: {self.encoder.num_items:,}")
        print(f"  Embedding dim: {self.config.model.embedding_dim}")
        print(f"  Parameters: {num_params:,}")
        print(f"  Device: {self.device}")
        print("=" * 60)

    def train(self):
        """Train the model."""
        print("\n" + "=" * 60)
        print("TRAINING")
        print("=" * 60)

        # Select training function based on model type
        train_fn = train_purs if self.config.model.model_name == "purs" else train_bpr
        eval_fn = evaluate_purs if self.config.model.model_name == "purs" else evaluate_bpr

        best_ndcg = 0.0
        patience_counter = 0

        for epoch in range(1, self.config.model.epochs + 1):
            # Train one epoch
            avg_loss = train_fn(
                model=self.model,
                train_loader=self.train_loader,
                optimizer=self.optimizer,
                device=self.device,
                epoch=epoch,
                verbose=self.config.experiment.verbose,
            )

            # Log training metrics
            self.metrics_logger.log_train_metrics(epoch, {"loss": avg_loss})

            print(f"Epoch {epoch}/{self.config.model.epochs} - Loss: {avg_loss:.4f}")

            # Evaluate
            if epoch % self.config.experiment.eval_every == 0:
                eval_metrics = eval_fn(
                    model=self.model,
                    eval_loader=self.eval_loader,
                    device=self.device,
                    k_values=self.config.experiment.k_values,
                    verbose=self.config.experiment.verbose,
                )

                # Log evaluation metrics
                self.metrics_logger.log_eval_metrics(epoch, eval_metrics)

                # Print metrics
                print(f"  Evaluation:")
                for metric, value in eval_metrics.items():
                    print(f"    {metric}: {value:.4f}")

                # Early stopping check
                current_ndcg = eval_metrics.get("ndcg@10", 0.0)
                if current_ndcg > best_ndcg:
                    best_ndcg = current_ndcg
                    patience_counter = 0

                    # Save best model
                    self.save_checkpoint(epoch, is_best=True)
                else:
                    patience_counter += 1

                if patience_counter >= self.config.experiment.early_stopping_patience:
                    print(f"\nEarly stopping triggered at epoch {epoch}")
                    break

            # Save checkpoint
            if epoch % self.config.experiment.checkpoint_every == 0:
                self.save_checkpoint(epoch, is_best=False)

        print("=" * 60)
        print(f"✓ Training complete. Best NDCG@10: {best_ndcg:.4f}")
        print("=" * 60)

        return best_ndcg

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint_dir = self.config.paths.models
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config.to_dict(),
        }

        # Save regular checkpoint
        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)

        # Save best model
        if is_best:
            best_path = checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"  ✓ Best model saved to {best_path}")

    def load_checkpoint(self, checkpoint_path: Path):
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"✓ Loaded checkpoint from {checkpoint_path}")

    def run(self):
        """
        Run complete experiment pipeline.

        Returns:
            Dictionary with final results
        """
        start_time = datetime.now()

        print("\n" + "=" * 60)
        print(f"EXPERIMENT: {self.config.experiment.experiment_name}")
        print("=" * 60)
        print_environment_info()

        # Run pipeline
        self.load_data()
        self.create_dataloaders()
        self.create_model()
        best_ndcg = self.train()

        # Final evaluation
        print("\n" + "=" * 60)
        print("FINAL EVALUATION")
        print("=" * 60)

        # Use appropriate evaluation function
        eval_fn = evaluate_purs if self.config.model.model_name == "purs" else evaluate_bpr

        final_metrics = eval_fn(
            model=self.model,
            eval_loader=self.eval_loader,
            device=self.device,
            k_values=self.config.experiment.k_values,
            verbose=True,
        )

        print("\nFinal Results:")
        for metric, value in final_metrics.items():
            print(f"  {metric}: {value:.4f}")

        # Save results
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        results = {
            "experiment_name": self.config.experiment.experiment_name,
            "environment": self.env,
            "device": self.device,
            "duration_seconds": duration,
            "best_ndcg@10": best_ndcg,
            **final_metrics,
        }

        # Log final results
        self.metrics_logger.log_final_results(results)

        # Save encoder
        encoder_path = self.config.paths.models / "id_encoder.pkl"
        self.encoder.save(encoder_path)

        print(f"\n✓ Experiment complete in {duration/60:.1f} minutes")
        print(f"✓ Results saved to {self.metrics_logger.get_experiment_dir()}")
        print("=" * 60 + "\n")

        return results


def main():
    """Main entry point for running experiments."""
    import argparse

    parser = argparse.ArgumentParser(description="Run BPR recommendation experiment")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help='Config name ("local", "kaggle") or path to config file',
    )
    parser.add_argument(
        "--data-subset",
        type=float,
        default=None,
        help="Fraction of data to use (for testing)",
    )

    args = parser.parse_args()

    # Load config
    if args.config and Path(args.config).exists():
        config = Config.from_yaml(Path(args.config))
    else:
        config = load_config(args.config)

    # Override data subset if specified
    if args.data_subset is not None:
        config.data.data_subset = args.data_subset

    # Run experiment
    runner = ExperimentRunner(config)
    results = runner.run()

    return results


if __name__ == "__main__":
    main()
