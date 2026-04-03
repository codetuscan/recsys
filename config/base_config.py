"""
Base configuration classes using dataclasses for type safety.
"""

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Literal
import yaml


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""

    dataset_name: str = "movielens-32m"
    min_interactions_per_user: int = 5
    min_interactions_per_item: int = 5
    test_ratio: float = 0.2
    negative_samples_train: int = 1
    negative_samples_eval: int = 99
    use_temporal_split: bool = True  # Leave-one-out temporal split
    data_subset: Optional[float] = None  # For testing: use fraction of data


@dataclass
class ModelConfig:
    """Configuration for model architecture and training."""

    model_name: Literal["bpr", "purs"] = "bpr"
    embedding_dim: int = 64
    learning_rate: float = 0.001
    regularization: float = 0.01
    batch_size: int = 2048
    epochs: int = 10
    num_workers: int = 4  # DataLoader workers
    pin_memory: bool = True  # For GPU training

    # PURS-specific parameters
    history_length: int = 10  # Sequence length for GRU (PURS)
    gru_hidden_dim: int = 32  # GRU hidden dimension (PURS)
    num_clusters: int = 10  # Number of clusters for Mean Shift (PURS)
    unexpectedness_weight: float = 0.5  # Scaling factor for unexpectedness component (PURS)


@dataclass
class ExperimentConfig:
    """Configuration for experiment execution."""

    experiment_name: str = "bpr_baseline"
    seed: int = 42
    device: str = "auto"  # "auto", "cpu", "cuda"
    checkpoint_every: int = 2  # Save checkpoint every N epochs
    eval_every: int = 1  # Evaluate every N epochs
    k_values: list[int] = field(default_factory=lambda: [5, 10, 20])  # For metrics@K
    early_stopping_patience: int = 5
    verbose: bool = True


@dataclass
class PathConfig:
    """Configuration for file paths (auto-generated from environment)."""

    raw_data: Path = field(default_factory=Path)
    processed_data: Path = field(default_factory=Path)
    outputs: Path = field(default_factory=Path)
    models: Path = field(default_factory=Path)
    logs: Path = field(default_factory=Path)
    results: Path = field(default_factory=Path)


@dataclass
class Config:
    """Main configuration class combining all sub-configs."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    environment: str = "auto"

    def to_dict(self):
        """Convert config to dictionary."""
        return asdict(self)

    def save(self, path: Path):
        """Save configuration to YAML file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    @classmethod
    def from_dict(cls, config_dict: dict):
        """Create Config from dictionary."""
        return cls(
            data=DataConfig(**config_dict.get("data", {})),
            model=ModelConfig(**config_dict.get("model", {})),
            experiment=ExperimentConfig(**config_dict.get("experiment", {})),
            paths=PathConfig(**config_dict.get("paths", {})),
            environment=config_dict.get("environment", "auto"),
        )

    @classmethod
    def from_yaml(cls, path: Path):
        """Load configuration from YAML file."""
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)


def load_config(
    config_name: Optional[str] = None, config_path: Optional[Path] = None
) -> Config:
    """
    Load configuration with environment-specific defaults.

    Args:
        config_name: Name of config ("local", "kaggle"). If None, auto-detects.
        config_path: Direct path to config file. Overrides config_name.

    Returns:
        Config object with environment-specific settings
    """
    from ..utils.environment import detect_environment, get_data_paths

    # Detect environment if not specified
    if config_name is None:
        config_name = detect_environment()

    # Load from path if specified
    if config_path is not None:
        config = Config.from_yaml(config_path)
    else:
        # Try to load environment-specific config file
        config_dir = Path(__file__).parent
        config_file = config_dir / f"{config_name}_config.yaml"

        if config_file.exists():
            config = Config.from_yaml(config_file)
        else:
            # Use default config
            config = Config()

    # Set environment
    config.environment = config_name

    # Set paths based on environment
    env_paths = get_data_paths(config_name)
    config.paths = PathConfig(
        raw_data=env_paths["raw"],
        processed_data=env_paths["processed"],
        outputs=env_paths["outputs"],
        models=env_paths["models"],
        logs=env_paths["logs"],
        results=env_paths["results"],
    )

    # Auto-detect device if set to "auto"
    if config.experiment.device == "auto":
        import torch

        config.experiment.device = "cuda" if torch.cuda.is_available() else "cpu"

    return config


def create_default_configs():
    """
    Create default local and Kaggle configuration files.
    """
    config_dir = Path(__file__).parent

    # Local config (smaller batch size, fewer epochs for testing)
    local_config = Config(
        model=ModelConfig(batch_size=1024, epochs=5),
        experiment=ExperimentConfig(experiment_name="local_test"),
    )
    local_config.save(config_dir / "local_config.yaml")

    # Kaggle config (larger batch size, more epochs)
    kaggle_config = Config(
        model=ModelConfig(batch_size=4096, epochs=20),
        experiment=ExperimentConfig(experiment_name="kaggle_experiment"),
    )
    kaggle_config.save(config_dir / "kaggle_config.yaml")

    print(f"Created default configs in {config_dir}")


if __name__ == "__main__":
    # Create default config files when run as script
    create_default_configs()
