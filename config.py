"""Centralized experiment configuration.

All frequently changed hyperparameters are defined here so that model, data,
physics, and training code remain independent from experiment-specific settings.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class DataConfig:
    """Configuration for loading, gridding, normalizing, and splitting data."""

    csv_path: Path = Path("../RegressionData.csv")
    features: tuple[str, ...] = (
        "Thickness",
        "AvgNTG",
        "AvgPerm",
        "AvgPoro",
        "NetPerm",
    )
    target_column: str = "WellProbability"
    x_column: str = "XPos"
    y_column: str = "YPos"

    window_size: int = 5
    train_ratio: float = 0.60
    val_ratio: float = 0.20
    test_ratio: float = 0.20
    random_seed: int = 42

    batch_size: int = 64
    num_workers: int = 0
    pin_memory: bool = True

    normalization_epsilon: float = 1.0e-8
    normalization_scope: Literal["train_samples", "full_grid"] = "train_samples"

    def validate(self) -> None:
        """Validate data-related configuration values before execution."""
        ratio_sum = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(ratio_sum - 1.0) > 1.0e-8:
            raise ValueError(
                "train_ratio + val_ratio + test_ratio must equal 1.0; "
                f"received {ratio_sum:.6f}."
            )
        if self.window_size < 3 or self.window_size % 2 == 0:
            raise ValueError("window_size must be an odd integer >= 3.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for the Geo-Attention CNN backbone."""

    initial_channels: int = 16
    output_channels: int = 32
    se_reduction: int = 2
    hidden_units: int = 64
    dropout: float = 0.30


@dataclass(frozen=True)
class PhysicsConfig:
    """Configuration for the Darcy-type discrete physics constraint."""

    permeability_feature: str = "AvgPerm"
    lambda_physics: float = 0.05


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for optimization, early stopping, and output management."""

    epochs: int = 500
    patience: int = 15
    learning_rate: float = 1.0e-3
    weight_decay: float = 0.0
    gradient_clip_norm: float | None = None
    device: Literal["auto", "cpu", "cuda"] = "auto"

    output_dir: Path = Path("outputs")
    checkpoint_name: str = "best_model.pt"
    metrics_name: str = "metrics.json"
    predictions_name: str = "predictions.csv"
    history_name: str = "training_history.csv"


@dataclass(frozen=True)
class ExperimentConfig:
    """Top-level configuration object for the complete experiment."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def validate(self) -> None:
        """Validate cross-module configuration consistency."""
        self.data.validate()
        if self.physics.permeability_feature not in self.data.features:
            raise ValueError(
                "The permeability feature used by the physics loss must be included "
                f"in data.features. Received: {self.physics.permeability_feature!r}."
            )
        if self.physics.lambda_physics < 0:
            raise ValueError("lambda_physics must be non-negative.")
        if self.training.epochs <= 0:
            raise ValueError("epochs must be positive.")
        if self.training.patience <= 0:
            raise ValueError("patience must be positive.")

    def to_dict(self) -> dict:
        """Return a JSON-serializable configuration dictionary."""
        config_dict = asdict(self)
        config_dict["data"]["csv_path"] = str(self.data.csv_path)
        config_dict["training"]["output_dir"] = str(self.training.output_dir)
        return config_dict
