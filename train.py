"""End-to-end entry point for the refactored Geo-Attention PINN project."""

from __future__ import annotations

import pandas as pd
import torch

from geo_attention_pinn.artifacts import (
    ensure_output_directory,
    save_checkpoint,
    save_configuration,
    save_history,
    save_metrics,
    save_predictions,
)
from geo_attention_pinn.config import ExperimentConfig
from geo_attention_pinn.data import prepare_data
from geo_attention_pinn.metrics import regression_metrics
from geo_attention_pinn.model import GeoAttentionCNN
from geo_attention_pinn.reproducibility import resolve_device, set_global_seed
from geo_attention_pinn.trainer import predict_center_patches, train_model
from geo_attention_pinn.visualization import (
    save_prediction_maps,
    save_training_history,
)


def _build_split_labels(num_samples: int, data) -> pd.Series:
    """Build a human-readable split label for each supervised sample."""
    labels = pd.Series(["unassigned"] * num_samples, dtype="object")
    labels.iloc[data.splits.train] = "train"
    labels.iloc[data.splits.val] = "validation"
    labels.iloc[data.splits.test] = "test"
    return labels


def main() -> None:
    """Run preprocessing, training, independent evaluation, and artifact export."""
    config = ExperimentConfig()
    config.validate()

    set_global_seed(config.data.random_seed)
    device = resolve_device(config.training.device)
    output_dir = ensure_output_directory(config)

    print(f"Using device: {device}")
    print(f"Loading dataset from: {config.data.csv_path}")

    data = prepare_data(config.data)
    permeability_channel_index = config.data.features.index(
        config.physics.permeability_feature
    )
    patch_center_index = config.data.window_size // 2

    model = GeoAttentionCNN(
        in_channels=len(config.data.features),
        window_size=config.data.window_size,
        config=config.model,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    model, history, best_validation_loss = train_model(
        model=model,
        train_loader=data.train_loader,
        val_loader=data.val_loader,
        optimizer=optimizer,
        device=device,
        permeability_channel_index=permeability_channel_index,
        patch_center_index=patch_center_index,
        lambda_physics=config.physics.lambda_physics,
        epochs=config.training.epochs,
        patience=config.training.patience,
        gradient_clip_norm=config.training.gradient_clip_norm,
    )

    predictions_all = predict_center_patches(
        model=model,
        center_patches=data.patches.center,
        batch_size=config.data.batch_size,
        device=device,
    )

    split_predictions = {
        "train": predictions_all[data.splits.train],
        "validation": predictions_all[data.splits.val],
        "test": predictions_all[data.splits.test],
    }
    split_targets = {
        "train": data.patches.targets[data.splits.train].reshape(-1),
        "validation": data.patches.targets[data.splits.val].reshape(-1),
        "test": data.patches.targets[data.splits.test].reshape(-1),
    }

    metrics = {
        split_name: regression_metrics(
            split_targets[split_name],
            split_predictions[split_name],
        )
        for split_name in ("train", "validation", "test")
    }

    print("\nIndependent test metrics")
    for metric_name, metric_value in metrics["test"].items():
        print(f"  {metric_name.upper():>4s}: {metric_value:.6f}")

    split_labels = _build_split_labels(len(data.frame), data)
    save_configuration(config, output_dir)
    save_checkpoint(model, config, best_validation_loss, output_dir)
    save_metrics(metrics, config, output_dir)
    save_history(history, config, output_dir)
    save_predictions(
        data=data,
        predictions_all=pd.Series(predictions_all),
        split_labels=split_labels,
        config=config,
        output_dir=output_dir,
    )
    save_prediction_maps(
        frame=data.frame,
        predictions=predictions_all,
        config=config.data,
        output_path=output_dir / "prediction_maps.png",
    )
    save_training_history(
        history=history,
        output_path=output_dir / "training_history.png",
    )

    print(f"\nAll experiment artifacts were saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
