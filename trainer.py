"""Training, validation, early stopping, and batched inference."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from .physics import (
    darcy_five_point_residual,
    physics_loss,
    predict_five_directions,
)


@dataclass
class EpochStatistics:
    """Mean loss values accumulated over one complete epoch."""

    total: float
    data: float
    physics: float


@dataclass
class TrainingHistory:
    """Serializable per-epoch training history."""

    epoch: list[int] = field(default_factory=list)
    train_total: list[float] = field(default_factory=list)
    train_data: list[float] = field(default_factory=list)
    train_physics: list[float] = field(default_factory=list)
    val_total: list[float] = field(default_factory=list)
    val_data: list[float] = field(default_factory=list)
    val_physics: list[float] = field(default_factory=list)

    def append(
        self,
        epoch: int,
        train_stats: EpochStatistics,
        val_stats: EpochStatistics,
    ) -> None:
        """Append one epoch of training and validation statistics."""
        self.epoch.append(epoch)
        self.train_total.append(train_stats.total)
        self.train_data.append(train_stats.data)
        self.train_physics.append(train_stats.physics)
        self.val_total.append(val_stats.total)
        self.val_data.append(val_stats.data)
        self.val_physics.append(val_stats.physics)

    def to_frame(self) -> pd.DataFrame:
        """Convert the history to a DataFrame for CSV export."""
        return pd.DataFrame(
            {
                "epoch": self.epoch,
                "train_total": self.train_total,
                "train_data": self.train_data,
                "train_physics": self.train_physics,
                "val_total": self.val_total,
                "val_data": self.val_data,
                "val_physics": self.val_physics,
            }
        )


class EarlyStopping:
    """Retain the best validation state and stop after a patience interval."""

    def __init__(self, patience: int) -> None:
        self.patience = patience
        self.best_loss = float("inf")
        self.bad_epochs = 0
        self.best_state: dict[str, torch.Tensor] | None = None

    def update(self, validation_loss: float, model: nn.Module) -> bool:
        """Update the tracker and return True if training should terminate."""
        if validation_loss < self.best_loss:
            self.best_loss = validation_loss
            self.bad_epochs = 0
            self.best_state = copy.deepcopy(model.state_dict())
        else:
            self.bad_epochs += 1

        return self.bad_epochs >= self.patience


def _move_batch_to_device(
    batch: tuple[torch.Tensor, ...],
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    """Move all directional patches and targets to the compute device."""
    return tuple(tensor.to(device, non_blocking=True) for tensor in batch)


def _compute_batch_losses(
    model: nn.Module,
    batch: tuple[torch.Tensor, ...],
    criterion: nn.Module,
    permeability_channel_index: int,
    patch_center_index: int,
    lambda_physics: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute supervised, physical, and total losses for one batch."""
    center, right, left, bottom, top, targets = batch

    predictions = predict_five_directions(
        model=model,
        center=center,
        right=right,
        left=left,
        bottom=bottom,
        top=top,
    )

    data_loss = criterion(predictions.center, targets)
    residual = darcy_five_point_residual(
        predictions=predictions,
        center_patches=center,
        right_patches=right,
        left_patches=left,
        bottom_patches=bottom,
        top_patches=top,
        permeability_channel_index=permeability_channel_index,
        patch_center_index=patch_center_index,
    )
    physical_loss = physics_loss(residual)
    total_loss = data_loss + lambda_physics * physical_loss

    return total_loss, data_loss, physical_loss


def _run_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    permeability_channel_index: int,
    patch_center_index: int,
    lambda_physics: float,
    optimizer: torch.optim.Optimizer | None,
    gradient_clip_norm: float | None,
) -> EpochStatistics:
    """Run one training or validation epoch.

    Passing an optimizer enables gradient updates. Passing ``None`` switches the
    function to inference-only validation mode.
    """
    is_training = optimizer is not None
    model.train(mode=is_training)

    total_sum = 0.0
    data_sum = 0.0
    physics_sum = 0.0
    sample_count = 0

    context = torch.enable_grad() if is_training else torch.inference_mode()
    with context:
        for raw_batch in data_loader:
            batch = _move_batch_to_device(raw_batch, device)
            batch_size = batch[0].shape[0]

            if is_training:
                optimizer.zero_grad(set_to_none=True)

            total_loss, data_loss, physical_loss = _compute_batch_losses(
                model=model,
                batch=batch,
                criterion=criterion,
                permeability_channel_index=permeability_channel_index,
                patch_center_index=patch_center_index,
                lambda_physics=lambda_physics,
            )

            if is_training:
                total_loss.backward()
                if gradient_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=gradient_clip_norm,
                    )
                optimizer.step()

            total_sum += total_loss.item() * batch_size
            data_sum += data_loss.item() * batch_size
            physics_sum += physical_loss.item() * batch_size
            sample_count += batch_size

    if sample_count == 0:
        raise RuntimeError("The DataLoader contains no samples.")

    return EpochStatistics(
        total=total_sum / sample_count,
        data=data_sum / sample_count,
        physics=physics_sum / sample_count,
    )


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    permeability_channel_index: int,
    patch_center_index: int,
    lambda_physics: float,
    epochs: int,
    patience: int,
    gradient_clip_norm: float | None = None,
) -> tuple[nn.Module, TrainingHistory, float]:
    """Train the model with validation-based early stopping."""
    criterion = nn.MSELoss()
    early_stopping = EarlyStopping(patience=patience)
    history = TrainingHistory()

    for epoch in range(1, epochs + 1):
        train_stats = _run_epoch(
            model=model,
            data_loader=train_loader,
            criterion=criterion,
            device=device,
            permeability_channel_index=permeability_channel_index,
            patch_center_index=patch_center_index,
            lambda_physics=lambda_physics,
            optimizer=optimizer,
            gradient_clip_norm=gradient_clip_norm,
        )
        val_stats = _run_epoch(
            model=model,
            data_loader=val_loader,
            criterion=criterion,
            device=device,
            permeability_channel_index=permeability_channel_index,
            patch_center_index=patch_center_index,
            lambda_physics=lambda_physics,
            optimizer=None,
            gradient_clip_norm=None,
        )

        history.append(epoch, train_stats, val_stats)

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"Epoch [{epoch:03d}/{epochs}] | "
                f"Train Total: {train_stats.total:.6f} | "
                f"Val Total: {val_stats.total:.6f}"
            )

        if early_stopping.update(val_stats.total, model):
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    if early_stopping.best_state is None:
        raise RuntimeError("No valid model state was recorded during training.")

    model.load_state_dict(early_stopping.best_state)
    return model, history, early_stopping.best_loss


def predict_center_patches(
    model: nn.Module,
    center_patches: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Predict center-cell prospectivity in memory-safe batches."""
    model.eval()
    prediction_batches: list[np.ndarray] = []

    with torch.inference_mode():
        for start in range(0, len(center_patches), batch_size):
            stop = min(start + batch_size, len(center_patches))
            inputs = torch.from_numpy(center_patches[start:stop]).to(
                device,
                non_blocking=True,
            )
            batch_predictions = model(inputs).cpu().numpy().reshape(-1)
            prediction_batches.append(batch_predictions)

    return np.concatenate(prediction_batches, axis=0)
