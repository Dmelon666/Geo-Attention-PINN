"""Darcy-type five-point discrete physics residual."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class DirectionalPredictions:
    """Network predictions for center and four orthogonal neighboring patches."""

    center: torch.Tensor
    right: torch.Tensor
    left: torch.Tensor
    bottom: torch.Tensor
    top: torch.Tensor


def predict_five_directions(
    model: nn.Module,
    center: torch.Tensor,
    right: torch.Tensor,
    left: torch.Tensor,
    bottom: torch.Tensor,
    top: torch.Tensor,
) -> DirectionalPredictions:
    """Evaluate all five directional patches with one shared-CNN forward pass.

    Concatenating the directional batches is faster than five independent calls
    and prevents BatchNorm running statistics from being updated five times for
    one optimization batch.
    """
    batch_size = center.shape[0]
    stacked_inputs = torch.cat((center, right, left, bottom, top), dim=0)
    stacked_predictions = model(stacked_inputs)

    p_center, p_right, p_left, p_bottom, p_top = torch.split(
        stacked_predictions,
        batch_size,
        dim=0,
    )
    return DirectionalPredictions(
        center=p_center,
        right=p_right,
        left=p_left,
        bottom=p_bottom,
        top=p_top,
    )


def _positive_center_permeability(
    patches: torch.Tensor,
    permeability_channel_index: int,
    patch_center_index: int,
) -> torch.Tensor:
    """Extract standardized center permeability and map it to a positive coefficient."""
    standardized_permeability = patches[
        :,
        permeability_channel_index,
        patch_center_index,
        patch_center_index,
    ].unsqueeze(1)
    return F.softplus(standardized_permeability)


def darcy_five_point_residual(
    predictions: DirectionalPredictions,
    center_patches: torch.Tensor,
    right_patches: torch.Tensor,
    left_patches: torch.Tensor,
    bottom_patches: torch.Tensor,
    top_patches: torch.Tensor,
    permeability_channel_index: int,
    patch_center_index: int,
) -> torch.Tensor:
    """Compute the transmissibility-weighted Darcy-type five-point residual.

    Arithmetic averaging reconstructs relative interface transmissibilities,
    preserving the mathematical formulation of the original implementation.
    """
    k_center = _positive_center_permeability(
        center_patches,
        permeability_channel_index,
        patch_center_index,
    )
    k_right = _positive_center_permeability(
        right_patches,
        permeability_channel_index,
        patch_center_index,
    )
    k_left = _positive_center_permeability(
        left_patches,
        permeability_channel_index,
        patch_center_index,
    )
    k_bottom = _positive_center_permeability(
        bottom_patches,
        permeability_channel_index,
        patch_center_index,
    )
    k_top = _positive_center_permeability(
        top_patches,
        permeability_channel_index,
        patch_center_index,
    )

    transmissibility_right = 0.5 * (k_center + k_right)
    transmissibility_left = 0.5 * (k_center + k_left)
    transmissibility_bottom = 0.5 * (k_center + k_bottom)
    transmissibility_top = 0.5 * (k_center + k_top)

    return (
        transmissibility_right * (predictions.right - predictions.center)
        + transmissibility_left * (predictions.left - predictions.center)
        + transmissibility_bottom * (predictions.bottom - predictions.center)
        + transmissibility_top * (predictions.top - predictions.center)
    )


def physics_loss(residual: torch.Tensor) -> torch.Tensor:
    """Return the mean squared Darcy-type residual."""
    return torch.mean(residual.square())
