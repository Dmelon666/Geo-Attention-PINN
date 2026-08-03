"""Geo-Attention CNN model definition."""

from __future__ import annotations

import torch
from torch import nn

from .config import ModelConfig


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel-attention block."""

    def __init__(self, channels: int, reduction: int = 2) -> None:
        super().__init__()
        hidden_channels = max(channels // reduction, 1)

        self.global_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, hidden_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Reweight latent channels using globally pooled channel descriptors."""
        batch_size, channels, _, _ = inputs.shape
        descriptor = self.global_pool(inputs).reshape(batch_size, channels)
        weights = self.excitation(descriptor).reshape(batch_size, channels, 1, 1)
        return inputs * weights


class GeoAttentionCNN(nn.Module):
    """CNN-SE regression backbone used by the discrete Geo-Attention PINN."""

    def __init__(
        self,
        in_channels: int,
        window_size: int,
        config: ModelConfig,
    ) -> None:
        super().__init__()

        if window_size != 5:
            raise ValueError(
                "The current fully connected head is configured for a 5x5 input patch. "
                "Update the flattened dimension if another window size is required."
            )

        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels, config.initial_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(config.initial_channels),
            nn.ReLU(inplace=True),
        )

        self.residual_branch = nn.Sequential(
            nn.Conv2d(
                config.initial_channels,
                config.initial_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(config.initial_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                config.initial_channels,
                config.initial_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(config.initial_channels),
        )

        self.channel_attention = SEBlock(
            channels=config.initial_channels,
            reduction=config.se_reduction,
        )
        self.residual_activation = nn.ReLU(inplace=True)

        self.downsample = nn.Sequential(
            nn.Conv2d(
                config.initial_channels,
                config.output_channels,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(config.output_channels),
            nn.ReLU(inplace=True),
        )

        flattened_units = config.output_channels * 3 * 3
        self.regression_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_units, config.hidden_units),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_units, 1),
            nn.Sigmoid(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Predict center-cell prospectivity from a multi-channel local patch."""
        shallow_features = self.initial_conv(inputs)
        residual_features = self.residual_branch(shallow_features)
        attended_features = self.channel_attention(residual_features)
        fused_features = self.residual_activation(attended_features + shallow_features)
        downsampled_features = self.downsample(fused_features)
        return self.regression_head(downsampled_features)
