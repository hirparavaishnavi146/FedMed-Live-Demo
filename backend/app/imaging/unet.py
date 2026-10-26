"""A configurable 3D U-Net for volumetric MRI segmentation.

Tensor shapes (this is the documented contract, and the tests assert every line):

    input      (N, in_channels, D, H, W)   float32, logits-free raw intensities
    output     (N, out_channels, D, H, W)  float32, **raw logits**

The output is logits, not probabilities. `BCEWithLogitsLoss` and
`nn.CrossEntropyLoss` both expect logits for numerical stability, and Dice is
computed after a `sigmoid` at a chosen threshold (see `metrics.py`). Returning
probabilities here would silently double-apply a sigmoid in training.

Spatial dimensions must be divisible by ``2 ** len(features)``. Each encoder
stage halves D/H/W with a stride-2 pooling, and the bottleneck halves once more,
so a size that is not divisible cannot be reassembled by the decoder. That
requirement is validated at construction time with an explicit error rather than
failing deep inside a forward pass.

WHY THE DEFAULTS ARE SMALL
    The reference machine has no CUDA GPU and 7.9 GB of RAM (see
    docs/IMAGING.md). The default 3-stage, 32-base-channel network at 32^3 voxels
    is sized so a training step completes on that CPU in seconds. It is a real
    3D U-Net with real gradients; it is not sized for BraTS-scale volumes.

WHAT THIS IS NOT
    Training this network does not make the project clinically usable, and no
    metric it produces is a clinical result. The data it is tested on is
    synthetic. See the disclaimer in docs/IMAGING.md.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

# Defaults for the CPU-only reference machine.
DEFAULT_FEATURES: tuple[int, ...] = (8, 16, 32)
DEFAULT_IN_CHANNELS = 1  # single MRI modality (e.g. T1)
DEFAULT_OUT_CHANNELS = 1  # binary foreground/background
DEFAULT_DROPOUT = 0.0


class ConvBlock3D(nn.Module):
    """Two 3x3x3 convolutions with normalization and activation.

    GroupNorm rather than BatchNorm: 3D segmentation is trained with a small batch
    (often 1-2 on this hardware), and BatchNorm's running statistics are unstable
    at those sizes. GroupNorm is batch-independent, so a batch of 1 behaves the
    same as a batch of 8.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        # 8 groups, or fewer when there are fewer channels.
        groups = min(8, out_channels)
        while out_channels % groups != 0:
            groups -= 1
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down3D(nn.Module):
    """Halve the spatial size, then extract features."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv = ConvBlock3D(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class Up3D(nn.Module):
    """Double the spatial size, concatenate the skip connection, then convolve."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size=2, stride=2
        )
        self.conv = ConvBlock3D(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        upsampled = self.up(x)
        if upsampled.shape[-3:] != skip.shape[-3:]:
            # Only reachable if a spatial size was not divisible as required. Fail
            # loudly rather than concatenating mismatched volumes along the wrong
            # axis, which torch would reject in a confusing way.
            raise ValueError(
                "Upsampled features "
                f"{tuple(upsampled.shape[-3:])} do not match skip connection "
                f"{tuple(skip.shape[-3:])}. Every spatial dimension must be "
                "divisible by 2 ** (number of encoder stages - 1)."
            )
        return self.conv(torch.cat((skip, upsampled), dim=1))


class UNet3D(nn.Module):
    """Configurable 3D U-Net.

    Args:
        in_channels: input channels (1 for a single MRI modality).
        out_channels: output channels (1 for binary segmentation).
        features: channel width per encoder stage. Its length sets the depth.
        dropout: dropout probability applied at the bottleneck. 0 disables it.

    Shape contract:
        input  (N, in_channels, D, H, W) -> output (N, out_channels, D, H, W)
        D, H, W must each be divisible by ``2 ** len(features)``.
    """

    def __init__(
        self,
        in_channels: int = DEFAULT_IN_CHANNELS,
        out_channels: int = DEFAULT_OUT_CHANNELS,
        features: tuple[int, ...] = DEFAULT_FEATURES,
        dropout: float = DEFAULT_DROPOUT,
    ) -> None:
        super().__init__()
        if not features:
            raise ValueError("features must contain at least one stage.")
        if any(width <= 0 for width in features):
            raise ValueError(f"every feature width must be positive, got {features}.")
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("in_channels and out_channels must be positive.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}.")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.features = tuple(features)
        # The bottleneck adds one more downsampling stage than there are feature
        # widths, so every spatial dimension must be divisible by 2 ** stages.
        self.size_multiple = 2 ** len(self.features)

        widths = [in_channels, *self.features]

        # Encoder: one feature map per requested width, halving the resolution
        # each time after the first. These are the skip connections.
        self.stem = ConvBlock3D(widths[0], widths[1])
        self.downs = nn.ModuleList(
            Down3D(widths[index], widths[index + 1]) for index in range(1, len(widths) - 1)
        )

        # Bottleneck: one further halving below the deepest encoder feature map.
        # Without this the first decoder stage would upsample to a resolution the
        # deepest skip is not at, and the concatenation would fail.
        bottleneck = self.features[-1] * 2
        self.bottleneck = Down3D(self.features[-1], bottleneck)
        self.dropout = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()

        # Decoder mirrors the encoder: each stage upsamples by 2 and concatenates
        # the encoder feature map at that resolution, deepest first.
        reversed_features = list(reversed(self.features))
        self.ups = nn.ModuleList()
        current = bottleneck
        for skip_channels, out_channels_here in zip(reversed_features, reversed_features):
            self.ups.append(Up3D(current, skip_channels, out_channels_here))
            current = out_channels_here

        # 1x1x1 projection to the requested number of output channels, producing
        # logits. No activation here by design.
        self.head = nn.Conv3d(current, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 5:
            raise ValueError(
                f"expected a 5D tensor (N, C, D, H, W), received {x.dim()}D with "
                f"shape {tuple(x.shape)}."
            )
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"expected {self.in_channels} input channel(s), received {x.shape[1]}."
            )
        spatial = x.shape[-3:]
        bad = [
            (axis, size)
            for axis, size in zip(("D", "H", "W"), spatial)
            if size % self.size_multiple != 0
        ]
        if bad:
            raise ValueError(
                f"spatial size {tuple(spatial)} is not divisible by "
                f"{self.size_multiple} (2 ** stages); offending axes: "
                f"{[f'{axis}={size}' for axis, size in bad]}."
            )

        skips: list[torch.Tensor] = [self.stem(x)]
        for down in self.downs:
            skips.append(down(skips[-1]))

        features = self.dropout(self.bottleneck(skips[-1]))

        for up, skip in zip(self.ups, reversed(skips)):
            features = up(features, skip)

        return self.head(features)

    def describe(self) -> dict[str, Any]:
        """Architecture facts, published by the API so they can be inspected."""
        return {
            "name": "UNet3D",
            "inputShape": ["N", self.in_channels, "D", "H", "W"],
            "outputShape": ["N", self.out_channels, "D", "H", "W"],
            "features": list(self.features),
            "stages": len(self.features),
            "sizeMultiple": self.size_multiple,
            "outputIsLogits": True,
            "parameters": parameter_count(self),
            "dropout": DEFAULT_DROPOUT,
        }


def parameter_count(model: nn.Module) -> int:
    """Total trainable scalars."""
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


def build_unet(seed: int | None = None, **kwargs: Any) -> UNet3D:
    """Construct a U-Net, optionally with reproducible initial weights.

    The global RNG state is saved and restored around construction, so building a
    seeded model is reproducible *and* leaves a caller's own sequence untouched
    (the training loop seeds its data shuffling independently).

    Seeding every parameter, not only the >= 2-D ones: an earlier version re-drew
    only the convolution weights and left biases at whatever the global RNG
    produced at construction time, so two calls with the same seed returned
    different models. GroupNorm's 1-D weight (default 1) and bias (default 0) must
    also keep their defaults — zeroing them would silence those layers.
    """
    state = torch.random.get_rng_state()
    if seed is not None:
        torch.manual_seed(int(seed))
    try:
        return UNet3D(**kwargs)
    finally:
        torch.random.set_rng_state(state)
