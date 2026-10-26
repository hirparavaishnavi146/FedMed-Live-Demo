"""The model the federated clients train locally.

IMPORTANT — WHAT THIS IS NOT
    This is a small convolutional network that learns to segment a synthetic
    sphere inside a 16x16x16 patch. It is a real, trainable model: it has
    parameters, gradients and a loss that decreases, which is what the federated
    machinery needs in order to be exercised end to end.

    It is NOT the MRI segmentation model. The 3D U-Net (PyTorch + MONAI) that
    this project is ultimately about is a later stage and is deliberately not
    implemented here. Nothing in this module reads, accepts or understands MRI
    data, and no result produced with this model says anything about
    segmentation performance on real scans.

Why a stand-in at all: federated averaging has to average *something*. Using a
tiny real model keeps a full round finishable on a CPU-only machine in seconds,
and keeps the learning visible (the loss falls), so the aggregation code is
tested against genuine training rather than against fixed numbers.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
import torch
from torch import nn

# Input patch geometry. Kept small on purpose: CPU-only reference machine.
PATCH_SIZE = 16
IN_CHANNELS = 1
OUT_CHANNELS = 1


class TinyPatchSegmenter(nn.Module):
    """A very small 3D convolutional network.

    Roughly 550 parameters. Sized so that a round over three clients completes in
    well under a second on a CPU, leaving the federated logic as the thing under
    test rather than the arithmetic.
    """

    def __init__(self, in_channels: int = IN_CHANNELS) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(4, 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        # 1x1x1 convolution acting as a per-voxel classifier.
        self.head = nn.Conv3d(4, OUT_CHANNELS, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))


def build_model(seed: int | None = None) -> TinyPatchSegmenter:
    """Construct a model, optionally with reproducible initial weights."""
    if seed is not None:
        # Seeded here so two runs with the same configuration start identically.
        generator = torch.Generator().manual_seed(seed)
        model = TinyPatchSegmenter()
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.copy_(torch.empty_like(parameter).normal_(generator=generator))
        return model
    return TinyPatchSegmenter()


def state_to_numpy(model: nn.Module) -> "OrderedDict[str, np.ndarray]":
    """Model parameters as a name -> NumPy array mapping.

    This is the data that leaves a client. It contains weights only: no training
    examples, no labels, no identifiers. The federated contract is that the
    server never receives anything else.
    """
    return OrderedDict(
        (name, parameter.detach().cpu().numpy().copy())
        for name, parameter in model.state_dict().items()
    )


def numpy_to_state(model: nn.Module, parameters: dict[str, np.ndarray]) -> None:
    """Load aggregated parameters back into a model in place.

    Raises ValueError on any structural mismatch. This is the last line of
    defence: aggregation already validates structure, so reaching an error here
    means something bypassed it.
    """
    current = model.state_dict()
    missing = set(current.keys()) - set(parameters.keys())
    unexpected = set(parameters.keys()) - set(current.keys())
    if missing or unexpected:
        raise ValueError(
            "Aggregated parameters do not match the model: "
            f"missing {sorted(missing)}, unexpected {sorted(unexpected)}"
        )
    for name, array in parameters.items():
        tensor = torch.as_tensor(np.ascontiguousarray(array))
        if tuple(tensor.shape) != tuple(current[name].shape):
            raise ValueError(
                f"Parameter '{name}' has shape {list(tensor.shape)} but the model "
                f"expects {list(current[name].shape)}"
            )
        current[name].copy_(tensor)


def parameter_count(model: nn.Module) -> int:
    """Total number of trainable scalars, reported in the API so the model's
    small size is visible rather than implied."""
    return int(sum(parameter.numel() for parameter in model.parameters()))
