"""MRI preprocessing: loading, orientation and spacing normalization, intensity
normalization, spatial resizing and label preparation.

Built on MONAI's dictionary transforms, with the installed API verified against
`monai 1.6.1` rather than assumed.

PIPELINE ORDER, AND WHY EACH STEP IS THERE
    1. `LoadImaged`            read NIfTI; keep the affine so orientation and
                               spacing can be reasoned about
    2. `EnsureChannelFirstd`   add the channel axis the network expects: (D,H,W)
                               -> (1,D,H,W)
    3. `Orientationd`          reorient to a canonical axis code, so two sites
                               that scanned the same anatomy in different patient
                               positions produce comparable arrays
    4. `Spacingd`              resample to a common voxel size. Without this,
                               voxels mean different physical sizes at different
                               sites and a convolution sees different anatomy
    5. `ResizeWithPadOrCropd`  pad or crop to a fixed shape the network accepts
    6. `NormalizeIntensityd`   z-score, computed over non-zero voxels only, since
                               MRI intensity is not calibrated and the air around
                               the head would otherwise dominate mean and std
    7. label binary step       any label value > 0 becomes foreground, so a
                               multi-class or partially-labelled mask still yields
                               a usable binary target (class 0 is background)

    Normalization is applied **after** the resize, so it standardizes the exact
    tensor the network receives. Normalizing a whole volume and then cropping it
    leaves the input to the model unstandardized, because the cropped region's
    statistics differ from the ones the normalization measured.

    Interpolation differs per key and this is not cosmetic: the image is resampled
    bilinearly, but the label uses **nearest neighbour**. Interpolating a label
    would invent fractional class values and blur boundaries, silently corrupting
    every Dice score computed against it.

WHAT IS NOT HERE
    No registration, no bias-field correction, no skull stripping, and no
    augmentation. Each of those changes the data and would need its own
    justification and tests. This module is the preprocessing the tests cover,
    not a claim to be a complete clinical pipeline.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from monai.transforms import (
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    EnsureTyped,
    Lambdad,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    ResizeWithPadOrCropd,
    Spacingd,
)

# Stated explicitly so the network's divisibility requirement is satisfied by
# construction: every default target shape is divisible by 8 (2 ** 3 stages).
DEFAULT_TARGET_SHAPE = (32, 32, 32)
DEFAULT_TARGET_SPACING = (1.0, 1.0, 1.0)
DEFAULT_ORIENTATION = "RAS"


@dataclass(frozen=True)
class PreprocessConfig:
    """Every choice that changes the processed voxels.

    Recorded alongside results, because "preprocessed with MONAI" is not a
    reproducible description on its own.
    """

    target_spacing: tuple[float, float, float] = DEFAULT_TARGET_SPACING
    target_shape: tuple[int, int, int] = DEFAULT_TARGET_SHAPE
    orientation: str = DEFAULT_ORIENTATION
    normalize: bool = True
    normalize_nonzero: bool = True
    crop_foreground: bool = False
    foreground_margin: int = 0
    label_threshold: float = 0.0

    def __post_init__(self) -> None:
        if len(self.target_spacing) != 3 or any(s <= 0 for s in self.target_spacing):
            raise ValueError(
                f"target_spacing must be three positive values, got {self.target_spacing}."
            )
        if len(self.target_shape) != 3 or any(s <= 0 for s in self.target_shape):
            raise ValueError(
                f"target_shape must be three positive values, got {self.target_shape}."
            )

    def as_dict(self) -> dict[str, Any]:
        data = dataclasses.asdict(self)
        data["target_spacing"] = list(self.target_spacing)
        data["target_shape"] = list(self.target_shape)
        return data

    def describe(self) -> str:
        return (
            f"orientation={self.orientation}, spacing={self.target_spacing}, "
            f"shape={self.target_shape}, normalize={self.normalize}"
            f"{'(nonzero)' if self.normalize_nonzero else ''}, "
            f"crop_foreground={self.crop_foreground}"
        )

    @property
    def pipeline_steps(self) -> list[str]:
        """The steps actually applied, for the run's provenance record."""
        steps = [
            "LoadImaged",
            "EnsureChannelFirstd",
            f"Orientationd(axcodes={self.orientation})",
            f"Spacingd(pixdim={self.target_spacing}, mode=bilinear/nearest)",
        ]
        if self.crop_foreground:
            steps.append(f"CropForegroundd(margin={self.foreground_margin})")
        steps.append(f"ResizeWithPadOrCropd(spatial_size={self.target_shape})")
        if self.normalize:
            steps.append(
                f"NormalizeIntensityd(nonzero={self.normalize_nonzero}, channel_wise=True)"
            )
        steps.append(f"Lambdad(label: x > {self.label_threshold})")
        return steps


def build_preprocessing_pipeline(config: PreprocessConfig, with_label: bool = True) -> Compose:
    """Assemble the MONAI transform chain for the given configuration.

    `with_label=False` builds an image-only pipeline, for inference on data that
    has no reference mask. The label steps are then omitted rather than pointed at
    a missing key, which would fail inside LoadImaged.
    """
    image_keys = ["image"]
    both_keys = ["image", "label"] if with_label else ["image"]

    transforms: list[Any] = [
        LoadImaged(keys=both_keys, image_only=False),
        EnsureChannelFirstd(keys=both_keys),
        Orientationd(keys=both_keys, axcodes=config.orientation),
        Spacingd(
            keys=both_keys,
            pixdim=tuple(config.target_spacing),
            # bilinear for the image, nearest for the label: see the module docstring.
            mode=("bilinear", "nearest") if with_label else "bilinear",
        ),
    ]

    if config.crop_foreground:
        transforms.append(
            CropForegroundd(
                keys=both_keys,
                source_key="image",
                margin=config.foreground_margin,
            )
        )

    # Resize BEFORE normalizing, deliberately. Normalizing first standardizes the
    # whole acquired volume, and the crop that follows then selects a sub-region
    # whose statistics differ from the ones the normalization used — so the tensor
    # the network actually receives is not standardized at all. Normalizing after
    # the resize standardizes the exact input the model consumes.
    transforms.append(
        ResizeWithPadOrCropd(keys=both_keys, spatial_size=tuple(config.target_shape))
    )

    if config.normalize:
        transforms.append(
            NormalizeIntensityd(
                keys=image_keys,
                nonzero=config.normalize_nonzero,
                channel_wise=True,
            )
        )

    if with_label:
        # Binarise the label last, so interpolation has already happened. Doing this
        # before Spacingd would let nearest-neighbour resampling operate on an already
        # binarised mask, which is equivalent here but would break the moment the
        # threshold is not zero.
        transforms.append(
            Lambdad(keys=["label"], func=lambda tensor: (tensor > config.label_threshold).float())
        )

    transforms.append(EnsureTyped(keys=both_keys))

    return Compose(transforms)


def preprocess_case(
    image_path: Path,
    label_path: Path | None,
    config: PreprocessConfig | None = None,
) -> dict[str, Any]:
    """Run the pipeline over one case and return tensors plus provenance.

    Returns:
        image: (1, D, H, W) float32
        label: (1, D, H, W) float32, 0.0 or 1.0 — None when no label was given
        affine / spacing / orientation: what the processed volume actually has,
            read back from the output rather than assumed from the config
        steps: the pipeline steps applied
    """
    config = config or PreprocessConfig()
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"No image at {image_path}.")

    keys = ["image", "label"] if label_path is not None else ["image"]
    data: dict[str, Any] = {"image": str(image_path)}
    if label_path is not None:
        label_path = Path(label_path)
        if not label_path.exists():
            raise FileNotFoundError(f"No label at {label_path}.")
        data["label"] = str(label_path)

    pipeline = build_preprocessing_pipeline(config, with_label=label_path is not None)
    processed = pipeline(data)

    image = processed["image"]
    label = processed.get("label")

    result: dict[str, Any] = {
        "image": torch.as_tensor(np.asarray(image)).float(),
        "label": None if label is None else torch.as_tensor(np.asarray(label)).float(),
        "sourceImage": str(image_path),
        "sourceLabel": None if label_path is None else str(label_path),
        "steps": config.pipeline_steps,
        "config": config.as_dict(),
        "keys": keys,
    }

    # Read the geometry back from the output, so a caller can verify the spacing
    # and orientation steps actually did something instead of trusting the config.
    affine = getattr(image, "affine", None)
    if affine is not None:
        affine_array = np.asarray(affine)
        result["affine"] = affine_array
        # Voxel sizes are the column norms of the 3x3 direction block.
        result["spacing"] = tuple(
            float(np.linalg.norm(affine_array[:3, axis])) for axis in range(3)
        )
    result["shape"] = tuple(int(size) for size in result["image"].shape)

    return result


def processed_spacing(affine: np.ndarray) -> tuple[float, float, float]:
    """Voxel size per axis, derived from an affine matrix."""
    array = np.asarray(affine)
    return tuple(float(np.linalg.norm(array[:3, axis])) for axis in range(3))


def as_network_input(case: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Split a processed case into the (input, target) pair a model consumes.

    The tensors are already (1, D, H, W); this adds nothing and reshapes nothing,
    so a shape surprise here would mean the pipeline produced the wrong rank.
    """
    image = case["image"]
    label = case.get("label")
    if image.dim() != 4:
        raise ValueError(
            f"expected a processed image of rank 4 (C, D, H, W), got shape "
            f"{tuple(image.shape)}. Run it through preprocess_case first."
        )
    return image, label
