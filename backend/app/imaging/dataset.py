"""File-backed imaging datasets, split by case.

Two properties that matter and are easy to get wrong:

    * **Splitting is by case, not by patch.** Patches from one volume must not
      appear in both the training and the validation split. If they do, validation
      measures how well the model remembers a volume it trained on, and the Dice
      it reports is meaningless as an estimate of generalisation.
    * **Preprocessing happens once, on load.** Each case is preprocessed into a
      fixed-size tensor, so training iterates over prepared arrays rather than
      re-running the MONAI pipeline every epoch. That keeps an epoch on the CPU
      reference machine to a few seconds.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import torch

from .preprocess import PreprocessConfig, preprocess_case
from .synthetic import label_path_for, split_cases


@dataclass
class PreparedCase:
    """One preprocessed case, held as tensors."""

    image: torch.Tensor  # (1, D, H, W)
    label: torch.Tensor  # (1, D, H, W)
    source: str
    spacing: tuple[float, ...] | None = None
    shape: tuple[int, ...] | None = None

    @property
    def foreground_voxels(self) -> int:
        return int(self.label.sum().item())

    @property
    def foreground_fraction(self) -> float:
        total = int(self.label.numel())
        return float(self.foreground_voxels / total) if total else 0.0


@dataclass
class CaseDataset:
    """A split of prepared cases.

    Mirrors the shape of the federated layer's `LocalDataset`: the same
    `size`, `batches()` and summary interface, so the federated client can train
    on this without a second data abstraction existing.
    """

    cases: list[PreparedCase] = field(default_factory=list)
    name: str = "dataset"

    @property
    def size(self) -> int:
        """Number of cases. This is the FedAvg weight: one volume is one example."""
        return len(self.cases)

    @property
    def total_foreground_voxels(self) -> int:
        return sum(case.foreground_voxels for case in self.cases)

    @property
    def shape(self) -> tuple[int, ...] | None:
        return self.cases[0].shape if self.cases else None

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        case = self.cases[index]
        return case.image.clone(), case.label.clone()

    def batches(
        self,
        batch_size: int,
        generator: torch.Generator | None = None,
        shuffle: bool = False,
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """Yield (inputs, targets) batches of shape (B, 1, D, H, W).

        Order is fixed unless `shuffle=True` and a generator is supplied, so an
        unshuffled run is reproducible by construction.
        """
        if batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, got {batch_size}.")
        if not self.cases:
            return

        order = list(range(len(self.cases)))
        if shuffle and generator is not None:
            order = torch.randperm(len(self.cases), generator=generator).tolist()

        for start in range(0, len(order), batch_size):
            chunk = order[start : start + batch_size]
            images = torch.stack([self.cases[index].image for index in chunk])
            labels = torch.stack([self.cases[index].label for index in chunk])
            yield images, labels

    def describe(self) -> dict[str, Any]:
        """Facts safe to publish: counts and geometry, never voxel data."""
        return {
            "name": self.name,
            "cases": self.size,
            "shape": list(self.shape) if self.shape else None,
            "foregroundVoxels": self.total_foreground_voxels,
            "meanForegroundFraction": (
                float(
                    sum(case.foreground_fraction for case in self.cases) / self.size
                )
                if self.size
                else 0.0
            ),
        }


def load_cases(
    image_paths: list[Path],
    config: PreprocessConfig,
    label_lookup: Any = label_path_for,
) -> list[PreparedCase]:
    """Preprocess each image into a `PreparedCase`.

    A case with an empty label is kept: dropping it here would hide a data problem
    rather than surface it, and the metrics already exclude empty pairs explicitly
    and count them.
    """
    prepared: list[PreparedCase] = []
    for image_path in image_paths:
        label_path = label_lookup(image_path)
        processed = preprocess_case(image_path, label_path, config)
        prepared.append(
            PreparedCase(
                image=processed["image"],
                label=processed["label"],
                source=str(image_path),
                spacing=processed.get("spacing"),
                shape=processed.get("shape"),
            )
        )
    return prepared


def build_site_datasets(
    image_paths: list[Path],
    config: PreprocessConfig,
    validation_fraction: float,
    seed: int = 0,
    name: str = "site",
) -> dict[str, CaseDataset]:
    """Split cases into train/validation and preprocess each side.

    Returns ``{"train": CaseDataset, "validation": CaseDataset}``. The validation
    dataset may be empty when there are too few cases for a split, which the
    training loop handles by reporting no validation metrics rather than zeroes.
    """
    train_paths, validation_paths = split_cases(image_paths, validation_fraction, seed=seed)

    train = CaseDataset(load_cases(train_paths, config), name=f"{name}-train")
    validation = CaseDataset(
        load_cases(validation_paths, config), name=f"{name}-validation"
    )

    if train.size == 0:
        raise ValueError(
            f"No training cases for {name}: {len(image_paths)} case(s) were split with "
            f"validation_fraction={validation_fraction}."
        )
    return {"train": train, "validation": validation}


def dataset_summary(
    train: CaseDataset,
    validation: CaseDataset,
    config: PreprocessConfig,
) -> dict[str, Any]:
    """Provenance for a run: what data, how split, processed how."""
    return {
        "train": train.describe(),
        "validation": validation.describe(),
        "preprocess": config.as_dict(),
        "preprocessDescription": config.describe(),
        "dataSource": "synthetic phantoms, procedurally generated",
        "containsPatientData": False,
        "splitBy": "case",
        "note": (
            "Splitting is by whole case, so no patch from a training volume appears "
            "in validation."
        ),
    }
