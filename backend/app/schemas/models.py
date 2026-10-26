"""Schemas for aggregated model versions and their metrics."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from .common import CamelModel
from .imaging import ImagingExperiment


class ModelStatus(str, Enum):
    candidate = "candidate"
    active = "active"
    archived = "archived"


class ModelVersion(CamelModel):
    """A model version in the registry.

    Every metric here is an ILLUSTRATIVE value from `synthetic_data`. No
    segmentation model produced these numbers, and none of them measures anything
    real. The measured results of an actual training run live in
    `ModelResults.measured_experiment`, deliberately in a different field and a
    different type so the two can never be read as one list.
    """

    version: str
    created_at: datetime
    status: ModelStatus
    dice: float
    iou: float
    accuracy: float
    loss: float
    sensitivity: float
    specificity: float
    rounds: int
    notes: str


class ModelResults(CamelModel):
    """The model-results payload plus the derived active/best versions."""

    models: list[ModelVersion]
    active_version: ModelVersion | None
    best_version: ModelVersion | None
    latest_version: ModelVersion | None
    # True when `models` holds illustrative values rather than measurements. Always
    # true today: the registry is seeded sample data.
    registry_is_illustrative: bool = True
    registry_notice: str = (
        "Every value in `models` is illustrative sample data. No segmentation model "
        "produced these numbers."
    )
    # The measured results of a real training run, when one has been performed.
    measured_experiment: ImagingExperiment


class ModelComparisonRow(CamelModel):
    """One row of the version-comparison table."""

    version: str
    status: ModelStatus
    dice: float
    iou: float
    accuracy: float
    loss: float
    rounds: int
    is_best: bool


class ModelTargets(CamelModel):
    """Design targets used to draw progress bars. Not achievements."""

    dice: float = 0.85
    iou: float = 0.75
    accuracy: float = 0.95
