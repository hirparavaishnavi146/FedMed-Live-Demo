"""Schema for a measured standalone imaging experiment.

This is deliberately separate from `models.ModelVersion`, which is the
*illustrative* version registry. Keeping them in different types is what stops a
measured Dice and a placeholder Dice from ending up in the same list, which is the
single easiest way for a dashboard to accidentally lie.

`available` is explicit rather than inferring absence from a null payload, so a
client cannot confuse "no experiment has been run" with "the endpoint broke".
"""

from __future__ import annotations

from datetime import datetime

from .common import CamelModel


class ImagingEpoch(CamelModel):
    """One epoch's measured values."""

    epoch: int
    train_loss: float
    validation_loss: float | None = None
    validation_dice: float | None = None
    validation_iou: float | None = None
    validation_sensitivity: float | None = None
    validation_specificity: float | None = None
    validation_samples: int = 0
    duration_seconds: float = 0.0


class ImagingHoldout(CamelModel):
    """The best checkpoint re-evaluated from disk."""

    dice: float | None = None
    loss: float | None = None
    iou: float | None = None
    sensitivity: float | None = None
    specificity: float | None = None
    cases: int = 0
    source: str = ""


class ImagingModelInfo(CamelModel):
    """The architecture that was trained."""

    architecture: str
    features: list[int]
    parameter_count: int
    output_is_logits: bool = True


class ImagingProvenance(CamelModel):
    """Where the numbers came from — required for them to mean anything."""

    data_source: str
    contains_patient_data: bool
    clinically_validated: bool
    is_demo_value: bool
    split_by: str = "case"
    strategy: str = ""
    note: str


class ImagingExperiment(CamelModel):
    """A measured experiment, or an explicit statement that none exists."""

    available: bool
    # Present only when available is true.
    source: str | None = None
    site: str | None = None
    generated_at: datetime | None = None
    config: dict = {}
    config_description: str | None = None
    epochs: list[ImagingEpoch] = []
    epoch_count: int = 0
    first_train_loss: float | None = None
    last_train_loss: float | None = None
    loss_improved: bool | None = None
    best_dice: float | None = None
    best_epoch: int | None = None
    holdout: ImagingHoldout | None = None
    model: ImagingModelInfo | None = None
    provenance: ImagingProvenance | None = None
    duration_seconds: float | None = None
    # Why it is unavailable, when it is.
    unavailable_reason: str | None = None

    def summary(self) -> dict:
        """The compact form, for a payload where the epoch list would be noise."""
        return {
            "available": self.available,
            "site": self.site,
            "bestDice": self.best_dice,
            "bestEpoch": self.best_epoch,
            "lastTrainLoss": self.last_train_loss,
            "lossImproved": self.loss_improved,
            "epochCount": self.epoch_count,
            "holdoutDice": None if self.holdout is None else self.holdout.dice,
            "parameterCount": None if self.model is None else self.model.parameter_count,
            "generatedAt": self.generated_at,
        }
