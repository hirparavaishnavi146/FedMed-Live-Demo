"""Segmentation metrics, computed from actual predictions.

Every function here takes the model's output and the reference mask and returns a
number derived from those tensors. Nothing is estimated, interpolated or filled
in from a lookup, because a metric that is not computed from a prediction is not
a measurement of anything.

Conventions used throughout:

    * ``prediction`` is either raw **logits** (when ``from_logits=True``, the
      default) or probabilities. Logits are converted with a sigmoid.
    * Binarisation uses a threshold, default 0.5.
    * A sample whose predicted mask AND reference mask are both empty is
      **skipped** and the count is reported. Scoring it 1.0 would inflate Dice on
      a dataset containing many empty patches, which is the normal case in 3D
      segmentation where most patches are background.
    * Masks are compared as booleans per voxel; a voxel counts as positive when
      its value is >= 0.5. This keeps the rules the same for integer label maps
      and for float probability maps.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

DEFAULT_THRESHOLD = 0.5
_EPS = 1e-7


def to_probabilities(prediction: torch.Tensor, from_logits: bool = True) -> torch.Tensor:
    """Convert raw logits to probabilities in [0, 1]."""
    return torch.sigmoid(prediction) if from_logits else prediction


def binarise(
    tensor: torch.Tensor,
    threshold: float = DEFAULT_THRESHOLD,
    from_logits: bool = False,
) -> torch.Tensor:
    """Threshold to a boolean mask of the same shape.

    ``from_logits=True`` applies the sigmoid first, so callers cannot accidentally
    threshold logits at 0.5 (which would treat every logit above zero as
    foreground).
    """
    if from_logits:
        tensor = torch.sigmoid(tensor)
    return tensor >= threshold


def _flatten(tensor: torch.Tensor) -> torch.Tensor:
    """(N, ...) -> (N, voxels) so per-sample metrics can be formed."""
    return tensor.reshape(tensor.shape[0], -1)


@dataclass
class ConfusionCounts:
    """Voxel counts over a batch, plus how many samples were usable."""

    true_positive: float = 0.0
    false_positive: float = 0.0
    true_negative: float = 0.0
    false_negative: float = 0.0
    samples_scored: int = 0
    samples_skipped: int = 0
    per_sample_dice: list[float] = field(default_factory=list)

    @property
    def total(self) -> float:
        return (
            self.true_positive + self.false_positive + self.true_negative + self.false_negative
        )


def confusion_counts(
    prediction: torch.Tensor,
    target: torch.Tensor,
    threshold: float = DEFAULT_THRESHOLD,
    from_logits: bool = True,
) -> ConfusionCounts:
    """Count TP/FP/TN/FN voxels over a batch of masks."""
    predicted_mask = binarise(prediction, threshold, from_logits=from_logits)
    target_mask = binarise(target, threshold, from_logits=False)

    if predicted_mask.shape != target_mask.shape:
        raise ValueError(
            f"prediction shape {tuple(predicted_mask.shape)} does not match target "
            f"shape {tuple(target_mask.shape)}."
        )

    predicted_flat = _flatten(predicted_mask)
    target_flat = _flatten(target_mask)

    counts = ConfusionCounts()
    for index in range(predicted_flat.shape[0]):
        predicted_row = predicted_flat[index]
        target_row = target_flat[index]

        positive_prediction = float(predicted_row.sum())
        positive_target = float(target_row.sum())

        if positive_prediction == 0.0 and positive_target == 0.0:
            # Nothing to segment and nothing predicted. Excluded from the average,
            # and counted so the exclusion is visible rather than silent.
            counts.samples_skipped += 1
            continue

        intersection = float((predicted_row & target_row).sum())
        counts.true_positive += intersection
        counts.false_positive += positive_prediction - intersection
        counts.false_negative += positive_target - intersection
        counts.true_negative += float((~predicted_row & ~target_row).sum())
        counts.samples_scored += 1

        denominator = positive_prediction + positive_target
        if denominator > 0.0:
            counts.per_sample_dice.append(2.0 * intersection / denominator)

    return counts


def dice_score(
    prediction: torch.Tensor,
    target: torch.Tensor,
    threshold: float = DEFAULT_THRESHOLD,
    from_logits: bool = True,
) -> float:
    """Mean Dice over the batch, ignoring samples that are empty on both sides.

    Dice = 2|P n T| / (|P| + |T|), computed per sample and averaged, which is the
    convention for 3D medical segmentation. Returns the global Dice over the batch
    when no sample had a positive voxel, so an all-background batch still yields a
    defined value rather than an exception.
    """
    counts = confusion_counts(prediction, target, threshold, from_logits)
    if counts.per_sample_dice:
        return float(sum(counts.per_sample_dice) / len(counts.per_sample_dice))
    denominator = 2.0 * counts.true_positive + counts.false_positive + counts.false_negative
    return float((2.0 * counts.true_positive) / denominator) if denominator > 0.0 else 0.0


def iou_score(
    prediction: torch.Tensor,
    target: torch.Tensor,
    threshold: float = DEFAULT_THRESHOLD,
    from_logits: bool = True,
) -> float:
    """Intersection over Union (Jaccard) from the same voxel counts as Dice."""
    counts = confusion_counts(prediction, target, threshold, from_logits)
    denominator = counts.true_positive + counts.false_positive + counts.false_negative
    return float(counts.true_positive / denominator) if denominator > 0.0 else 0.0


def sensitivity(
    prediction: torch.Tensor,
    target: torch.Tensor,
    threshold: float = DEFAULT_THRESHOLD,
    from_logits: bool = True,
) -> float:
    """Recall: of the true positive voxels, the fraction found. 0.0 when none exist."""
    counts = confusion_counts(prediction, target, threshold, from_logits)
    denominator = counts.true_positive + counts.false_negative
    return float(counts.true_positive / denominator) if denominator > 0.0 else 0.0


def specificity(
    prediction: torch.Tensor,
    target: torch.Tensor,
    threshold: float = DEFAULT_THRESHOLD,
    from_logits: bool = True,
) -> float:
    """Of the true negative voxels, the fraction kept negative."""
    counts = confusion_counts(prediction, target, threshold, from_logits)
    denominator = counts.true_negative + counts.false_positive
    return float(counts.true_negative / denominator) if denominator > 0.0 else 0.0


def segmentation_report(
    prediction: torch.Tensor,
    target: torch.Tensor,
    loss: float | None = None,
    threshold: float = DEFAULT_THRESHOLD,
    from_logits: bool = True,
) -> dict[str, float | int | None]:
    """All metrics for one batch, from a single pass over the confusion counts.

    Returned keys are the ones the API and the UI report. `samplesSkipped` is
    included deliberately: a Dice averaged over 1 of 20 patches is a much weaker
    statement than one averaged over 20, and the caller should be able to see
    which it has.
    """
    counts = confusion_counts(prediction, target, threshold, from_logits)
    denominator = 2.0 * counts.true_positive + counts.false_positive + counts.false_negative
    dice = (
        float(sum(counts.per_sample_dice) / len(counts.per_sample_dice))
        if counts.per_sample_dice
        else (float((2.0 * counts.true_positive) / denominator) if denominator > 0.0 else 0.0)
    )
    union = counts.true_positive + counts.false_positive + counts.false_negative
    recall_denominator = counts.true_positive + counts.false_negative
    specificity_denominator = counts.true_negative + counts.false_positive

    return {
        "loss": None if loss is None else float(loss),
        "dice": dice,
        "iou": float(counts.true_positive / union) if union > 0.0 else 0.0,
        "sensitivity": (
            float(counts.true_positive / recall_denominator) if recall_denominator > 0.0 else 0.0
        ),
        "specificity": (
            float(counts.true_negative / specificity_denominator)
            if specificity_denominator > 0.0
            else 0.0
        ),
        "threshold": float(threshold),
        "samplesScored": counts.samples_scored,
        "samplesSkipped": counts.samples_skipped,
        "truePositiveVoxels": int(counts.true_positive),
        "falsePositiveVoxels": int(counts.false_positive),
        "falseNegativeVoxels": int(counts.false_negative),
        "trueNegativeVoxels": int(counts.true_negative),
    }
