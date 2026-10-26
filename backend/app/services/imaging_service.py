"""Reads the measured standalone imaging experiment from its artifact.

The experiment is produced by `app.imaging.pipeline`, which writes
`experiment.json` into its output directory. This service reads that file and
converts it into the API schema. It never runs training — a GET request must not
start a multi-second job — and it never fabricates a result.

If no artifact exists, the response says so explicitly (`available: false` plus a
reason) rather than returning zeroes. Zeroes would render as a measured Dice of
0.0, which is a different and false statement.

Location is configurable via ``FEDMED_IMAGING_EXPERIMENT`` so a deployment can
point at wherever the experiment was run.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..schemas.imaging import (
    ImagingEpoch,
    ImagingExperiment,
    ImagingHoldout,
    ImagingModelInfo,
    ImagingProvenance,
)

#: Default artifact path, relative to the backend package's parent (the repo root).
DEFAULT_EXPERIMENT_PATH = Path("experiments") / "imaging" / "experiment.json"


def _candidate_paths() -> list[Path]:
    """Where to look for the artifact, most specific first."""
    paths: list[Path] = []
    override = os.environ.get("FEDMED_IMAGING_EXPERIMENT")
    if override:
        paths.append(Path(override))
    # backend/app/services/ -> backend/ -> repo root
    backend_dir = Path(__file__).resolve().parents[2]
    paths.append(backend_dir / DEFAULT_EXPERIMENT_PATH)
    paths.append(Path.cwd() / DEFAULT_EXPERIMENT_PATH)
    paths.append(backend_dir.parent / DEFAULT_EXPERIMENT_PATH)
    return paths


def find_experiment_file() -> Path | None:
    """The first existing artifact path, or None."""
    for path in _candidate_paths():
        if path.exists() and path.is_file():
            return path
    return None


def _unavailable(reason: str) -> ImagingExperiment:
    return ImagingExperiment(available=False, unavailable_reason=reason)


def _as_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _timestamp(path: Path) -> datetime:
    """When the artifact was written, in UTC.

    Read from the file's mtime rather than trusted from the document, so a stale
    copy of an experiment cannot claim to be newer than it is.
    """
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)


def load_experiment(path: Path | None = None) -> ImagingExperiment:
    """Load the measured experiment, or explain why it is unavailable.

    Never raises for a missing or unreadable artifact: the API reports
    availability in the payload instead of failing the request, so the frontend can
    render "no measured experiment yet" rather than an error page.
    """
    resolved = Path(path) if path is not None else find_experiment_file()
    if resolved is None:
        return _unavailable(
            "No imaging experiment artifact found. Run "
            "`python -m app.imaging.pipeline` to produce one; until then every "
            "value shown is illustrative."
        )
    if not resolved.exists():
        return _unavailable(f"No experiment artifact at {resolved}.")

    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        return _unavailable(f"Experiment artifact at {resolved} could not be read: {exc}.")

    if not isinstance(payload, dict):
        return _unavailable(f"Experiment artifact at {resolved} is not a JSON object.")

    result = payload.get("result")
    if not isinstance(result, dict):
        return _unavailable(
            f"Experiment artifact at {resolved} has no 'result' object, so it does not "
            "look like output from app.imaging.pipeline."
        )

    experiment_config = payload.get("experiment") or {}
    epochs = [
        ImagingEpoch(
            epoch=int(record.get("epoch", index + 1)),
            train_loss=_as_float(record.get("train_loss")) or 0.0,
            validation_loss=_as_float(record.get("validation_loss")),
            validation_dice=_as_float(record.get("validation_dice")),
            validation_iou=_as_float(record.get("validation_iou")),
            validation_sensitivity=_as_float(record.get("validation_sensitivity")),
            validation_specificity=_as_float(record.get("validation_specificity")),
            validation_samples=int(record.get("validation_samples") or 0),
            duration_seconds=_as_float(record.get("duration_seconds")) or 0.0,
        )
        for index, record in enumerate(result.get("epochs") or [])
        if isinstance(record, dict)
    ]

    holdout_payload = payload.get("bestCheckpointEvaluation")
    holdout = None
    if isinstance(holdout_payload, dict):
        holdout = ImagingHoldout(
            dice=_as_float(holdout_payload.get("dice")),
            loss=_as_float(holdout_payload.get("loss")),
            iou=_as_float(holdout_payload.get("iou")),
            sensitivity=_as_float(holdout_payload.get("sensitivity")),
            specificity=_as_float(holdout_payload.get("specificity")),
            cases=int(holdout_payload.get("cases") or 0),
            source=str(holdout_payload.get("source") or ""),
        )

    model_payload = payload.get("model") or {}
    model = None
    if model_payload:
        model = ImagingModelInfo(
            architecture=str(model_payload.get("architecture", "UNet3D")),
            features=[int(width) for width in model_payload.get("features") or []],
            parameter_count=int(model_payload.get("parameterCount") or 0),
            output_is_logits=bool(model_payload.get("outputIsLogits", True)),
        )

    honesty = payload.get("honesty") or {}
    dataset = result.get("dataset") or {}
    provenance = ImagingProvenance(
        data_source=str(honesty.get("dataSource") or dataset.get("dataSource") or "unknown"),
        contains_patient_data=bool(honesty.get("containsPatientData", False)),
        clinically_validated=bool(honesty.get("clinicallyValidated", False)),
        # Defaults to True: if the artifact does not say, the safe reading is that
        # these are not measurements.
        is_demo_value=bool(honesty.get("isDemoValue", True)),
        split_by=str(dataset.get("splitBy") or "case"),
        strategy=str(result.get("configDescription") or ""),
        note=str(
            honesty.get("note")
            or "Measured by a local training run on synthetic data. Not a clinical result."
        ),
    )

    return ImagingExperiment(
        available=True,
        source=str(resolved),
        site=experiment_config.get("site"),
        generated_at=_timestamp(resolved),
        config={
            "epochs": experiment_config.get("epochs"),
            "batchSize": experiment_config.get("batchSize"),
            "learningRate": experiment_config.get("learningRate"),
            "seed": experiment_config.get("seed"),
            "validationFraction": experiment_config.get("validationFraction"),
            "targetShape": experiment_config.get("targetShape"),
        },
        config_description=str(result.get("configDescription") or ""),
        epochs=epochs,
        epoch_count=len(epochs),
        first_train_loss=_as_float(result.get("firstTrainLoss")),
        last_train_loss=_as_float(result.get("lastTrainLoss")),
        loss_improved=(
            None if result.get("lossImproved") is None else bool(result.get("lossImproved"))
        ),
        best_dice=_as_float(result.get("bestDice")),
        best_epoch=(
            None if result.get("bestEpoch") is None else int(result.get("bestEpoch"))
        ),
        holdout=holdout,
        model=model,
        provenance=provenance,
        duration_seconds=_as_float(result.get("durationSeconds")),
    )


def is_available() -> bool:
    """Whether a measured experiment exists. Cheap enough for a health check."""
    return find_experiment_file() is not None
