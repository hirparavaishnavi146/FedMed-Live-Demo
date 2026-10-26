"""The U-Net training job behind the dashboard's Start Training button.

This is a real training job, not a simulation. It calls the same
``run_experiment`` the CLI uses, on the dataset that was assigned to the hospital
client, and reports the loss and Dice the loop actually measured.

Where the model runs, and why
-----------------------------
3D MRI training does not fit the free hosting tier. Measured on this project:

    API process with torch loaded         ~340 MB
    3D U-Net training, smallest config     384 MB peak
    Render free tier                       512 MB total

384 + 340 exceeds 512, so the two cannot share an instance. The job therefore
runs where there is enough memory — a local research machine or a dedicated
worker — and the deployed dashboard stays a read-only view. When torch is not
installed, ``start`` raises a 503 with the exact command to run locally, rather
than accepting the request and failing later or pretending it worked.

How a local run reaches the public dashboard
--------------------------------------------
``run_experiment`` writes ``experiment.json`` into the persistent output
directory, which is exactly the artifact ``GET /api/v1/model/experiment`` reads.
Point ``FEDMED_IMAGING_EXPERIMENT`` at the file a local run produced and the
deployed dashboard displays those real measured numbers. Nothing else needs to
be uploaded, and no dataset or checkpoint has to leave the machine that trained.

Checkpoints are written under the output directory's ``checkpoints/`` subfolder,
which is Git-ignored and is never served by the API.
"""

from __future__ import annotations

import importlib
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..errors import ApiError, ConflictError, ValidationProblem
from ..schemas.datasets import (
    ImagingEpoch,
    ImagingJobConfig,
    ImagingJobStatus,
    TrainingJobState,
)
from . import dataset_service

# Where a finished run's artifacts go. The same directory the CLI uses, so the
# dashboard's existing /model/experiment endpoint picks the result up.
DEFAULT_OUTPUT_DIR = Path("experiments/imaging")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _pipeline():
    """Import the imaging pipeline, which pulls in torch.

    Imported on demand so the API starts and serves its read endpoints in a
    deployment without the machine-learning stack installed.
    """
    return importlib.import_module("..imaging.pipeline", __package__)


def torch_available() -> tuple[bool, str | None]:
    """Whether real training is possible here, and why not when it is not."""
    try:
        importlib.import_module("torch")
    except ImportError:
        return False, (
            "PyTorch is not installed in this deployment, so 3D MRI training cannot "
            "run here. The dashboard stays readable. Train locally with the full "
            "requirements.txt and point FEDMED_IMAGING_EXPERIMENT at the produced "
            "experiment.json, or run uvicorn from a machine that has torch."
        )
    return True, None


class _StopRequested(Exception):
    """Raised inside the epoch callback to unwind a stopped run."""


class ImagingTrainingJob:
    """One training job at a time, on a background thread."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._status = ImagingJobStatus()
        self._stop_requested = False

    # -- inspection --------------------------------------------------------
    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._status.state is TrainingJobState.running

    def status(self) -> ImagingJobStatus:
        with self._lock:
            return self._status

    def reset(self) -> ImagingJobStatus:
        """Clear a finished job. Refuses while one is running."""
        with self._lock:
            if self._status.state is TrainingJobState.running:
                raise ConflictError(
                    "A training job is running. Stop it before clearing.",
                    code="training_already_running",
                )
            self._status = ImagingJobStatus()
            return self._status

    # -- control -----------------------------------------------------------
    def start(self, dataset_id: str, config: ImagingJobConfig | None = None) -> ImagingJobStatus:
        """Validate readiness, then start real training on a background thread."""
        available, reason = torch_available()
        if not available:
            # Refused up front with the cause, not accepted and failed later.
            raise ApiError(reason or "Training is unavailable.", code="training_unavailable", status_code=503)

        with self._lock:
            if self._status.state is TrainingJobState.running:
                raise ConflictError(
                    "A training job is already running. Stop it before starting another.",
                    code="training_already_running",
                    detail={"jobId": self._status.job_id},
                )

            record = dataset_service.registry.get(dataset_id)

            # Readiness is checked before the thread starts, so the caller gets a
            # specific error naming what is missing rather than a job that dies.
            if not record.validation.valid:
                raise ValidationProblem(
                    f"Dataset {record.name!r} did not pass validation. Fix the "
                    f"{record.validation.error_count} reported error(s) and validate "
                    "it again.",
                    code="dataset_not_ready",
                    detail={"datasetId": record.id, "errorCount": record.validation.error_count},
                )
            if record.assigned_hospital_id is None:
                raise ValidationProblem(
                    f"Dataset {record.name!r} is not assigned to a hospital client. "
                    "Assign it before starting training.",
                    code="dataset_not_assigned",
                    detail={"datasetId": record.id},
                )
            if not record.validation.cases:
                raise ValidationProblem(
                    f"Dataset {record.name!r} contains no cases.",
                    code="dataset_empty",
                    detail={"datasetId": record.id},
                )

            effective = config or ImagingJobConfig()
            job_id = f"job-{uuid.uuid4().hex[:12]}"
            self._stop_requested = False

            self._status = ImagingJobStatus(
                state=TrainingJobState.running,
                job_id=job_id,
                dataset_id=record.id,
                dataset_name=record.name,
                hospital_id=record.assigned_hospital_id,
                hospital_name=record.assigned_hospital_name,
                config=effective,
                epochs_total=effective.epochs,
                started_at=_utc_now(),
                measured=False,
                message=(
                    f"Training on {record.name!r} for {record.assigned_hospital_name}. "
                    f"{effective.epochs} epoch(s) planned."
                ),
            )

            self._thread = threading.Thread(
                target=self._run,
                args=(record, effective, job_id),
                name="fedmed-imaging-job",
                daemon=True,
            )
            self._thread.start()
            return self._status

    def request_stop(self) -> ImagingJobStatus:
        """Ask the job to stop. The current epoch finishes first."""
        with self._lock:
            if self._status.state is not TrainingJobState.running:
                raise ConflictError(
                    "No training job is running, so there is nothing to stop.",
                    code="training_not_running",
                )
            self._status.message = "Stop requested; finishing the current epoch."
            self._stop_requested = True
            return self._status

    # -- the work ----------------------------------------------------------
    def _run(self, record: Any, config: ImagingJobConfig, job_id: str) -> None:
        """Run the real experiment, reporting each epoch as it finishes."""
        try:
            pipeline = _pipeline()
            output_dir = DEFAULT_OUTPUT_DIR
            output_dir.mkdir(parents=True, exist_ok=True)

            experiment = pipeline.ExperimentConfig(
                site=record.assigned_hospital_id or "HOSP-01",
                epochs=config.epochs,
                learning_rate=config.learning_rate,
                seed=config.seed,
                validation_fraction=config.validation_fraction,
                features=tuple(config.features),
                target_shape=tuple(config.target_shape),
                data_dir=record.path,
                output_dir=output_dir,
            )

            def report(epoch_record: Any) -> None:
                with self._lock:
                    if self._status.job_id != job_id:
                        # A newer job replaced this one; stop writing to it.
                        return
                    if self._stop_requested:
                        # Unwinds the loop at an epoch boundary. The epoch that just
                        # finished is still recorded, so a stopped run reports the
                        # measurements it actually made.
                        raise _StopRequested

                    self._status.epochs_completed = epoch_record.epoch
                    self._status.current_epoch = epoch_record.epoch
                    self._status.train_loss = epoch_record.train_loss
                    # Track the best Dice seen so far. Overwriting it each epoch
                    # would report the last epoch's Dice as the best.
                    if epoch_record.validation_dice is not None:
                        if (
                            self._status.best_validation_dice is None
                            or epoch_record.validation_dice > self._status.best_validation_dice
                        ):
                            self._status.best_validation_dice = epoch_record.validation_dice
                            self._status.best_epoch = epoch_record.epoch
                    self._status.epochs = [
                        *self._status.epochs,
                        ImagingEpoch(
                            epoch=epoch_record.epoch,
                            train_loss=epoch_record.train_loss,
                            validation_loss=epoch_record.validation_loss,
                            validation_dice=epoch_record.validation_dice,
                            validation_iou=epoch_record.validation_iou,
                            duration_seconds=epoch_record.duration_seconds,
                        ),
                    ]

            report_data = pipeline.run_experiment(experiment, on_epoch=report)

            result = report_data.get("result", {})
            epochs = result.get("epochs") or []
            with self._lock:
                if self._status.job_id != job_id:
                    return
                self._status.state = TrainingJobState.completed
                self._status.finished_at = _utc_now()
                self._status.measured = True
                self._status.checkpoint_dir = str(output_dir / "checkpoints")
                # Keys come from TrainingResult.as_dict(): the summary fields are
                # camelCase, the per-epoch records are snake_case. Read from the
                # artifact's real shape rather than assuming one convention.
                self._status.first_train_loss = result.get("firstTrainLoss")
                self._status.train_loss = result.get("lastTrainLoss")
                if epochs:
                    self._status.current_epoch = epochs[-1].get("epoch")
                    self._status.epochs_completed = int(
                        result.get("epochCount") or len(epochs)
                    )
                self._status.best_validation_dice = result.get("bestDice")
                self._status.best_epoch = result.get("bestEpoch")
                self._status.duration_seconds = result.get("durationSeconds")
                self._status.message = (
                    f"Completed {len(epochs)} epoch(s) on {record.name!r}. "
                    "Checkpoints and experiment.json were written to "
                    f"{output_dir}."
                )
        except _StopRequested:
            with self._lock:
                if self._status.job_id != job_id:
                    return
                self._status.state = TrainingJobState.stopped
                self._status.finished_at = _utc_now()
                # Whatever was measured before the stop is real and is kept.
                self._status.measured = bool(self._status.epochs)
                self._status.message = (
                    f"Stopped after {self._status.epochs_completed} epoch(s) at your "
                    "request. Measurements from the completed epochs are shown; this "
                    "is not a finished run."
                )
        except Exception as exc:  # noqa: BLE001 - reported to the caller as state
            with self._lock:
                if self._status.job_id != job_id:
                    return
                self._status.state = TrainingJobState.failed
                self._status.finished_at = _utc_now()
                self._status.measured = False
                self._status.error = f"{type(exc).__name__}: {exc}"
                self._status.message = (
                    "Training failed. The error is reported as it happened; no "
                    "partial result is presented as a success."
                )


job = ImagingTrainingJob()
