"""Schemas for dataset management and the U-Net training job."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from .common import CamelModel


class IssueSeverity(str, Enum):
    error = "error"
    warning = "warning"


class DatasetSource(str, Enum):
    """Where a dataset came from. Kept explicit so the UI never has to guess."""

    directory = "server-directory"
    upload = "upload"


class DatasetCase(CamelModel):
    """One image/mask pair and what was found in it."""

    name: str
    image_path: str
    label_path: str | None = None
    image_shape: list[int] | None = None
    label_shape: list[int] | None = None
    spacing: list[float] | None = None
    voxels: int | None = None
    foreground_voxels: int | None = None
    foreground_fraction: float | None = None
    label_values: list[float] = []
    valid: bool
    issues: list[dict] = []


class ValidationLimits(CamelModel):
    """The limits a dataset was checked against, so the UI can state them."""

    max_file_bytes: int
    max_total_bytes: int
    max_cases: int
    min_voxels: int
    max_voxels: int
    accepted_suffixes: list[str]


class DatasetSummary(CamelModel):
    """A validated dataset, as listed in the dashboard."""

    id: str
    name: str
    source: DatasetSource
    path: str
    valid: bool
    case_count: int
    valid_case_count: int
    error_count: int
    warning_count: int
    total_bytes: int
    modality: str
    modality_source: str
    created_at: datetime
    issues: list[dict] = []
    cases: list[DatasetCase] = []
    limits: ValidationLimits

    # Assignment (requirement: select a hospital client and assign a dataset).
    assigned_hospital_id: str | None = None
    assigned_hospital_name: str | None = None
    assigned_at: datetime | None = None

    # True only when validated AND assigned, i.e. ready to train on.
    ready_to_train: bool = False


class DatasetList(CamelModel):
    datasets: list[DatasetSummary] = []
    total: int = 0


class AssignDatasetRequest(CamelModel):
    """Assign a validated dataset to a simulated hospital client."""

    hospital_id: str


class ValidateDirectoryRequest(CamelModel):
    """Validate a dataset that is already on this machine."""

    path: str
    name: str | None = None


class DatasetDeleteResult(CamelModel):
    id: str
    deleted: bool
    files_removed: int = 0
    detail: str


class TrainingJobState(str, Enum):
    idle = "idle"
    running = "running"
    completed = "completed"
    failed = "failed"
    stopped = "stopped"


class ImagingJobConfig(CamelModel):
    """Parameters this 3D U-Net actually supports.

    Every value here reaches the real training loop. There is deliberately no
    field for a parameter the model does not have (no batch size above 1, no
    mixed precision, no optimiser choice) rather than accepting one and ignoring it.
    """

    epochs: int = 4
    learning_rate: float = 1e-3
    seed: int = 42
    validation_fraction: float = 0.25
    target_shape: list[int] = [32, 32, 32]
    features: list[int] = [16, 32, 64]
    label_threshold: float = 0.5


class ImagingEpoch(CamelModel):
    epoch: int
    train_loss: float
    validation_loss: float | None = None
    validation_dice: float | None = None
    validation_iou: float | None = None
    duration_seconds: float


class ImagingJobStatus(CamelModel):
    """Live state of a U-Net training job."""

    state: TrainingJobState = TrainingJobState.idle
    job_id: str | None = None
    dataset_id: str | None = None
    dataset_name: str | None = None
    hospital_id: str | None = None
    hospital_name: str | None = None
    config: ImagingJobConfig | None = None

    epochs_total: int = 0
    epochs_completed: int = 0
    current_epoch: int | None = None

    train_loss: float | None = None
    first_train_loss: float | None = None
    best_validation_dice: float | None = None
    best_epoch: int | None = None

    started_at: datetime | None = None
    finished_at: datetime | None = None
    duration_seconds: float | None = None

    epochs: list[ImagingEpoch] = []
    checkpoint_dir: str | None = None
    error: str | None = None
    message: str = ""

    # Stated on every response so a consumer never has to infer it.
    measured: bool = False
    execution: str = "local-cpu"
    trains_on_real_patient_data: bool = False
    available: bool = True
    unavailable_reason: str | None = None


class StartTrainingRequest(CamelModel):
    dataset_id: str
    config: ImagingJobConfig | None = None
