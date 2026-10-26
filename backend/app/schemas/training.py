"""Schemas for training configuration, jobs and round history."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import Field, model_validator

from .common import CamelModel


class RoundStatus(str, Enum):
    complete = "complete"
    running = "running"
    queued = "queued"


class AggregationMethod(str, Enum):
    fedavg = "FedAvg"
    fedprox = "FedProx"
    fedadam = "FedAdam"


class TrainingRound(CamelModel):
    """One federated round as reported by the aggregator.

    In the current stage these rows are generated from a fixed curve, not
    produced by an aggregation of real client updates.
    """

    round: int
    status: RoundStatus
    global_loss: float
    global_dice: float
    global_accuracy: float
    participants: int
    expected_participants: int
    duration_seconds: int
    timestamp: datetime


class MetricPoint(CamelModel):
    """A single point on the training-history charts."""

    round: int
    loss: float
    validation_loss: float
    dice: float
    accuracy: float


class TrainingConfig(CamelModel):
    """Requested run configuration.

    Bounds match the frontend form validation exactly, so the browser and the
    API reject the same values.
    """

    rounds: int = Field(default=10, ge=1, le=50)
    local_epochs: int = Field(default=2, ge=1, le=20)
    learning_rate: float = Field(default=0.001, gt=0, le=0.1)
    batch_size: int = Field(default=2, ge=1, le=8)
    aggregation: AggregationMethod = AggregationMethod.fedavg
    min_clients: int = Field(default=2, ge=1, le=3)
    validation_split: float = Field(default=0.2, ge=0.05, le=0.5)
    random_seed: int = Field(default=42, ge=0, le=99999)
    # Patch edge length for the federated client model. Bounds match the
    # federated layer's own limits exactly, so the API and the run cannot
    # disagree. This is the small stand-in model's patch, not an MRI volume.
    patch_size: int = Field(default=16, ge=8, le=32)
    use_differential_privacy: bool = False
    use_secure_aggregation: bool = False
    use_homomorphic_encryption: bool = False

    @model_validator(mode="after")
    def _check_total_work(self) -> "TrainingConfig":
        """Refuse configurations that would ask for absurd local work.

        The reference machine is CPU only, so the product of rounds and local
        epochs is capped. The frontend applies the same 200-pass limit.
        """
        if self.rounds * self.local_epochs > 200:
            raise ValueError(
                "rounds multiplied by local_epochs must not exceed 200 "
                "(the reference machine has no GPU)"
            )
        return self


class TrainingJobState(str, Enum):
    idle = "idle"
    running = "running"
    completed = "completed"
    stopped = "stopped"


class TrainingStatus(CamelModel):
    """State of the simulated training job.

    ``is_simulated`` and ``trains_model`` are always present so a client can
    tell, from the response alone, that no model is being trained.
    """

    state: TrainingJobState
    is_simulated: bool
    trains_model: bool
    job_id: str | None
    config: TrainingConfig | None
    total_rounds: int
    completed_rounds: int
    current_round: int | None
    started_at: datetime | None
    finished_at: datetime | None
    seconds_per_round: float
    message: str
    history: list[TrainingRound] = Field(default_factory=list)


class TrainingStartRequest(CamelModel):
    """Body for POST /api/v1/training/start.

    An empty body is valid: the defaults above are used.
    """

    config: TrainingConfig = Field(default_factory=TrainingConfig)
