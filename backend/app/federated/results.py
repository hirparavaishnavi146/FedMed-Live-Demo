"""Round results and run history.

These types are what the API and the dashboard display. They carry *measured*
values: every loss and Dice number in here was produced by an actual local
training run on actual (synthetic) examples, and every duration is a real
measurement.

They also carry the facts that stop a result being overstated:

    * `execution` says how the run was carried out. A local simulation is labelled
      as one, so a reader cannot mistake it for a cross-silo deployment.
    * `excluded_clients` records clients that failed and were left out, together
      with the reason. A round that ran on two of three clients says so instead of
      reporting as if all three took part.
    * `measured` distinguishes this run's own numbers from any placeholder values
      elsewhere in the system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from .config import RoundConfig


class RunStatus(str, Enum):
    idle = "idle"
    running = "running"
    completed = "completed"
    stopped = "stopped"
    failed = "failed"


class ExecutionMode(str, Enum):
    """How a federated run was actually carried out.

    This is the field that keeps the project honest about what has and has not
    been demonstrated.
    """

    # All clients and the server inside one Python process on one machine.
    local_simulation = "local-simulation"
    # Separate Flower client processes talking to a server process over gRPC.
    distributed_local = "distributed-local"
    # Separate machines, separate hospital networks. NOT implemented or tested;
    # present so the enum states what it would be called.
    cross_silo = "cross-silo"


@dataclass(frozen=True)
class ClientOutcome:
    """What one client did in one round, or why it did not take part."""

    client: str
    participated: bool
    num_examples: int = 0
    weight: float = 0.0
    train_loss: float | None = None
    validation_loss: float | None = None
    validation_dice: float | None = None
    duration_seconds: float = 0.0
    failure: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "client": self.client,
            "participated": self.participated,
            "numExamples": self.num_examples,
            # How much this client actually influenced the global model.
            "weight": round(self.weight, 6),
            "trainLoss": None if self.train_loss is None else round(self.train_loss, 6),
            "validationLoss": None if self.validation_loss is None else round(self.validation_loss, 6),
            "validationDice": None if self.validation_dice is None else round(self.validation_dice, 6),
            "durationSeconds": round(self.duration_seconds, 4),
            "failure": self.failure,
        }


@dataclass
class RoundResult:
    """The outcome of one federated round."""

    round: int
    strategy: str
    outcomes: list[ClientOutcome]
    aggregated_loss: float
    aggregated_validation_loss: float | None
    aggregated_validation_dice: float | None
    total_examples: int
    duration_seconds: float
    started_at: datetime
    finished_at: datetime
    aggregated_parameters: int
    # True when the round used real local training. Set explicitly so a caller
    # never has to infer it.
    measured: bool = True

    @property
    def participating(self) -> list[str]:
        return [outcome.client for outcome in self.outcomes if outcome.participated]

    @property
    def excluded(self) -> list[ClientOutcome]:
        return [outcome for outcome in self.outcomes if not outcome.participated]

    @property
    def expected_clients(self) -> int:
        return len(self.outcomes)

    def as_dict(self) -> dict[str, Any]:
        return {
            "round": self.round,
            "strategy": self.strategy,
            "measured": self.measured,
            "participants": len(self.participating),
            "expectedParticipants": self.expected_clients,
            "participatingClients": self.participating,
            "excludedClients": [outcome.as_dict() for outcome in self.excluded],
            "clients": [outcome.as_dict() for outcome in self.outcomes],
            "totalExamples": self.total_examples,
            "aggregatedLoss": round(self.aggregated_loss, 6),
            "aggregatedValidationLoss": (
                None
                if self.aggregated_validation_loss is None
                else round(self.aggregated_validation_loss, 6)
            ),
            "aggregatedValidationDice": (
                None
                if self.aggregated_validation_dice is None
                else round(self.aggregated_validation_dice, 6)
            ),
            "durationSeconds": round(self.duration_seconds, 3),
            "startedAt": self.started_at.isoformat(),
            "finishedAt": self.finished_at.isoformat(),
            "aggregatedParameters": self.aggregated_parameters,
        }


@dataclass
class FederatedRun:
    """One federated experiment: its configuration, rounds and outcome."""

    run_id: str
    config: RoundConfig
    status: RunStatus = RunStatus.idle
    execution: ExecutionMode = ExecutionMode.local_simulation
    rounds: list[RoundResult] = field(default_factory=list)
    clients: list[dict[str, Any]] = field(default_factory=list)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None
    current_round: int | None = None
    # Parameter count of the model being aggregated, so the API can state the
    # size of what is actually exchanged.
    model_parameters: int = 0
    seed: int = 0
    message: str = ""

    @property
    def completed_rounds(self) -> int:
        return len(self.rounds)

    @property
    def first_loss(self) -> float | None:
        return self.rounds[0].aggregated_loss if self.rounds else None

    @property
    def last_loss(self) -> float | None:
        return self.rounds[-1].aggregated_loss if self.rounds else None

    @property
    def last_dice(self) -> float | None:
        return self.rounds[-1].aggregated_validation_dice if self.rounds else None

    @property
    def loss_improvement(self) -> float | None:
        """How much the aggregated loss fell from the first round to the last."""
        if self.first_loss is None or self.last_loss is None:
            return None
        return self.first_loss - self.last_loss

    def as_dict(self) -> dict[str, Any]:
        return {
            "runId": self.run_id,
            "status": self.status.value,
            "execution": self.execution.value,
            "measured": True,
            "isSimulatedData": True,
            "trainsOnRealPatientData": False,
            "strategy": self.config.strategy,
            "seed": self.seed,
            "config": self.config.as_dict(),
            "configDescription": self.config.describe(),
            "clients": self.clients,
            "modelParameters": self.model_parameters,
            "totalRounds": self.config.rounds,
            "completedRounds": self.completed_rounds,
            "currentRound": self.current_round,
            "startedAt": self.started_at.isoformat() if self.started_at else None,
            "finishedAt": self.finished_at.isoformat() if self.finished_at else None,
            "firstLoss": None if self.first_loss is None else round(self.first_loss, 6),
            "lastLoss": None if self.last_loss is None else round(self.last_loss, 6),
            "lastValidationDice": None if self.last_dice is None else round(self.last_dice, 6),
            "lossImprovement": (
                None if self.loss_improvement is None else round(self.loss_improvement, 6)
            ),
            "error": self.error,
            "message": self.message,
            "history": [result.as_dict() for result in self.rounds],
        }


def utc_now() -> datetime:
    return datetime.now(timezone.utc)
