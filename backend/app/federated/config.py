"""Round configuration for federated runs, with seeds.

Every value that affects the outcome is recorded here, so a run can be described
and repeated. `as_dict()` is what gets stored alongside a run's results, which is
what makes a result reproducible rather than merely reported.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any

from .errors import RoundConfigurationError

# Bounds are deliberately narrow: the reference machine is CPU-only (no GPU), so
# the limits keep a round finishable rather than aspirational.
ROUNDS_RANGE = (1, 50)
LOCAL_EPOCHS_RANGE = (1, 20)
LEARNING_RATE_RANGE = (1e-5, 0.1)
BATCH_SIZE_RANGE = (1, 8)
CLIENTS_RANGE = (1, 3)
VALIDATION_SPLIT_RANGE = (0.05, 0.5)
PATCH_SIZE_RANGE = (8, 32)


@dataclass(frozen=True)
class RoundConfig:
    """Configuration for one federated experiment.

    `seed` seeds every client's data generation and local training, so the same
    configuration produces the same run on the same machine.
    """

    rounds: int = 5
    local_epochs: int = 2
    learning_rate: float = 0.01
    batch_size: int = 4
    min_clients: int = 2
    validation_split: float = 0.2
    seed: int = 42
    patch_size: int = 16
    strategy: str = "FedAvg"
    # How many of the configured clients take part. Kept explicit so a run's
    # description states the participation level rather than implying all did.
    clients: int = 3
    # Client names, so results can be attributed without a lookup table.
    client_labels: tuple[str, ...] = field(
        default_factory=lambda: ("HOSP-01", "HOSP-02", "HOSP-03")
    )
    # Simulated client failures, used to exercise the exclusion path. Empty in a
    # normal run; the API never sets this outside tests.
    fail_clients: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        checks: list[tuple[str, Any, tuple[float, float]]] = [
            ("rounds", self.rounds, ROUNDS_RANGE),
            ("local_epochs", self.local_epochs, LOCAL_EPOCHS_RANGE),
            ("learning_rate", self.learning_rate, LEARNING_RATE_RANGE),
            ("batch_size", self.batch_size, BATCH_SIZE_RANGE),
            ("min_clients", self.min_clients, CLIENTS_RANGE),
            ("validation_split", self.validation_split, VALIDATION_SPLIT_RANGE),
            ("patch_size", self.patch_size, PATCH_SIZE_RANGE),
            ("clients", self.clients, CLIENTS_RANGE),
        ]
        for name, value, (low, high) in checks:
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise RoundConfigurationError(
                    f"{name} must be a number, received {type(value).__name__}.",
                    detail={"field": name},
                )
            if not low <= value <= high:
                raise RoundConfigurationError(
                    f"{name} must be between {low} and {high}, received {value}.",
                    detail={"field": name, "value": value, "min": low, "max": high},
                )

        if self.seed < 0:
            raise RoundConfigurationError(
                "seed must be zero or greater.", detail={"seed": self.seed}
            )
        if self.strategy != "FedAvg":
            raise RoundConfigurationError(
                f"Only FedAvg is implemented; received '{self.strategy}'.",
                detail={"strategy": self.strategy, "supported": ["FedAvg"]},
            )
        if self.clients > len(self.client_labels):
            raise RoundConfigurationError(
                f"{self.clients} clients requested but only "
                f"{len(self.client_labels)} are configured.",
                detail={"clients": self.clients, "configured": len(self.client_labels)},
            )
        if self.min_clients > self.clients:
            raise RoundConfigurationError(
                "min_clients cannot exceed the number of clients in the run.",
                detail={"min_clients": self.min_clients, "clients": self.clients},
            )
        # A round must be able to gather enough updates to aggregate.
        if self.clients - len(self.fail_clients) < self.min_clients:
            raise RoundConfigurationError(
                "Not enough clients would survive the configured failures to meet "
                "min_clients.",
                detail={
                    "clients": self.clients,
                    "failing": list(self.fail_clients),
                    "min_clients": self.min_clients,
                },
            )

    @property
    def active_labels(self) -> tuple[str, ...]:
        """The labels of the clients that take part in this run."""
        return tuple(self.client_labels[: self.clients])

    def as_dict(self) -> dict[str, Any]:
        """Plain, JSON-serialisable description of this configuration."""
        data = dataclasses.asdict(self)
        data["client_labels"] = list(self.client_labels)
        data["fail_clients"] = list(self.fail_clients)
        return data

    def describe(self) -> str:
        """One-line summary used in logs and the run's stored metadata."""
        return (
            f"{self.strategy}, {self.rounds} rounds, {self.clients} clients, "
            f"{self.local_epochs} local epochs, lr={self.learning_rate}, "
            f"batch={self.batch_size}, patch={self.patch_size}, seed={self.seed}"
        )
