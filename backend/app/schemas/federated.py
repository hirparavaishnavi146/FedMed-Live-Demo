"""Schemas for federated learning runs.

Two things these schemas are careful about:

1.  `measured` — a round produced by a real federated run is `measured: true`; the
    seeded illustrative baseline is `measured: false`. The dashboard can therefore
    distinguish "we ran this" from "this is a placeholder", and so can a reader of
    the raw JSON.

2.  `execution` — states how the run was carried out. `local-simulation` means one
    process on one machine. Nothing here is ever reported as cross-silo.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import Field

from .common import CamelModel


class ClientRoundOutcome(CamelModel):
    """What one client did in one round, or why it did not take part."""

    client: str
    participated: bool
    num_examples: int = 0
    # Share of the aggregate this client contributed. Sums to 1 across
    # participating clients, which makes the weighting visible.
    weight: float = 0.0
    train_loss: float | None = None
    validation_loss: float | None = None
    validation_dice: float | None = None
    duration_seconds: float = 0.0
    failure: str | None = None


class FederatedRound(CamelModel):
    """One completed federated round, with measured values."""

    round: int
    strategy: str
    measured: bool = True
    participants: int
    expected_participants: int
    participating_clients: list[str] = Field(default_factory=list)
    excluded_clients: list[ClientRoundOutcome] = Field(default_factory=list)
    clients: list[ClientRoundOutcome] = Field(default_factory=list)
    total_examples: int
    aggregated_loss: float
    aggregated_validation_loss: float | None = None
    aggregated_validation_dice: float | None = None
    duration_seconds: float
    started_at: datetime
    finished_at: datetime
    aggregated_parameters: int


class RunConfigPayload(CamelModel):
    """The configuration a federated run actually used.

    Deliberately separate from `TrainingConfig`: the API request shape and the
    federated run shape are not the same thing (a run carries client labels and
    the resolved strategy), and conflating them would let a request field quietly
    imply something the run did not do.
    """

    rounds: int
    local_epochs: int
    learning_rate: float
    batch_size: int
    min_clients: int
    validation_split: float
    seed: int
    patch_size: int
    strategy: str
    clients: int
    client_labels: list[str]
    fail_clients: list[str]


class FederatedRunStatus(CamelModel):
    """The state of the current federated experiment."""

    run_id: str | None
    status: str
    execution: str
    measured: bool
    is_simulated_data: bool
    trains_on_real_patient_data: bool
    strategy: str
    seed: int
    config: RunConfigPayload | None = None
    config_description: str
    clients: list[dict] = Field(default_factory=list)
    # None when torch is not installed in this deployment, which is reported
    # explicitly rather than as 0. See backend/requirements-web.txt.
    model_parameters: int | None
    total_rounds: int
    completed_rounds: int
    current_round: int | None
    started_at: datetime | None
    finished_at: datetime | None
    first_loss: float | None
    last_loss: float | None
    last_validation_dice: float | None
    loss_improvement: float | None
    error: str | None
    message: str
    history: list[FederatedRound] = Field(default_factory=list)


class FederatedHistory(CamelModel):
    """Round history plus an explicit statement of where it came from.

    `source` is `federated-run` when the rows were measured by a real run, and
    `seed` when they are the illustrative baseline.
    """

    source: str
    measured: bool
    rounds: list[FederatedRound]
    note: str = ""


class FederatedConfigSummary(CamelModel):
    """What the federated layer is and what it currently supports."""

    strategy: str
    strategies_supported: list[str]
    execution_modes: list[str]
    current_execution: str
    # None when torch is not installed in this deployment, reported explicitly
    # rather than as an empty string. See backend/requirements-web.txt.
    model: str | None
    # None when torch is not installed in this deployment, which is reported
    # explicitly rather than as 0. See backend/requirements-web.txt.
    model_parameters: int | None
    client_model_description: str
    client_profiles: list[dict]
    aggregator: str
    weighting: str
    clients_send: list[str]
    clients_never_send: list[str]
    limitations: list[str]
