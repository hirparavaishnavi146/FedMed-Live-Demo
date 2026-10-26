"""Training endpoints, backed by the federated learning core.

WHAT ACTUALLY HAPPENS WHEN YOU CALL POST /api/v1/training/start
    A real federated run starts. Each configured hospital client trains the model
    on its own synthetically generated data, returns parameters and its example
    count, and the server aggregates them with sample-count-weighted FedAvg. The
    losses and Dice in the results are measured during that training.

WHAT IS STILL NOT HAPPENING
    The data is synthetic, the model is a small stand-in rather than the MRI
    3D U-Net, and the clients run inside this one process (`local-simulation`).
    There is no secure aggregation, homomorphic encryption or differential
    privacy: a request asking for any of those is refused rather than executed
    without them.

Every response states its own `execution` and `measured` fields, so a reader
never has to infer which of those two paragraphs applies.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from ...errors import ConflictError, NotFoundError
from ...schemas.common import Envelope, make_meta
from ...security import Principal, require_train
from ...schemas.federated import (
    FederatedConfigSummary,
    FederatedHistory,
    FederatedRound,
    FederatedRunStatus,
)
from ...schemas.training import MetricPoint, TrainingStartRequest
from ...services import federated_service

router = APIRouter(prefix="/training", tags=["training"])


@router.get(
    "/status",
    response_model=Envelope[FederatedRunStatus],
    summary="Federated run status",
    description=(
        "State of the current federated experiment, including how it is being "
        "executed (`execution`) and whether its numbers were measured. Returns "
        "idle with an explanation when no run has been started."
    ),
)
def training_status() -> Envelope[FederatedRunStatus]:
    return Envelope(data=federated_service.status(), meta=make_meta())


@router.get(
    "/history",
    response_model=Envelope[FederatedHistory],
    summary="Round history",
    description=(
        "Round-by-round results. `source` and `measured` state whether these rows "
        "were measured by a real federated run (`federated-run`, true) or are the "
        "illustrative seed baseline (`seed`, false)."
    ),
)
def training_history() -> Envelope[FederatedHistory]:
    source, measured, rounds, note = federated_service.history()
    return Envelope(
        data=FederatedHistory(
            source=source, measured=measured, rounds=rounds, note=note
        ),
        meta=make_meta(),
    )


@router.get(
    "/rounds/{round_number}",
    response_model=Envelope[FederatedRound],
    summary="One round in detail",
    description=(
        "A single round with its per-client breakdown: what each client "
        "contributed, how much weight it carried, and any client excluded from the "
        "aggregate with the reason."
    ),
    responses={404: {"description": "No such round in the current history"}},
)
def training_round(round_number: int) -> Envelope[FederatedRound]:
    _, _, rounds, _note = federated_service.history()
    for row in rounds:
        if int(row["round"]) == round_number:
            return Envelope(data=row, meta=make_meta())
    raise NotFoundError(
        f"No round {round_number} in the current history.",
        code="round_not_found",
        detail={"round": round_number, "available": [row["round"] for row in rounds]},
    )


@router.get(
    "/metrics",
    response_model=Envelope[list[MetricPoint]],
    summary="Chart series",
    description="Loss and Dice per round, built from the same results as /history.",
)
def training_metrics(
    limit: int | None = Query(default=None, ge=1, le=200),
) -> Envelope[list[MetricPoint]]:
    # Uses the federated service's own metric history, which returns nothing when
    # no round has been measured. Building points straight from the history rows
    # would have charted the seeded baseline as though it were a result.
    points = [MetricPoint(**point) for point in federated_service.metric_history(limit=limit)]
    return Envelope(data=points, meta=make_meta())


@router.get(
    "/config",
    response_model=Envelope[FederatedConfigSummary],
    summary="What the federated layer does",
    description=(
        "The strategy, execution modes, model, client profiles, exactly which "
        "fields a client sends, which it never sends, and the known limitations."
    ),
)
def training_config() -> Envelope[FederatedConfigSummary]:
    return Envelope(data=federated_service.config_summary(), meta=make_meta())


@router.post(
    "/start",
    response_model=Envelope[FederatedRunStatus],
    status_code=201,
    summary="Start a federated run",
    description=(
        "Starts a real federated run over synthetic data: each client trains "
        "locally and returns parameters, which are aggregated with "
        "sample-count-weighted FedAvg. Returns 409 if a run is already in "
        "progress, and 422 for an invalid configuration or a request for a "
        "strategy or privacy mechanism that is not implemented."
    ),
    responses={
        409: {"description": "A run is already in progress"},
        422: {"description": "Invalid or unsupported configuration"},
    },
)
def start_training(
    payload: TrainingStartRequest | None = None,
    principal: Principal = Depends(require_train),
) -> Envelope[FederatedRunStatus]:
    config = payload.config if payload is not None else TrainingStartRequest().config
    return Envelope(data=federated_service.start(config), meta=make_meta())


@router.post(
    "/stop",
    response_model=Envelope[FederatedRunStatus],
    summary="Stop the running experiment",
    description=(
        "Asks the run to stop after the round in progress. Rounds already "
        "completed are kept, because they really happened."
    ),
    responses={409: {"description": "No run is in progress"}},
)
def stop_training(
    principal: Principal = Depends(require_train),
) -> Envelope[FederatedRunStatus]:
    return Envelope(data=federated_service.stop(), meta=make_meta())


@router.post(
    "/reset",
    response_model=Envelope[FederatedRunStatus],
    summary="Clear the finished run",
    description="Discards a completed run, so a new one can start from a clean state.",
    responses={409: {"description": "A run is still in progress"}},
)
def reset_training(
    principal: Principal = Depends(require_train),
) -> Envelope[FederatedRunStatus]:
    try:
        federated_service.reset()
    except RuntimeError as exc:
        raise ConflictError(str(exc), code="training_in_progress") from exc
    return Envelope(data=federated_service.status(), meta=make_meta())
