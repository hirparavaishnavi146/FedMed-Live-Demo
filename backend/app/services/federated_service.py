"""Service layer for federated runs, used by the API.

Holds the single in-memory run, converts the API's training configuration into a
federated `RoundConfig`, and reports status and history.

Two deliberate refusals live here:

    * `FedProx` and `FedAdam` are rejected, because only FedAvg is implemented.
      Accepting the setting and quietly running FedAvg would misreport the method.
    * The privacy flags are rejected while they are unimplemented. The alternative
      — accepting `useSecureAggregation: true` and running plain FedAvg — is
      exactly the "hide the updates and relabel the dashboard" behaviour this
      project must not do. A run either applies a mechanism or says it did not.
"""

from __future__ import annotations

import importlib
from typing import Any

from ..errors import ApiError, ConflictError, ValidationProblem
from ..federated.config import RoundConfig
from ..federated.errors import FederatedError, RoundConfigurationError
from ..federated.results import ExecutionMode, RunStatus
from ..federated.simulation import local_runner
from ..schemas.training import TrainingConfig


def _torch_module(name: str):
    """Import a torch-backed federated module (``data`` or ``model``) on demand.

    Both import torch. torch is deliberately not imported when the API starts, so
    that the dashboard boots and serves in a deployment whose training component is
    absent or offline. The modules imported here are the real ones — never a second,
    torch-free copy of the federated logic.
    """
    return importlib.import_module(f"..federated.{name}", __package__)


def _client_profiles():
    """The client profiles, or an empty tuple when torch is not installed."""
    try:
        return _torch_module("data").DEFAULT_PROFILES
    except ImportError:
        return ()


def _model_facts() -> dict[str, Any]:
    """The model's class name and parameter count.

    Returns ``None`` for each value when torch is not installed. An explicit null is
    the honest answer: reporting ``0`` would read as a model with no parameters,
    which is a different and false statement.
    """
    try:
        model_module = _torch_module("model")
        model = model_module.build_model()
    except ImportError:
        return {"name": None, "parameters": None}
    return {"name": type(model).__name__, "parameters": model_module.parameter_count(model)}

# Stated once, used in several responses.
CLIENT_MODEL_DESCRIPTION = (
    "A small 3D convolutional network (~550 parameters) that segments a synthetic "
    "sphere in a 16x16x16 patch. It is a real trainable model used to exercise "
    "federated aggregation. It is NOT the MRI segmentation model: that is the 3D "
    "U-Net in app/imaging (PyTorch + MONAI), which is trained separately and is not "
    "aggregated by this federated layer."
)

AGGREGATOR_DESCRIPTION = (
    "Sample-count-weighted Federated Averaging (FedAvg): each client's parameters "
    "are weighted by the number of local training examples behind them. Updates "
    "that disagree on structure or shape, or that contain non-finite values, are "
    "refused rather than averaged."
)

LIMITATIONS: list[str] = [
    "The data is synthetic, generated procedurally. There is no patient data "
    "anywhere and no real scan is read.",
    "The client model is a small stand-in, not the MRI 3D U-Net. Accuracy numbers "
    "from these runs say nothing about segmentation performance on real images.",
    "The default execution runs every client inside one Python process on one "
    "machine. That exercises the aggregation and the client boundary; it does not "
    "demonstrate a deployment across separate hospital networks.",
    "Client isolation here is structural (the data is never returned by the client "
    "API). In a real deployment it is enforced by network policy and by the client "
    "running inside the hospital's own infrastructure, which is not tested here.",
    "No secure aggregation, no homomorphic encryption and no differential privacy. "
    "None of the three is implemented, so none is claimed or simulated.",
    "Runs live in memory: restarting the API discards the run and its history.",
    "There is no authentication, so this is not a multi-tenant service.",
]


def _privacy_flags_refused(config: TrainingConfig) -> list[str]:
    """Names of the unimplemented privacy mechanisms a request asked for."""
    requested: list[str] = []
    if config.use_differential_privacy:
        requested.append("differential privacy")
    if config.use_secure_aggregation:
        requested.append("secure aggregation")
    if config.use_homomorphic_encryption:
        requested.append("homomorphic encryption")
    return requested


def to_round_config(config: TrainingConfig) -> RoundConfig:
    """Convert an API training configuration into a federated round configuration.

    Raises rather than silently adjusting anything.
    """
    if config.aggregation.value != "FedAvg":
        raise ValidationProblem(
            f"Only FedAvg is implemented; '{config.aggregation.value}' was requested.",
            code="unsupported_strategy",
            detail={"requested": config.aggregation.value, "supported": ["FedAvg"]},
        )

    requested_privacy = _privacy_flags_refused(config)
    if requested_privacy:
        raise ValidationProblem(
            "This run requested "
            + ", ".join(requested_privacy)
            + ". None of those mechanisms is implemented, so the run is refused "
            "rather than executed without them under a label that would imply "
            "otherwise.",
            code="privacy_mechanism_not_implemented",
            detail={"requested": requested_privacy, "implemented": []},
        )

    try:
        return RoundConfig(
            rounds=config.rounds,
            local_epochs=config.local_epochs,
            learning_rate=config.learning_rate,
            batch_size=config.batch_size,
            min_clients=config.min_clients,
            validation_split=config.validation_split,
            seed=config.random_seed,
            patch_size=config.patch_size,
            strategy=config.aggregation.value,
            clients=min(3, len(_client_profiles())),
        )
    except RoundConfigurationError as exc:
        raise ValidationProblem(
            exc.message, code=exc.code, detail=exc.detail
        ) from exc


def _require_training_dependencies() -> None:
    """Refuse training honestly when the training stack is not installed.

    Called before the configuration is validated. A missing dependency is not a
    fault in the caller's configuration, and reporting it as a configuration error
    (422) would blame them for something they did not get wrong. It would also
    surface as a message about client counts, which points at the wrong cause.
    """
    try:
        _torch_module("model")
    except ImportError as exc:
        # torch is absent in this deployment (requirements-web.txt). The training
        # code is the real one and is simply not installed, so say exactly that.
        raise ApiError(
            "Federated training is not available in this deployment: the training "
            "dependency (torch) is not installed. The dashboard remains readable. "
            "Run training locally, or deploy the training worker with the full "
            "requirements.txt.",
            code="training_unavailable",
            status_code=503,
        ) from exc


def start(config: TrainingConfig) -> dict[str, Any]:
    """Start a federated run. Returns its status payload."""
    _require_training_dependencies()
    round_config = to_round_config(config)

    if local_runner.is_running:
        raise ConflictError(
            "A federated run is already in progress. Stop it before starting another.",
            code="training_already_running",
            detail={"runId": local_runner.run.run_id if local_runner.run else None},
        )

    try:
        local_runner.start(round_config)
    except ImportError as exc:
        # torch is not installed in this deployment. The training code is the real
        # one and is simply absent, so say that plainly: a 500 would read as a bug,
        # and pretending the run started would be worse. See docs/DEPLOYMENT.md.
        raise ApiError(
            "Federated training is not available in this deployment: the training "
            "dependency (torch) is not installed. The dashboard remains readable. "
            "Run training locally, or deploy the training worker with the full "
            "requirements.txt.",
            code="training_unavailable",
            status_code=503,
        ) from exc
    except RuntimeError as exc:
        raise ConflictError(str(exc), code="training_already_running") from exc
    except FederatedError as exc:
        raise ValidationProblem(exc.message, code=exc.code, detail=exc.detail) from exc

    return status()


def stop() -> dict[str, Any]:
    """Ask the running experiment to stop after its current round."""
    if not local_runner.is_running:
        raise ConflictError(
            "No federated run is in progress, so there is nothing to stop.",
            code="training_not_running",
        )
    local_runner.request_stop()
    return status()


def status() -> dict[str, Any]:
    """Current run status. Present and honest when no run has been started."""
    run = local_runner.run

    if run is None:
        return {
            "runId": None,
            "status": RunStatus.idle.value,
            "execution": ExecutionMode.local_simulation.value,
            "measured": True,
            "isSimulatedData": True,
            "trainsOnRealPatientData": False,
            "strategy": "FedAvg",
            "seed": 0,
            "config": None,
            "configDescription": "No run has been started in this process.",
            "clients": [],
            "modelParameters": _model_facts()["parameters"],
            "totalRounds": 0,
            "completedRounds": 0,
            "currentRound": None,
            "startedAt": None,
            "finishedAt": None,
            "firstLoss": None,
            "lastLoss": None,
            "lastValidationDice": None,
            "lossImprovement": None,
            "error": None,
            "message": (
                "No federated run has been started. The round history below is the "
                "illustrative seed data and is marked measured: false."
            ),
            "history": [],
        }

    payload = run.as_dict()
    # The API's config shape is its own; expose the run's config as-is instead of
    # forcing it through TrainingConfig, which cannot represent clients/labels.
    return payload


def history() -> tuple[str, bool, list[dict[str, Any]], str]:
    """Round history with its provenance.

    Returns (source, measured, rounds, note). Falls back to the seeded baseline
    and says so, rather than presenting placeholder values as a run's output.
    """
    run = local_runner.run
    if run is not None and run.rounds:
        return (
            "federated-run",
            True,
            [result.as_dict() for result in run.rounds],
            (
                f"Measured by run {run.run_id} ({run.execution.value}) over synthetic "
                "data. Each round's loss and Dice were computed during local training "
                "on the clients' own held-back splits."
            ),
        )

    if run is not None:
        return (
            "federated-run",
            True,
            [],
            (
                f"Run {run.run_id} is {run.status.value} with no completed rounds yet. "
                "No rounds are shown, because none has been measured."
            ),
        )

    from . import synthetic_data

    seed_rounds = synthetic_data.build_seed_rounds()
    return (
        "seed",
        False,
        [_seed_round_to_dict(row, total=len(seed_rounds)) for row in seed_rounds],
        (
            "Illustrative seed values, not measurements. No federated run has been "
            "started in this process, so no round has been measured."
        ),
    )


def _seed_round_to_dict(row: Any, *, total: int) -> dict[str, Any]:
    """Shape a seeded baseline row like a federated round, marked unmeasured."""
    expected = len(_client_profiles())
    return {
        "round": row.round,
        "strategy": "seed",
        "measured": False,
        "participants": row.participants,
        "expectedParticipants": expected,
        "participatingClients": [],
        "excludedClients": [],
        "clients": [],
        "totalExamples": 0,
        "aggregatedLoss": row.global_loss,
        "aggregatedValidationLoss": None,
        "aggregatedValidationDice": None,
        "durationSeconds": row.duration_seconds,
        "startedAt": row.timestamp.isoformat(),
        "finishedAt": row.timestamp.isoformat(),
        "aggregatedParameters": 0,
    }


def metric_history(limit: int | None = None) -> list[dict[str, Any]]:
    """Chart points measured by a real run, or an empty list.

    Deliberately returns nothing when no round has been measured: a chart of
    placeholder values sitting where a result should be is exactly the confusion
    the `measured` flag exists to prevent.
    """
    _source, measured, rounds, _note = history()
    if not measured or not rounds:
        return []

    points: list[dict[str, Any]] = []
    for row in rounds:
        loss = float(row["aggregatedLoss"])
        validation = row.get("aggregatedValidationLoss")
        dice = row.get("aggregatedValidationDice")
        points.append(
            {
                "round": int(row["round"]),
                "loss": loss,
                "validationLoss": float(validation) if validation is not None else loss,
                "dice": float(dice) if dice is not None else 0.0,
                "accuracy": 0.0,
            }
        )
    return points[-limit:] if limit is not None else points


def config_summary() -> dict[str, Any]:
    """Description of the federated layer, for the API and the dashboard."""
    run = local_runner.run
    facts = _model_facts()
    return {
        "strategy": "FedAvg",
        "strategiesSupported": ["FedAvg"],
        "executionModes": [mode.value for mode in ExecutionMode],
        "currentExecution": (
            run.execution.value if run is not None else ExecutionMode.local_simulation.value
        ),
        "model": facts["name"],
        "modelParameters": facts["parameters"],
        "clientModelDescription": CLIENT_MODEL_DESCRIPTION,
        "clientProfiles": [
            {
                "name": profile.name,
                "samples": profile.samples,
                "noise": profile.noise,
                "radiusRange": list(profile.radius_range),
                "description": profile.description,
            }
            for profile in _client_profiles()
        ],
        "aggregator": AGGREGATOR_DESCRIPTION,
        "weighting": "Local training example count per client (FedAvg sample counts).",
        "clientsSend": [
            "model parameters (weights and biases)",
            "number of local training examples used",
            "mean training loss for the round",
            "validation loss and Dice on the client's own held-back split",
            "local epochs and wall-clock duration",
        ],
        "clientsNeverSend": [
            "training examples or patches",
            "labels or masks",
            "patient or study identifiers",
            "per-sample losses or gradients",
        ],
        "limitations": LIMITATIONS,
    }


def reset() -> None:
    """Clear a finished run. Used by tests and by the API's reset endpoint."""
    local_runner.reset()
