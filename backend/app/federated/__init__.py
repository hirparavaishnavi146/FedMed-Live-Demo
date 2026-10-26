"""FedMed federated learning core.

A real, runnable federated learning implementation:

    * `fedavg`     — sample-count-weighted averaging with validation
    * `client`     — the hospital client: trains locally, sends parameters only
    * `simulation` — server orchestration (in-process runner and round loop)
    * `flower_app` — the same loop across separate Flower processes over gRPC
    * `config`     — round configuration and seeds
    * `results`    — round results and run history
    * `errors`     — typed failures

What is real here: local training with genuine gradients, genuine sample-count
weighting, genuine aggregation, genuine per-round measurements.

What is not: the data is synthetic, and a run on one machine does not demonstrate
a cross-silo deployment across separate hospital infrastructures. See
docs/FEDERATED.md for what would still be required.
"""

from .config import RoundConfig
from .errors import (
    ClientTrainingError,
    FederatedError,
    InvalidWeightError,
    NoEligibleClientsError,
    RoundConfigurationError,
    UpdateValidationError,
)
from .fedavg import aggregate_losses, validate_weights, weighted_average

__all__ = [
    "RoundConfig",
    "FederatedError",
    "InvalidWeightError",
    "UpdateValidationError",
    "NoEligibleClientsError",
    "ClientTrainingError",
    "RoundConfigurationError",
    "weighted_average",
    "validate_weights",
    "aggregate_losses",
]
