"""Error types for the federated learning module.

Every failure the federated layer raises on purpose is a FederatedError, so a
caller can catch one base class and still read a specific `code` and `detail`.
The API layer maps these onto HTTP status codes.
"""

from __future__ import annotations

from typing import Any


class FederatedError(Exception):
    """Base class for federated learning failures."""

    code: str = "federated_error"

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        detail: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        if code is not None:
            self.code = code
        self.detail = detail or {}


class InvalidWeightError(FederatedError):
    """Aggregation weights are missing, negative, non-finite or all zero."""

    code = "invalid_weight"


class UpdateValidationError(FederatedError):
    """A client update does not fit the expected model structure.

    Raised for missing or extra parameters, shape mismatches, non-numeric
    payloads and non-finite values. An update that cannot be safely averaged is
    refused rather than silently coerced.
    """

    code = "invalid_update"


class NoEligibleClientsError(FederatedError):
    """Fewer clients are available than the round requires."""

    code = "no_eligible_clients"


class ClientTrainingError(FederatedError):
    """A client failed while training locally.

    The client is excluded from the round; the round itself may still proceed if
    enough clients remain.
    """

    code = "client_training_failed"


class RoundConfigurationError(FederatedError):
    """The requested federated configuration is not usable."""

    code = "invalid_round_config"
