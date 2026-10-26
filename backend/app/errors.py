"""Application errors and the JSON error shape used by every failure response.

Every error the API raises deliberately is an ApiError. Handlers in
``app.errors`` turn those into a predictable body:

    {"error": {"code": "...", "message": "...", "detail": {...}}}
"""

from __future__ import annotations

from typing import Any


class ApiError(Exception):
    """Base class for errors that should be returned to the client as JSON."""

    status_code: int = 500
    code: str = "internal_error"

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        status_code: int | None = None,
        detail: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        if code is not None:
            self.code = code
        if status_code is not None:
            self.status_code = status_code
        self.detail = detail or {}


class NotFoundError(ApiError):
    """A requested resource does not exist."""

    status_code = 404
    code = "not_found"


class ConflictError(ApiError):
    """The request conflicts with the current state of a resource."""

    status_code = 409
    code = "conflict"


class ValidationProblem(ApiError):
    """The request was understood but its contents are not acceptable."""

    status_code = 422
    code = "validation_error"
