"""Exception handlers.

Every failure leaves the API in the same JSON shape, so a client only needs one
error-parsing path:

    {"error": {"code": "...", "message": "...", "detail": {...}}}
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from .errors import ApiError

logger = logging.getLogger("fedmed.errors")

# Friendly codes for the framework-generated statuses we do not raise ourselves.
_FRAMEWORK_CODES = {
    404: "not_found",
    405: "method_not_allowed",
    422: "validation_error",
}


def _to_camel(key: str) -> str:
    """snake_case -> camelCase, idempotent for keys that are already camelCase."""
    head, *rest = key.split("_")
    return head + "".join(part[:1].upper() + part[1:] for part in rest)


def _payload(code: str, message: str, detail: dict[str, object] | None = None) -> dict[str, object]:
    # Error detail keys are camelCased too, so a client parses one casing
    # everywhere (the success models already serialise to camelCase).
    camel_detail = {_to_camel(key): value for key, value in (detail or {}).items()}
    return {"error": {"code": code, "message": message, "detail": camel_detail}}


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def handle_api_error(_: Request, exc: ApiError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=_payload(exc.code, exc.message, exc.detail),
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        _: Request, exc: RequestValidationError
    ) -> JSONResponse:
        # Keep only the fields that help the client fix the request. `ctx` can
        # contain exception objects, which are not JSON serialisable.
        problems = [
            {
                "location": ".".join(str(part) for part in error.get("loc", ())),
                "message": error.get("msg", "invalid value"),
                "type": error.get("type", "value_error"),
            }
            for error in exc.errors()
        ]
        first = problems[0]["message"] if problems else "Invalid request"
        return JSONResponse(
            status_code=422,
            content=_payload(
                "validation_error",
                first,
                {"problems": problems, "problemCount": len(problems)},
            ),
        )

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = _FRAMEWORK_CODES.get(exc.status_code, "http_error")
        detail = exc.detail if isinstance(exc.detail, str) else "Request failed"
        return JSONResponse(
            status_code=exc.status_code,
            content=_payload(code, detail, {"statusCode": exc.status_code}),
        )

    @app.exception_handler(Exception)
    async def handle_unexpected(_: Request, exc: Exception) -> JSONResponse:
        # Log the real cause, return a generic body: internal details must not
        # leak into a response.
        logger.exception("Unhandled error while serving a request", exc_info=exc)
        return JSONResponse(
            status_code=500,
            content=_payload(
                "internal_error",
                "The server hit an unexpected error. Check the server log for details.",
            ),
        )
