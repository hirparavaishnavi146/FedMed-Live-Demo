"""Shared FastAPI dependencies."""

from __future__ import annotations

from fastapi import Request

from .core.config import Settings


def get_app_settings(request: Request) -> Settings:
    """The settings the running application was created with.

    This reads from ``app.state`` rather than calling ``get_settings()`` on
    purpose: ``get_settings`` is cached, so a route that used it would silently
    read the process-wide defaults and ignore the settings an app passed to
    ``create_app`` (which is exactly what a test or a second deployment does).
    """
    return request.app.state.settings
