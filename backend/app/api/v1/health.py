"""Health check.

Deliberately outside the /api/v1 prefix and outside the response envelope so a
plain uptime probe or load balancer can read it without knowing the API shape.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from ...dependencies import get_app_settings
from ...core.config import Settings
from ...schemas.common import utc_now
from ...schemas.dashboard import HealthStatus

router = APIRouter(tags=["health"])


@router.get(
    "/health",
    response_model=HealthStatus,
    summary="Service health",
    description=(
        "Reports that the API process is answering, along with the environment "
        "and whether training is simulated. It does not check a database or a "
        "model, because neither exists yet."
    ),
)
def health(settings: Settings = Depends(get_app_settings)) -> HealthStatus:
    # Depends is required, not cosmetic: FastAPI would otherwise treat the
    # Settings annotation as a request model and build its own instance, and the
    # dependency reads app.state so the app's own settings are used.
    return HealthStatus(
        status="ok",
        service=settings.app_name,
        version=settings.version,
        demo_data=True,
        environment=settings.environment,
        simulated_training=settings.simulated_training,
        api_prefix=settings.api_prefix,
        time=utc_now(),
    )
