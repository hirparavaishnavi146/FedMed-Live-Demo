"""Dashboard endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from ...schemas.common import Envelope, make_meta
from ...schemas.dashboard import DashboardMetrics
from ...services import dashboard_service

router = APIRouter(prefix="/dashboard", tags=["dashboard"])


@router.get(
    "/metrics",
    response_model=Envelope[DashboardMetrics],
    summary="Dashboard overview",
    description=(
        "One aggregated payload for the dashboard landing page: hospital counts, "
        "rounds, the active model, training status, dataset totals and the chart "
        "series. All values are synthetic."
    ),
)
def dashboard_metrics() -> Envelope[DashboardMetrics]:
    return Envelope(data=dashboard_service.metrics(), meta=make_meta())
