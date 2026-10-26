"""Aggregates every v1 router behind the configured API prefix."""

from __future__ import annotations

from fastapi import APIRouter

from . import (
    cases,
    dashboard,
    datasets,
    hospitals,
    medical_reports,
    models,
    security,
    training,
)

api_router = APIRouter()
api_router.include_router(dashboard.router)
api_router.include_router(hospitals.router)
api_router.include_router(datasets.router)
api_router.include_router(cases.router)
api_router.include_router(medical_reports.router)
api_router.include_router(training.router)
api_router.include_router(models.router)
api_router.include_router(security.router)
