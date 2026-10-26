"""Model result endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from ...schemas.common import Envelope, make_meta
from ...schemas.imaging import ImagingExperiment
from ...schemas.models import ModelComparisonRow, ModelResults, ModelTargets, ModelVersion
from ...services import imaging_service, model_service

router = APIRouter(prefix="/model", tags=["model"])


@router.get(
    "/results",
    response_model=Envelope[ModelResults],
    summary="Model results",
    description=(
        "The illustrative version registry with its derived active, best and latest "
        "versions, plus `measuredExperiment`: the measured results of a real training "
        "run, or an explicit `available: false` when none has been performed.\n\n"
        "The registry values are illustrative sample data and `registryIsIllustrative` "
        "says so. Measured values are never merged into that list; they are a separate "
        "field so the two cannot be read as one set of comparable results."
    ),
)
def model_results() -> Envelope[ModelResults]:
    return Envelope(data=model_service.results(), meta=make_meta())


@router.get(
    "/experiment",
    response_model=Envelope[ImagingExperiment],
    summary="Measured imaging experiment",
    description=(
        "The measured results of the standalone 3D U-Net experiment, read from the "
        "artifact written by `python -m app.imaging.pipeline`. Reports "
        "`available: false` with a reason rather than zeroes when no run exists. "
        "Reading this endpoint never starts training."
    ),
)
def model_experiment() -> Envelope[ImagingExperiment]:
    return Envelope(data=imaging_service.load_experiment(), meta=make_meta())


@router.get(
    "/versions",
    response_model=Envelope[list[ModelVersion]],
    summary="Model versions",
    description="The version registry, oldest first.",
)
def model_versions() -> Envelope[list[ModelVersion]]:
    return Envelope(data=model_service.list_models(), meta=make_meta())


@router.get(
    "/versions/{version}",
    response_model=Envelope[ModelVersion],
    summary="One model version",
    description="Returns 404 with a machine-readable code when the version is unknown.",
    responses={404: {"description": "Unknown version"}},
)
def model_version(version: str) -> Envelope[ModelVersion]:
    return Envelope(data=model_service.get_model(version), meta=make_meta())


@router.get(
    "/comparison",
    response_model=Envelope[list[ModelComparisonRow]],
    summary="Version comparison",
    description="Flat rows for the comparison table, with the best version flagged.",
)
def model_comparison() -> Envelope[list[ModelComparisonRow]]:
    results = model_service.results()
    rows = [
        ModelComparisonRow(
            version=model.version,
            status=model.status,
            dice=model.dice,
            iou=model.iou,
            accuracy=model.accuracy,
            loss=model.loss,
            rounds=model.rounds,
            is_best=results.best_version is not None
            and model.version == results.best_version.version,
        )
        for model in results.models
    ]
    return Envelope(data=rows, meta=make_meta())


@router.get(
    "/targets",
    response_model=Envelope[ModelTargets],
    summary="Metric targets",
    description=(
        "The design targets used to draw progress bars. They are goals, not "
        "achievements: nothing has been trained to reach them."
    ),
)
def model_targets() -> Envelope[ModelTargets]:
    return Envelope(data=ModelTargets(), meta=make_meta())
