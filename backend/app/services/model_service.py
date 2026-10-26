"""Model service: the illustrative version registry and its derived selections.

The registry is seeded sample data — no model produced its numbers. Measured
results from an actual training run come from `imaging_service`, which reads the
artifact written by `app.imaging.pipeline`, and are returned in a separate field.
Keeping the two apart is the point: merging them would present placeholder values
and measurements as one list of comparable results.
"""

from __future__ import annotations

from ..errors import NotFoundError
from ..schemas.models import ModelResults, ModelStatus, ModelVersion
from . import imaging_service, synthetic_data


def list_models() -> list[ModelVersion]:
    """All versions, oldest first."""
    return synthetic_data.build_models()


def get_model(version: str) -> ModelVersion:
    wanted = version.strip().lower()
    for model in list_models():
        if model.version.lower() == wanted:
            return model
    raise NotFoundError(
        f"No model version '{version}'.",
        code="model_not_found",
        detail={"version": version, "available": [m.version for m in list_models()]},
    )


def _pick(models: list[ModelVersion], status: ModelStatus) -> ModelVersion | None:
    for model in models:
        if model.status is status:
            return model
    return None


def results() -> ModelResults:
    """The payload for GET /api/v1/model/results.

    ``active_version`` is the version currently marked active, ``best_version``
    the highest Dice, and ``latest_version`` the most recently created. They are
    separate on purpose: the newest version is not automatically the best.

    ``measured_experiment`` carries the results of a real training run when one
    exists, and an explicit ``available: false`` when none does. It is never
    merged into ``models``.
    """
    models = list_models()
    best = max(models, key=lambda m: m.dice) if models else None
    latest = max(models, key=lambda m: m.created_at) if models else None
    return ModelResults(
        models=models,
        active_version=_pick(models, ModelStatus.active),
        best_version=best,
        latest_version=latest,
        measured_experiment=imaging_service.load_experiment(),
    )
