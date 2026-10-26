"""Dashboard service: one aggregated payload for the landing page.

Everything is derived from the other services. Training figures come from the
federated layer, so what the dashboard shows is what a federated run actually
measured — and when no run has happened, the seeded baseline is sent in its own
field, marked as unmeasured, instead of being presented as a result.
"""

from __future__ import annotations

from ..schemas.dashboard import (
    ActivityEvent,
    ActivityKind,
    DashboardMetrics,
    ReportActivity,
)
from ..schemas.hospitals import Hospital
from ..schemas.models import ModelVersion
from ..schemas.security import SecurityEvent
from . import (
    federated_service,
    hospital_service,
    model_service,
    security_service,
    synthetic_data,
)


def medical_report_activity() -> dict:
    """Real upload/extraction counts from the medical-report registry.

    Imported lazily so the dashboard does not depend on the report subsystem's module
    graph at import time -- the two features are deliberately independent.
    """
    from ..medical_reports import analysis as report_analysis

    return report_analysis.activity_summary()


def safe_report_activity() -> dict:
    """The counts, or zeroes if the counter itself fails.

    The wrapper is at the CALL SITE rather than inside medical_report_activity, so a
    failure in the counter -- including one in its own body -- cannot take the dashboard
    down. A dashboard that 500s because an unrelated subsystem misbehaved would be a
    worse outcome than showing zeroes, and the report page remains the place to find out
    why if something is actually wrong.
    """
    try:
        return medical_report_activity()
    except Exception:  # noqa: BLE001
        return {}

ACTIVITY_LIMIT = 8


def _activity_from_federated(
    rounds: list[dict],
    models: list[ModelVersion],
    events: list[SecurityEvent],
    measured: bool,
) -> list[ActivityEvent]:
    """Activity feed built from state that already exists.

    Every line restates a round, a model version or an audit event from the same
    response. Rounds are only included when they were measured, so no line can
    describe a placeholder as something that happened.
    """
    items: list[ActivityEvent] = []

    # Only measured rounds become activity. The seeded baseline is deliberately
    # excluded: it used to produce lines like "Federated round 8 aggregated" from
    # placeholder values, which presented a seeded row as a real aggregation.
    for row in (rounds if measured else [])[-3:]:
        detail = (
            f"{row['participants']} of {row['expectedParticipants']} clients reported. "
            f"Loss {row['aggregatedLoss']:.3f}."
        )
        if row["excludedClients"]:
            detail += f" {len(row['excludedClients'])} client(s) excluded."
        items.append(
            ActivityEvent(
                id=f"round-{row['round']}",
                timestamp=row["finishedAt"],
                kind=ActivityKind.round,
                title=f"Federated round {row['round']} aggregated",
                detail=detail,
            )
        )

    if not items:
        items.append(
            ActivityEvent(
                id="no-run",
                timestamp=synthetic_data.SEED_NOW,
                kind=ActivityKind.system,
                title="No federated round has been measured",
                detail=(
                    "No run has completed in this process. Any baseline values shown "
                    "are illustrative and are marked measured: false."
                ),
            )
        )

    if models:
        newest = max(models, key=lambda model: model.created_at)
        items.append(
            ActivityEvent(
                id=f"model-{newest.version}",
                timestamp=newest.created_at,
                kind=ActivityKind.model,
                title=f"Model version {newest.version} recorded",
                detail=(
                    f"Status {newest.status.value}. This version registry is "
                    "illustrative; federated runs train the small client model."
                ),
            )
        )

    for event in events[:3]:
        kind = (
            ActivityKind.privacy
            if event.category.lower() in {"privacy", "transport", "integrity", "access"}
            else ActivityKind.system
        )
        items.append(
            ActivityEvent(
                id=event.id,
                timestamp=event.timestamp,
                kind=kind,
                title=event.category,
                detail=event.message,
            )
        )

    items.sort(key=lambda item: item.timestamp, reverse=True)
    return items[:ACTIVITY_LIMIT]


def metrics() -> DashboardMetrics:
    """Aggregate the current state into one response."""
    hospitals: list[Hospital] = hospital_service.list_hospitals()
    summary = hospital_service.summarise(hospitals)

    run_status = federated_service.status()
    _source, measured, rounds, _note = federated_service.history()
    has_measured_rounds = measured and bool(rounds)

    models = model_service.list_models()
    current = next(
        (model for model in models if model.status.value == "active"),
        models[-1] if models else None,
    )
    events = security_service.list_events()

    return DashboardMetrics(
        participating_hospitals=summary.online,
        total_hospitals=summary.total,
        hospital_summary=summary,
        federated_rounds=len(rounds) if has_measured_rounds else 0,
        last_round_at=rounds[-1]["finishedAt"] if has_measured_rounds else None,
        current_model=current,
        training_status=run_status["status"],
        # No longer a timer. Runs are real and their numbers are measured; the
        # residual field stays for compatibility but is now always false.
        training_is_simulated=False,
        training_trains_model=has_measured_rounds,
        training_execution=run_status["execution"],
        training_measured=measured,
        total_studies=summary.total_studies,
        total_size_gb=summary.total_size_gb,
        average_privacy_budget_used=summary.average_privacy_budget_used,
        privacy_target_epsilon=security_service.TARGET_EPSILON,
        metric_history=federated_service.metric_history() if has_measured_rounds else [],
        recent_rounds=rounds[-8:] if has_measured_rounds else [],
        # Always present and always explicitly unmeasured, so a fresh install has
        # something to chart without any claim that a run produced it.
        seeded_rounds=[_seed_row(row) for row in synthetic_data.build_seed_rounds()],
        activity=_activity_from_federated(rounds, models, events, measured),
        # Counted from reports users actually uploaded, so this is measured rather than
        # illustrative -- unlike the federated figures above, which describe a model
        # that has not been trained. Zero here means nothing has been uploaded yet.
        report_activity=ReportActivity(**safe_report_activity()),
    )


def _seed_row(row) -> dict:  # noqa: ANN001 - TrainingRound
    """Seeded baseline row, labelled as unmeasured and not from a strategy."""
    return {
        "round": row.round,
        "status": row.status.value,
        "globalLoss": row.global_loss,
        "globalDice": row.global_dice,
        "globalAccuracy": row.global_accuracy,
        "participants": row.participants,
        "expectedParticipants": row.expected_participants,
        "durationSeconds": row.duration_seconds,
        "timestamp": row.timestamp.isoformat(),
        "measured": False,
        "source": "seed",
    }
