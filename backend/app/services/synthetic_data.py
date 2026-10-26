"""The synthetic dataset.

This module is the single source of sample values for the whole backend. Every
other service derives its output from here, so there is exactly one place to
change when the real data layer arrives at milestone M2/M3.

NOTHING IN THIS FILE IS REAL. The hospital names are invented, the study counts
are invented, and no scan, patient or scanner is referenced anywhere.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from ..schemas.hospitals import (
    DatasetStatus,
    Hospital,
    HospitalStatus,
    LocalTrainingStatus,
)
from ..schemas.models import ModelStatus, ModelVersion
from ..schemas.security import Severity, SecurityEvent

# Fixed reference point for the seeded history, so repeated calls describe a
# consistent world instead of drifting every second. "Now" is used only for
# timestamps that should look fresh (job start, generated_at).
SEED_NOW = datetime.now(timezone.utc)

HOSPITAL_SEED: list[dict[str, object]] = [
    {
        "id": "HOSP-01",
        "name": "Northbridge Medical Centre",
        "location": "Boston, United States",
        "node_id": "client-northbridge:8081",
        "status": HospitalStatus.online,
        "dataset_status": DatasetStatus.ready,
        "studies": 412,
        "size_gb": 38.4,
        "sync_minutes_ago": 22,
        "local_training": LocalTrainingStatus.complete,
        "rounds_contributed": 8,
        "privacy_budget_used": 0.72,
        "split": "train 70% / val 15% / test 15%",
        "notes": (
            "Synthetic brain MRI phantoms with T1-weighted contrast. "
            "No identifiable patient data."
        ),
    },
    {
        "id": "HOSP-02",
        "name": "St. Aurelia University Hospital",
        "location": "Utrecht, Netherlands",
        "node_id": "client-aurelia:8082",
        "status": HospitalStatus.online,
        "dataset_status": DatasetStatus.ready,
        "studies": 356,
        "size_gb": 31.1,
        "sync_minutes_ago": 20,
        "local_training": LocalTrainingStatus.training,
        "rounds_contributed": 8,
        "privacy_budget_used": 1.15,
        "split": "train 75% / val 12.5% / test 12.5%",
        "notes": "Synthetic phantom volumes, harmonised voxel spacing before local training.",
    },
    {
        "id": "HOSP-03",
        "name": "Riverside Children's Research Institute",
        "location": "Toronto, Canada",
        "node_id": "client-riverside:8083",
        "status": HospitalStatus.degraded,
        "dataset_status": DatasetStatus.partial,
        "studies": 198,
        "size_gb": 17.6,
        "sync_minutes_ago": 66,
        "local_training": LocalTrainingStatus.failed,
        "rounds_contributed": 6,
        "privacy_budget_used": 2.04,
        "split": "train 60% / val 20% / test 20%",
        "notes": (
            "Two of three synthetic sub-datasets prepared. Last local run stopped "
            "on a memory limit (simulated)."
        ),
    },
]

# (round, loss, dice, accuracy, participants, duration_seconds)
ROUND_CURVE: list[tuple[int, float, float, float, int, int]] = [
    (1, 0.842, 0.412, 0.731, 3, 512),
    (2, 0.671, 0.507, 0.786, 3, 498),
    (3, 0.548, 0.581, 0.822, 3, 505),
    (4, 0.431, 0.646, 0.858, 3, 489),
    (5, 0.352, 0.702, 0.883, 3, 473),
    (6, 0.288, 0.746, 0.902, 2, 441),
    (7, 0.241, 0.779, 0.917, 3, 455),
    (8, 0.209, 0.804, 0.928, 3, 462),
]

# Seconds between the seeded round timestamps (30 simulated minutes).
ROUND_INTERVAL_SECONDS = 1920


def round_offset_seconds(index: int, total: int) -> int:
    """Seconds before SEED_NOW at which seeded round ``index`` finished."""
    return (total - index) * ROUND_INTERVAL_SECONDS


MODEL_SEED: list[dict[str, object]] = [
    {
        "version": "v0.1.0",
        "hours_ago": 15,
        "status": ModelStatus.archived,
        "dice": 0.581,
        "iou": 0.412,
        "accuracy": 0.822,
        "loss": 0.548,
        "sensitivity": 0.604,
        "specificity": 0.901,
        "rounds": 3,
        "notes": "First smoke-test of the aggregation loop. Single-fold validation.",
    },
    {
        "version": "v0.2.0",
        "hours_ago": 2,
        "status": ModelStatus.active,
        "dice": 0.746,
        "iou": 0.596,
        "accuracy": 0.902,
        "loss": 0.288,
        "sensitivity": 0.771,
        "specificity": 0.938,
        "rounds": 6,
        "notes": "Best candidate so far. Excludes one client that failed mid-round.",
    },
    {
        "version": "v0.3.0",
        "hours_ago": 0,
        "status": ModelStatus.candidate,
        "dice": 0.804,
        "iou": 0.671,
        "accuracy": 0.928,
        "loss": 0.209,
        "sensitivity": 0.829,
        "specificity": 0.951,
        "rounds": 8,
        "notes": "Newest aggregation result. Not yet reviewed by the research team.",
    },
]

SECURITY_EVENT_SEED: list[dict[str, object]] = [
    ("sec-6", 25, Severity.info, "Transport", "TLS handshake completed with client-aurelia.", "client-aurelia:8082"),
    ("sec-5", 43, Severity.warning, "Integrity", "Update checksum verified after one retry.", "client-riverside:8083"),
    ("sec-4", 60, Severity.info, "Access", "Research role signed in to the dashboard.", "dashboard"),
    ("sec-3", 66, Severity.warning, "Availability", "Client missed a round deadline and was excluded.", "client-riverside:8083"),
    ("sec-2", 94, Severity.info, "Privacy", "Differential-privacy budget reset for the new session.", "aggregator"),
    ("sec-1", 120, Severity.info, "System", "Audit log started. Entries below are simulated.", "aggregator"),
]


def build_hospitals(now: datetime | None = None) -> list[Hospital]:
    """Materialise the hospital list with timestamps relative to ``now``."""
    reference = now or datetime.now(timezone.utc)
    hospitals: list[Hospital] = []
    for entry in HOSPITAL_SEED:
        data = dict(entry)
        minutes = int(data.pop("sync_minutes_ago"))  # type: ignore[arg-type]
        hospitals.append(
            Hospital(
                **data,  # type: ignore[arg-type]
                last_sync=reference - timedelta(minutes=minutes),
            )
        )
    return hospitals


def build_seed_rounds(limit: int | None = None) -> list[TrainingRound]:
    """The baseline round history that exists before any job is started."""
    from ..schemas.training import RoundStatus, TrainingRound

    total = len(ROUND_CURVE)
    rows = ROUND_CURVE[: limit if limit is not None else total]
    return [
        TrainingRound(
            round=number,
            status=RoundStatus.complete,
            global_loss=loss,
            global_dice=dice,
            global_accuracy=accuracy,
            participants=participants,
            expected_participants=len(HOSPITAL_SEED),
            duration_seconds=duration,
            timestamp=SEED_NOW - timedelta(seconds=(total - number) * ROUND_INTERVAL_SECONDS),
        )
        for number, loss, dice, accuracy, participants, duration in rows
    ]


def build_models(now: datetime | None = None) -> list[ModelVersion]:
    reference = now or datetime.now(timezone.utc)
    models: list[ModelVersion] = []
    for entry in MODEL_SEED:
        data = dict(entry)
        hours = int(data.pop("hours_ago"))  # type: ignore[arg-type]
        models.append(
            ModelVersion(
                **data,  # type: ignore[arg-type]
                created_at=reference - timedelta(hours=hours),
            )
        )
    return models


def build_security_events(now: datetime | None = None) -> list[SecurityEvent]:
    reference = now or datetime.now(timezone.utc)
    return [
        SecurityEvent(
            id=event_id,
            timestamp=reference - timedelta(minutes=minutes),
            severity=severity,
            category=category,
            message=message,
            source=source,
        )
        for event_id, minutes, severity, category, message, source in SECURITY_EVENT_SEED
    ]
