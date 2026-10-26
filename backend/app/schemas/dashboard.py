"""Schema for the dashboard overview and the health check."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import Field

from .common import CamelModel
from .federated import FederatedRound
from .hospitals import HospitalSummary
from .models import ModelVersion
from .training import MetricPoint


class ActivityKind(str, Enum):
    round = "round"
    client = "client"
    model = "model"
    system = "system"
    privacy = "privacy"


class ActivityEvent(CamelModel):
    """One line in the dashboard activity feed."""

    id: str
    timestamp: datetime
    kind: ActivityKind
    title: str
    detail: str
    hospital_id: str | None = None


class ReportActivity(CamelModel):
    """What users have ACTUALLY uploaded, counted from the live registry.

    Deliberately separate from the federated figures above. Those describe an MRI model
    that has not been trained, so they are illustrative and cannot be derived from a
    medical report; these are real counts, and zero means nothing has been uploaded yet
    rather than being a placeholder.
    """

    measured: bool = True
    sessions: int = 0
    same_patient_confirmed: int = 0
    reports_uploaded: int = 0
    reports_readable: int = 0
    reports_unreadable: int = 0
    reports_read_by_ocr: int = 0
    analyses_run: int = 0
    values_extracted: int = 0
    values_outside_printed_range: int = 0
    #: Values whose report printed NO reference range. Counted separately because this
    #: application is not entitled to a verdict on them.
    values_without_printed_range: int = 0
    documented_diagnoses: int = 0
    documented_medications: int = 0
    reports_with_red_flags: int = 0
    report_types: dict[str, int] = Field(default_factory=dict)
    latest_urgency: str | None = None
    note: str = ""


class DashboardMetrics(CamelModel):
    """Everything the dashboard landing page needs, in one response.

    ``training_status`` is a plain label (``idle``, ``running``, ``completed``)
    and ``training_trains_model`` states whether the current job trains anything.
    Both are present so the UI can never imply real training is happening.
    """

    # Hospitals
    participating_hospitals: int
    total_hospitals: int
    hospital_summary: HospitalSummary

    # Rounds
    federated_rounds: int
    last_round_at: datetime | None

    # Model
    current_model: ModelVersion | None

    # Training job
    training_status: str
    training_is_simulated: bool
    training_trains_model: bool
    # How the run was executed, and whether its numbers were measured. Present so
    # a client cannot mistake a seeded placeholder for a result.
    training_execution: str = "local-simulation"
    training_measured: bool = False
    # The seeded baseline, always labelled unmeasured and never mixed into
    # metric_history or recent_rounds.
    seeded_rounds: list[dict] = Field(default_factory=list)

    # Dataset totals
    total_studies: int
    total_size_gb: float

    # Privacy
    average_privacy_budget_used: float
    privacy_target_epsilon: float

    # Chart data. Empty until a run has actually measured something.
    metric_history: list[MetricPoint]
    recent_rounds: list[FederatedRound]

    # Activity feed, derived from the rounds, models and audit events above
    activity: list[ActivityEvent]

    # What users have actually uploaded. Real counts, not placeholders.
    report_activity: ReportActivity = Field(default_factory=ReportActivity)


class HealthStatus(CamelModel):
    """Response for GET /health.

    The four core fields are what the frontend reads; the rest is diagnostics.
    """

    status: str
    service: str
    version: str
    demo_data: bool
    environment: str
    simulated_training: bool
    api_prefix: str
    time: datetime
