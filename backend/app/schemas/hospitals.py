"""Schemas for simulated hospital nodes."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from .common import CamelModel


class HospitalStatus(str, Enum):
    online = "online"
    degraded = "degraded"
    offline = "offline"


class DatasetStatus(str, Enum):
    ready = "ready"
    partial = "partial"
    not_prepared = "not-prepared"


class LocalTrainingStatus(str, Enum):
    idle = "idle"
    training = "training"
    complete = "complete"
    failed = "failed"


class Hospital(CamelModel):
    """One simulated research site.

    ``node_id`` is the address a Flower client would use once federated learning
    exists (milestone M4). Nothing listens on it today.
    """

    id: str
    name: str
    location: str
    node_id: str
    status: HospitalStatus
    dataset_status: DatasetStatus
    studies: int
    size_gb: float
    last_sync: datetime
    local_training: LocalTrainingStatus
    rounds_contributed: int
    privacy_budget_used: float
    split: str
    notes: str


class HospitalSummary(CamelModel):
    """Aggregate counts over the simulated nodes."""

    total: int
    online: int
    degraded: int
    offline: int
    total_studies: int
    total_size_gb: float
    average_privacy_budget_used: float
