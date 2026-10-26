"""Hospital service: reads the simulated node list and derives summary counts.

Stage 2 has no database. When the real data layer arrives (milestone M2) this
module is the only place that has to change, because the routes and schemas
already treat it as a source of truth.
"""

from __future__ import annotations

from ..errors import NotFoundError
from ..schemas.hospitals import (
    DatasetStatus,
    Hospital,
    HospitalStatus,
    HospitalSummary,
)
from ..schemas.training import TrainingConfig
from . import synthetic_data


def list_hospitals() -> list[Hospital]:
    """All simulated nodes."""
    return synthetic_data.build_hospitals()


def get_hospital(hospital_id: str) -> Hospital:
    """One node by id, raising a 404-shaped error when it does not exist."""
    wanted = hospital_id.strip().upper()
    for hospital in list_hospitals():
        if hospital.id.upper() == wanted:
            return hospital
    raise NotFoundError(
        f"No simulated hospital with id '{hospital_id}'.",
        code="hospital_not_found",
        detail={"hospital_id": hospital_id, "available": [h.id for h in list_hospitals()]},
    )


def summarise(hospitals: list[Hospital] | None = None) -> HospitalSummary:
    """Aggregate counts and totals over the supplied nodes."""
    nodes = hospitals if hospitals is not None else list_hospitals()
    if not nodes:
        return HospitalSummary(
            total=0,
            online=0,
            degraded=0,
            offline=0,
            total_studies=0,
            total_size_gb=0.0,
            average_privacy_budget_used=0.0,
        )

    def count(status: HospitalStatus) -> int:
        return sum(1 for node in nodes if node.status is status)

    return HospitalSummary(
        total=len(nodes),
        online=count(HospitalStatus.online),
        degraded=count(HospitalStatus.degraded),
        offline=count(HospitalStatus.offline),
        total_studies=sum(node.studies for node in nodes),
        total_size_gb=round(sum(node.size_gb for node in nodes), 2),
        average_privacy_budget_used=round(
            sum(node.privacy_budget_used for node in nodes) / len(nodes), 2
        ),
    )


def nodes_ready_for_training() -> int:
    """Nodes whose local dataset is fully prepared."""
    return sum(
        1
        for node in list_hospitals()
        if node.dataset_status is DatasetStatus.ready
    )


def apply_job_state_to_nodes(config: TrainingConfig | None, running: bool) -> list[Hospital]:
    """Reflect a simulated job onto the node list.

    Only the display state changes: while a job runs, nodes that are online and
    ready report ``training``. Nothing is executed on any node.
    """
    nodes = list_hospitals()
    if not running or config is None:
        return nodes

    for node in nodes:
        if node.dataset_status is DatasetStatus.ready and node.status is HospitalStatus.online:
            node.local_training = node.local_training.__class__.training
    return nodes
