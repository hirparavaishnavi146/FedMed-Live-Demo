"""Hospital endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Query

from ...schemas.common import Envelope, make_meta
from ...schemas.hospitals import Hospital, HospitalStatus, HospitalSummary
from ...services import hospital_service

router = APIRouter(prefix="/hospitals", tags=["hospitals"])


@router.get(
    "",
    response_model=Envelope[list[Hospital]],
    summary="List simulated hospital nodes",
    description=(
        "The simulated research sites. No real hospital is connected and no "
        "patient data is present. Supports optional filtering by connection "
        "status and a free-text search over name, id and node address."
    ),
)
def list_hospitals(
    status: HospitalStatus | None = Query(
        default=None,
        description="Only return nodes in this connection state.",
    ),
    search: str | None = Query(
        default=None,
        max_length=120,
        description="Case-insensitive match against name, id, location or node address.",
    ),
) -> Envelope[list[Hospital]]:
    nodes = hospital_service.list_hospitals()

    if status is not None:
        nodes = [node for node in nodes if node.status is status]

    if search:
        needle = search.strip().lower()
        nodes = [
            node
            for node in nodes
            if needle in f"{node.name} {node.id} {node.location} {node.node_id}".lower()
        ]

    return Envelope(data=nodes, meta=make_meta())


@router.get(
    "/summary",
    response_model=Envelope[HospitalSummary],
    summary="Hospital summary",
    description="Counts and totals across the simulated nodes.",
)
def hospital_summary() -> Envelope[HospitalSummary]:
    return Envelope(data=hospital_service.summarise(), meta=make_meta())


@router.get(
    "/{hospital_id}",
    response_model=Envelope[Hospital],
    summary="One hospital node",
    description="Returns 404 with a machine-readable error code when the id is unknown.",
    responses={404: {"description": "Unknown hospital id"}},
)
def get_hospital(hospital_id: str) -> Envelope[Hospital]:
    return Envelope(data=hospital_service.get_hospital(hospital_id), meta=make_meta())


@router.post(
    "/{hospital_id}/sync",
    response_model=Envelope[Hospital],
    summary="Request a node synchronisation (not implemented)",
    description=(
        "Refused with HTTP 501. Synchronisation needs the Flower server, which "
        "arrives at milestone M4, so this endpoint reports honestly instead of "
        "pretending to do the work."
    ),
    responses={501: {"description": "Synchronisation is not implemented"}},
)
def sync_hospital(hospital_id: str) -> Envelope[Hospital]:
    from ...errors import ApiError

    # Validate the id first, so a typo still returns 404 rather than 501.
    hospital_service.get_hospital(hospital_id)
    raise ApiError(
        "Node synchronisation is not implemented. It requires the Flower server "
        "and client nodes, which arrive at milestone M4.",
        code="not_implemented",
        status_code=501,
        detail={"hospital_id": hospital_id, "planned": "milestone M4"},
    )
