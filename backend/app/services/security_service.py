"""Security service: the honest privacy/security posture.

Two things changed in Stage 5:

    * the mechanisms below now describe real, implemented code, so `implemented`
      reflects the repository rather than a plan
    * `events` comes from the process's own audit log — recorded authorization
      outcomes and lifecycle changes — instead of seeded sample rows

The posture blocks (authentication, transport, secure aggregation, privacy
accounting) are built by `app.security.posture` from what is actually configured and
from the scheme of the live request.
"""

from __future__ import annotations

from typing import Any

from ..schemas.security import (
    DifferentialPrivacyConfig,
    EncryptionStatus,
    MechanismState,
    PrivacyBudget,
    PrivacyMechanism,
    SecurityEvent,
    SecurityStatus,
    Severity,
)
from ..security.posture import build_posture
from . import hospital_service

TARGET_EPSILON = 3.0

MECHANISMS: list[PrivacyMechanism] = [
    PrivacyMechanism(
        id="dp",
        name="Differential privacy",
        description=(
            "Client-side clipping and Gaussian noise on the aggregate, accounted with "
            "Opacus's RDP accountant. Implemented and covered by tests, but not "
            "configured to run: no epsilon has been spent."
        ),
        state=MechanismState.implemented,
        implemented=True,
        reference="app/federated/privacy.py",
    ),
    PrivacyMechanism(
        id="secagg",
        name="Secure aggregation",
        description=(
            "Clients mask their updates with pairwise pseudorandom masks so the server "
            "recovers only the weighted sum. Implemented with Flower's own masking "
            "primitive and covered by tests; not active until a shared secret is set."
        ),
        state=MechanismState.implemented,
        implemented=True,
        reference="app/federated/secure_aggregation.py",
    ),
    PrivacyMechanism(
        id="tls",
        name="Transport encryption",
        description=(
            "TLS support for the API (HTTPS) and for Flower's gRPC path, with a "
            "development self-signed certificate generator. Whether it is in effect is "
            "reported from the live request scheme, not assumed from configuration."
        ),
        state=MechanismState.implemented,
        implemented=True,
        reference="app/security/tls.py",
    ),
    PrivacyMechanism(
        id="he",
        name="Homomorphic encryption",
        description=(
            "Federated averaging over encrypted updates (TenSEAL/CKKS). Not "
            "implemented: no code imports TenSEAL and no encrypted computation runs."
        ),
        state=MechanismState.planned,
        implemented=False,
        reference="Not implemented",
    ),
]

ENCRYPTION = EncryptionStatus(
    transport="configuration-dependent",
    at_rest="not implemented",
    note=(
        "TLS is implemented and can be enabled; the live state is reported in "
        "`transport`, which reads the scheme of the request actually being served. "
        "Nothing at rest is encrypted or protected."
    ),
)

DP_CONFIG = DifferentialPrivacyConfig(
    mechanism="Gaussian (subsampled), client-level DP",
    noise_multiplier=1.0,
    clipping_norm=1.0,
    target_epsilon=TARGET_EPSILON,
    applied=False,
)


def _audit_events(audit: Any) -> list[SecurityEvent]:
    """The real audit log, mapped to the response shape.

    Returns an empty list when there is no log or nothing has been recorded. An empty
    history is the truthful answer for a process that has not done anything, and it is
    better than seeded rows that look like evidence.
    """
    if audit is None:
        return []
    records = audit.events()
    return [
        SecurityEvent(
            id=record.id,
            timestamp=record.timestamp,
            severity=Severity(record.severity),
            category=record.category,
            message=record.message,
            source=record.source,
        )
        for record in records
    ]


def list_events(
    audit: Any = None, limit: int | None = None, severity: Severity | None = None
) -> list[SecurityEvent]:
    """Recorded audit entries, newest first, optionally filtered."""
    events = _audit_events(audit)
    if severity is not None:
        events = [event for event in events if event.severity is severity]
    if limit is not None:
        events = events[:limit]
    return events


def budgets() -> list[PrivacyBudget]:
    """Simulated epsilon usage per node against the configured target.

    These are illustrative node figures from the hospital fixtures, not spent
    privacy budgets from a real mechanism. The `privacy_accounting` block reports the
    accounted value, which is absent until DP is configured.
    """
    return [
        PrivacyBudget(
            hospital_id=node.id,
            hospital=node.name,
            epsilon=node.privacy_budget_used,
            target_epsilon=TARGET_EPSILON,
            exceeded=node.privacy_budget_used > TARGET_EPSILON,
        )
        for node in hospital_service.list_hospitals()
    ]


def status(
    settings: Any = None,
    audit: Any = None,
    request_scheme: str = "http",
) -> SecurityStatus:
    """The full payload for GET /api/v1/security/status."""
    implemented = sum(1 for mechanism in MECHANISMS if mechanism.implemented)

    posture: dict[str, Any] = {}
    if settings is not None:
        posture = build_posture(settings, audit=audit, request_scheme=request_scheme)

    return SecurityStatus(
        mechanisms_implemented=implemented,
        mechanisms_total=len(MECHANISMS),
        encryption=ENCRYPTION,
        differential_privacy=DP_CONFIG,
        secure_aggregation_enabled=bool(
            posture.get("secure_aggregation")
            and posture["secure_aggregation"].state.value == "enabled"
        ),
        homomorphic_encryption_enabled=False,
        mechanisms=MECHANISMS,
        budgets=budgets(),
        events=_audit_events(audit),
        features=posture.get("features", []),
        authentication=posture.get("authentication"),
        transport=posture.get("transport"),
        secure_aggregation=posture.get("secure_aggregation"),
        privacy_accounting=posture.get("privacy_accounting"),
        audit=posture.get("audit"),
        disclaimers=posture.get("disclaimers", []),
    )
