"""Privacy and security endpoints.

These report what is actually configured and observed in the running process:

    /status      the full posture: authentication, transport, secure aggregation,
                 differential privacy, and the recorded audit counts
    /mechanisms  each mechanism with its implementation state
    /events      the real audit log written by this process
    /privacy/account  accounts the epsilon for a differential-privacy configuration

Nothing here claims a guarantee. `FeatureState.enabled` is reserved for a mechanism
that is implemented, configured and in effect; everything else reports
`disabled`, `unconfigured`, `unverified` or `not_implemented`.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query, Request

from ...errors import ValidationProblem
from ...federated.privacy import PrivacyConfig, PrivacyConfigurationError, account_privacy
from ...schemas.common import Envelope, make_meta
from ...schemas.privacy import PrivacyAccountRequest, PrivacyAccountResponse
from ...schemas.security import (
    AuditCleared,
    PrivacyMechanism,
    SecurityEvent,
    SecurityStatus,
    Severity,
)
from ...security import Principal, require_admin
from ...services import security_service

router = APIRouter(prefix="/security", tags=["security"])


@router.get(
    "/status",
    response_model=Envelope[SecurityStatus],
    summary="Privacy and security status",
    description=(
        "The real posture of this process. `authentication` reads the configured key "
        "list; `transport` reads the certificate AND the scheme of this request, so TLS "
        "reports `enabled` only when the request actually arrived over HTTPS. "
        "`features` gives each mechanism a state of enabled / disabled / unconfigured / "
        "unverified / not_implemented, and `events` is the recorded audit log rather "
        "than seeded rows. No privacy guarantee is made."
    ),
)
def security_status(request: Request) -> Envelope[SecurityStatus]:
    return Envelope(
        data=security_service.status(
            settings=request.app.state.settings,
            audit=getattr(request.app.state, "audit_log", None),
            # The decisive evidence for the transport block: how this request arrived.
            request_scheme=request.url.scheme,
        ),
        meta=make_meta(),
    )


@router.get(
    "/mechanisms",
    response_model=Envelope[list[PrivacyMechanism]],
    summary="Privacy mechanisms",
    description=(
        "Each mechanism with its implementation state. `implemented` means code exists "
        "in this repository and is covered by tests; it does not mean the mechanism is "
        "active or independently validated."
    ),
)
def security_mechanisms() -> Envelope[list[PrivacyMechanism]]:
    return Envelope(data=security_service.MECHANISMS, meta=make_meta())


@router.get(
    "/events",
    response_model=Envelope[list[SecurityEvent]],
    summary="Audit log",
    description=(
        "Events recorded by this process: authorization outcomes and training lifecycle "
        "changes. Secrets are redacted before storage. The log is bounded and "
        "process-local, so it is empty after a restart."
    ),
)
def security_events(
    request: Request,
    limit: int | None = Query(default=None, ge=1, le=200, description="Maximum entries."),
    severity: Severity | None = Query(default=None, description="Only this severity."),
) -> Envelope[list[SecurityEvent]]:
    return Envelope(
        data=security_service.list_events(
            audit=getattr(request.app.state, "audit_log", None),
            limit=limit,
            severity=severity,
        ),
        meta=make_meta(),
    )


@router.post(
    "/privacy/account",
    response_model=Envelope[PrivacyAccountResponse],
    summary="Account differential privacy",
    description=(
        "Computes the epsilon spent by a differential-privacy configuration using "
        "Opacus's RDP accountant, and returns it together with the assumptions it "
        "depends on. The privacy unit is one client's whole update, not an individual "
        "patient record, and the bound is not independently validated."
    ),
    responses={422: {"description": "Invalid privacy configuration"}},
)
def account_privacy_endpoint(payload: PrivacyAccountRequest) -> Envelope[PrivacyAccountResponse]:
    try:
        config = PrivacyConfig(
            enabled=payload.enabled,
            noise_multiplier=payload.noise_multiplier,
            clipping_norm=payload.clipping_norm,
            sample_rate=payload.sample_rate,
            num_rounds=payload.num_rounds,
            target_delta=payload.target_delta,
            target_epsilon=payload.target_epsilon,
        )
    except PrivacyConfigurationError as exc:
        raise ValidationProblem(
            str(exc), code="invalid_privacy_configuration", detail={"reason": str(exc)}
        ) from exc

    report = account_privacy(config)
    payload_dict = report.as_dict()
    return Envelope(
        data=PrivacyAccountResponse(
            epsilon=payload_dict["epsilon"],
            delta=payload_dict["delta"],
            steps=payload_dict["steps"],
            target_epsilon=payload_dict["targetEpsilon"],
            within_budget=payload_dict["withinBudget"],
            accountant=payload_dict["accountant"],
            mechanism=payload_dict["mechanism"],
            available=payload_dict["available"],
            unavailable_reason=payload_dict["unavailableReason"],
            privacy_unit=payload_dict["privacyUnit"],
            privacy_unit_short=payload_dict["privacyUnit"],
            config=payload_dict["config"],
            config_description=payload_dict["configDescription"],
            assumptions=payload_dict["assumptions"],
        ),
        meta=make_meta(),
    )


@router.post(
    "/audit/clear",
    response_model=Envelope[AuditCleared],
    summary="Clear the audit log",
    description=(
        "Drops every recorded audit entry. Requires the 'admin' scope. This is an "
        "administrative action on security-relevant state, which is why it is the only "
        "endpoint behind the admin role."
    ),
    responses={
        401: {"description": "Missing or invalid API key"},
        403: {"description": "Key lacks the admin scope"},
        503: {"description": "Authentication is not configured"},
    },
)
def clear_audit(
    request: Request,
    principal: Principal = Depends(require_admin),
) -> Envelope[AuditCleared]:
    audit = getattr(request.app.state, "audit_log", None)
    removed = len(audit) if audit is not None else 0
    if audit is not None:
        audit.clear()

    if audit is not None:
        # Record the clearing itself, so the log is not silently empty afterwards.
        audit.record(
            category="Access",
            message=(
                f"Audit log cleared by '{principal.label}', removing {removed} entr"
                f"{'y' if removed == 1 else 'ies'}."
            ),
            severity="warning",
            source=principal.label,
        )

    return Envelope(
        data=AuditCleared(removed=removed, cleared_by=principal.label),
        meta=make_meta(),
    )
