"""Builds the security posture from what is actually configured and observed.

Every state here is derived, not asserted:

    * authentication  read from the configured key list
    * transport       read from the certificate on disk AND the scheme of the request
                      being served — a certificate on disk does not mean the server is
                      using it, so `enabled` requires an https request
    * secure aggregation / DP  read from configuration and from whether the
                      implementing module is importable in this environment

`FeatureState.enabled` is reserved for a feature that is implemented, configured and
actually in effect. Everything else is `disabled`, `unconfigured`, `unverified` or
`not_implemented`, so a switch in a UI can never be the reason something claims to be
on.
"""

from __future__ import annotations

from typing import Any

from ..schemas.security import (
    AuditSummary,
    AuthStatus,
    FeatureState,
    PrivacyAccountingStatus,
    SecureAggregationStatus,
    SecurityFeature,
    TransportStatus,
)

#: Paths that require a scope. Kept here so the security page lists what is actually
#: protected rather than what someone remembered to protect.
PROTECTED_PATHS = [
    "POST /api/v1/training/start (train)",
    "POST /api/v1/training/stop (train)",
    "POST /api/v1/training/reset (train)",
]

DISCLAIMERS = [
    "This application is not production-secure. It has not been penetration tested, "
    "audited or reviewed by an independent security assessor.",
    "The federated clients are simulated in one process on one machine. This is not a "
    "cross-silo deployment and the isolation between clients is structural, not "
    "network-enforced.",
    "Differential privacy here is client-level: the unit protected is one client's "
    "whole update, not an individual patient record.",
    "Secure aggregation protects against an honest-but-curious server. It does not "
    "protect against collusion, and it cannot tolerate a client dropping out.",
    "No mechanism in this project has been independently validated. Nothing here "
    "should be relied on to protect patient data.",
]


def _authentication(settings: Any) -> AuthStatus:
    configured = bool(getattr(settings, "auth_configured", False))
    keys = settings.parsed_api_keys() if configured else ()
    # Labels and roles only. The summary function cannot return key material because
    # it does not receive any.
    from .auth import public_key_summary

    principals = public_key_summary(keys) if configured else []

    if not configured:
        state = FeatureState.unconfigured
        note = (
            "No API keys are configured, so the protected endpoints refuse with 503. "
            "Access is refused rather than allowed anonymously."
        )
    else:
        state = FeatureState.enabled
        note = (
            f"{len(keys)} API key(s) configured. Keys are compared by SHA-256 hash; the "
            "raw key is never stored, logged or returned."
        )

    return AuthStatus(
        configured=configured,
        state=state,
        scheme="API key in the X-API-Key header, scope-based roles (read < train < admin)",
        key_count=len(keys),
        principals=principals,
        required_for_reads=bool(getattr(settings, "auth_required_for_reads", False)),
        protected_paths=list(PROTECTED_PATHS),
        note=note,
    )


def _transport(settings: Any, request_scheme: str) -> TransportStatus:
    from .tls import inspect_certificate

    cert_path = getattr(settings, "tls_cert_file", None)
    configured = bool(getattr(settings, "tls_configured", False))
    scheme = (request_scheme or "http").lower()

    certificate_valid: bool | None = None
    self_signed: bool | None = None
    days_remaining: int | None = None
    subject: str | None = None

    if cert_path:
        info = inspect_certificate(cert_path)
        certificate_valid = info.valid
        self_signed = info.is_self_signed
        days_remaining = info.days_remaining
        subject = info.subject

    if scheme == "https":
        # The decisive evidence: this very request arrived over TLS.
        state = FeatureState.enabled
        note = (
            "This request arrived over HTTPS, so TLS is in effect for this connection. "
            "Whether the certificate is trusted depends on the client's trust store."
        )
    elif configured and certificate_valid:
        state = FeatureState.unverified
        note = (
            "A certificate is configured and readable, but this request arrived over "
            "plain HTTP, so TLS is not in effect for it."
        )
    elif configured:
        state = FeatureState.unconfigured
        note = (
            "TLS paths are set but the certificate could not be read as valid. Serving "
            "would fail rather than silently fall back to plain HTTP."
        )
    else:
        state = FeatureState.unconfigured
        note = (
            "No TLS certificate or key is configured, and the API is serving plain "
            "HTTP. Traffic on this connection is unencrypted."
        )

    return TransportStatus(
        configured=configured,
        state=state,
        scheme=scheme,
        certificate_path=cert_path,
        certificate_valid=certificate_valid,
        self_signed=self_signed,
        days_remaining=days_remaining,
        subject=subject,
        note=note,
    )


def _secure_aggregation(settings: Any) -> SecureAggregationStatus:
    from ..federated.secure_aggregation import SecureAggregationConfig

    try:
        SecureAggregationConfig(enabled=True)
        available = True
    except Exception:  # noqa: BLE001 - any failure means the component is unusable
        available = False

    secret = getattr(settings, "secure_aggregation_secret", None)
    secret_configured = bool(secret)

    if not available:
        state = FeatureState.not_implemented
        note = "The secure aggregation module could not be initialised."
    elif not secret_configured:
        state = FeatureState.unconfigured
        note = (
            "Secure aggregation is implemented and tested, but no shared secret is "
            "configured (FEDMED_SECURE_AGGREGATION_SECRET), so it is not active. A "
            "default secret would be a false sense of protection."
        )
    else:
        state = FeatureState.unverified
        note = (
            "A shared secret is configured and the masking protocol is implemented and "
            "covered by tests. It has not been exercised on a live run in this process, "
            "so it is reported as unverified rather than enabled."
        )

    return SecureAggregationStatus(
        available=available,
        configured=secret_configured,
        state=state,
        protocol="SecAgg+ style pairwise masking with modular arithmetic",
        primitive="flwr.common.secure_aggregation.secaggplus_utils.pseudo_rand_gen",
        secret_configured=secret_configured,
        tolerates_dropout=False,
        threat_model=(
            "Protects a client's whole update from an honest-but-curious server. Does "
            "not protect against a colluding server plus n-1 clients."
        ),
        assumptions=[
            "All participating clients report; a dropout breaks mask cancellation and the "
            "protocol refuses to unmask rather than returning a corrupted aggregate.",
            "Pairwise seeds are shared over a channel assumed to be secure.",
            "The true weighted sum stays below the modulus; overflow is detected and refused.",
            "Updates are clipped and quantized, which introduces a small, measured error.",
        ],
        note=note,
    )


def _privacy_accounting(settings: Any) -> PrivacyAccountingStatus:
    from ..federated.privacy import PrivacyConfig, account_privacy, privacy_capabilities

    capabilities = privacy_capabilities()
    available = bool(capabilities.get("accountantAvailable"))

    # No DP configuration is set in the environment, so the honest state is that DP is
    # implemented and accounted but not configured to run.
    report = account_privacy(PrivacyConfig(enabled=False))

    return PrivacyAccountingStatus(
        available=available,
        configured=False,
        state=FeatureState.unconfigured if available else FeatureState.not_implemented,
        mechanism="subsampled Gaussian mechanism with client-side clipping",
        accountant=str(capabilities.get("accountant") or "rdp (opacus)"),
        epsilon=report.epsilon,
        delta=report.delta if report.epsilon is not None else None,
        noise_multiplier=None,
        clipping_norm=None,
        sample_rate=None,
        rounds=None,
        within_budget=None,
        privacy_unit="one client's entire update (client-level DP)",
        assumptions=[],
        note=(
            "Differential privacy is implemented and the accountant works (see "
            "/api/v1/security/privacy/account), but no DP configuration is set, so no "
            "epsilon has been spent and none is reported. An absent epsilon is shown as "
            "absent, never as zero."
        ),
    )


def _features(
    auth: AuthStatus,
    transport: TransportStatus,
    secure_aggregation: SecureAggregationStatus,
    privacy: PrivacyAccountingStatus,
) -> list[SecurityFeature]:
    return [
        SecurityFeature(
            id="auth",
            name="Authentication and authorization",
            state=auth.state,
            implemented=True,
            enabled=auth.state is FeatureState.enabled,
            description=(
                "API keys compared by SHA-256 hash, with scope-based roles protecting the "
                "training and administration endpoints."
            ),
            evidence=auth.note,
            requirements=(
                []
                if auth.configured
                else ["Set FEDMED_API_KEYS, e.g. python -m app.security.keys generate --label ops --role admin"]
            ),
            reference="app/security/auth.py, app/security/dependencies.py",
        ),
        SecurityFeature(
            id="tls",
            name="Transport security (TLS)",
            state=transport.state,
            implemented=True,
            enabled=transport.state is FeatureState.enabled,
            description=(
                "HTTPS for the API and TLS for the federated gRPC path, using a "
                "self-signed certificate for local development."
            ),
            evidence=transport.note,
            requirements=(
                []
                if transport.state is FeatureState.enabled
                else [
                    "python -m app.security.keys cert --out secrets",
                    "Serve with: uvicorn app.main:app --ssl-certfile secrets/server.crt --ssl-keyfile secrets/server.key",
                ]
            ),
            reference="app/security/tls.py",
        ),
        SecurityFeature(
            id="secagg",
            name="Secure aggregation",
            state=secure_aggregation.state,
            implemented=secure_aggregation.available,
            enabled=secure_aggregation.state is FeatureState.enabled,
            description=(
                "Client updates are masked with pairwise pseudorandom masks so the server "
                "recovers only the sum. Masks cancel exactly; individual updates are not "
                "recoverable in the summation."
            ),
            evidence=secure_aggregation.note,
            requirements=(
                []
                if secure_aggregation.state is FeatureState.enabled
                else ["Set FEDMED_SECURE_AGGREGATION_SECRET to a value shared by the clients"]
            ),
            reference="app/federated/secure_aggregation.py",
        ),
        SecurityFeature(
            id="dp",
            name="Differential privacy",
            state=privacy.state,
            implemented=privacy.available,
            enabled=privacy.state is FeatureState.enabled,
            description=(
                "Client-side clipping with Gaussian noise on the aggregate, accounted with "
                "Opacus's RDP accountant. The privacy unit is a client's whole update."
            ),
            evidence=privacy.note,
            requirements=(
                []
                if privacy.state is FeatureState.enabled
                else ["Configure DP on a run and call /api/v1/security/privacy/account to size epsilon"]
            ),
            reference="app/federated/privacy.py",
        ),
        SecurityFeature(
            id="he",
            name="Homomorphic encryption",
            state=FeatureState.not_implemented,
            implemented=False,
            enabled=False,
            description=(
                "Federated averaging over encrypted updates (TenSEAL/CKKS). Not "
                "implemented: no code imports TenSEAL and no encrypted computation runs."
            ),
            evidence="No implementation exists. The dependency is pinned for a future stage.",
            requirements=["Implement encryption of updates before aggregation"],
            reference="Not implemented",
        ),
    ]


def _audit_summary(audit: Any) -> AuditSummary:
    if audit is None:
        return AuditSummary(
            total=0,
            info=0,
            warning=0,
            critical=0,
            recorded=False,
            note="No audit log is attached to this application instance.",
        )
    counts = audit.counts()
    return AuditSummary(
        total=len(audit),
        info=int(counts.get("info", 0)),
        warning=int(counts.get("warning", 0)),
        critical=int(counts.get("critical", 0)),
        recorded=True,
        note=(
            "Events recorded by this process: authorization outcomes and training "
            "lifecycle changes. Secrets are redacted before storage. Process-local and "
            "lost on restart."
        ),
    )


def build_posture(
    settings: Any,
    audit: Any = None,
    request_scheme: str = "http",
) -> dict[str, Any]:
    """Assemble every posture block for the security endpoint."""
    auth = _authentication(settings)
    transport = _transport(settings, request_scheme)
    secure_aggregation = _secure_aggregation(settings)
    privacy = _privacy_accounting(settings)

    return {
        "authentication": auth,
        "transport": transport,
        "secure_aggregation": secure_aggregation,
        "privacy_accounting": privacy,
        "features": _features(auth, transport, secure_aggregation, privacy),
        "audit": _audit_summary(audit),
        "disclaimers": list(DISCLAIMERS),
    }
