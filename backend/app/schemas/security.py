"""Schemas for the privacy and security posture.

Nothing in this module describes a working security mechanism. Every mechanism
reports an implementation state of ``planned`` or ``simulated``, and the
encryption status is ``none`` until milestone M5 delivers real code.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from .common import CamelModel


class Severity(str, Enum):
    info = "info"
    warning = "warning"
    critical = "critical"


class MechanismState(str, Enum):
    """Implementation state of a privacy or security mechanism.

    `implemented` means code exists in this repository and is covered by tests. It does
    not mean the mechanism is active, configured, or independently validated.
    """

    planned = "planned"
    simulated = "simulated"
    implemented = "implemented"
    unconfigured = "unconfigured"


class PrivacyMechanism(CamelModel):
    """A privacy or security mechanism and its honest implementation state."""

    id: str
    name: str
    description: str
    state: MechanismState
    implemented: bool
    reference: str


class SecurityEvent(CamelModel):
    """One entry in the audit log. All seeded entries are synthetic."""

    id: str
    timestamp: datetime
    severity: Severity
    category: str
    message: str
    source: str


class PrivacyBudget(CamelModel):
    """Simulated epsilon usage for one node against the configured target."""

    hospital_id: str
    hospital: str
    epsilon: float
    target_epsilon: float
    exceeded: bool


class DifferentialPrivacyConfig(CamelModel):
    """The intended DP configuration. It is not applied to anything."""

    mechanism: str
    noise_multiplier: float
    clipping_norm: float
    target_epsilon: float
    applied: bool


class EncryptionStatus(CamelModel):
    """Transport and at-rest encryption posture."""

    transport: str
    at_rest: str
    note: str


class FeatureState(str, Enum):
    """How a security feature actually stands, in the running process.

    These are distinct on purpose. "Enabled" is not the same as "the switch is on":
    a feature is only `enabled` when it is implemented, configured, AND has been
    exercised. `unverified` exists so the UI never has to guess between the two.
    """

    enabled = "enabled"
    disabled = "disabled"
    unconfigured = "unconfigured"
    unverified = "unverified"
    not_implemented = "not_implemented"


class SecurityFeature(CamelModel):
    """One security feature and the evidence for its state."""

    id: str
    name: str
    state: FeatureState
    implemented: bool
    enabled: bool
    description: str
    #: What was actually observed, so the state is checkable rather than asserted.
    evidence: str
    #: What would be needed to move this to `enabled`.
    requirements: list[str] = []
    #: Where the mechanism lives, or why it does not exist.
    reference: str = ""


class AuthStatus(CamelModel):
    """Authentication posture, read from the running configuration."""

    configured: bool
    state: FeatureState
    scheme: str
    key_count: int
    #: Labels and roles only. Never a key, never a hash, never a hash prefix.
    principals: list[dict]
    required_for_reads: bool
    protected_paths: list[str]
    note: str


class TransportStatus(CamelModel):
    """Transport security posture."""

    configured: bool
    state: FeatureState
    scheme: str
    certificate_path: str | None
    certificate_valid: bool | None
    self_signed: bool | None
    days_remaining: int | None
    subject: str | None
    note: str


class SecureAggregationStatus(CamelModel):
    """Secure aggregation posture."""

    available: bool
    configured: bool
    state: FeatureState
    protocol: str
    primitive: str
    secret_configured: bool
    tolerates_dropout: bool
    threat_model: str
    assumptions: list[str]
    note: str


class PrivacyAccountingStatus(CamelModel):
    """Differential privacy and its accounting."""

    available: bool
    configured: bool
    state: FeatureState
    mechanism: str
    accountant: str
    epsilon: float | None
    delta: float | None
    noise_multiplier: float | None
    clipping_norm: float | None
    sample_rate: float | None
    rounds: int | None
    within_budget: bool | None
    privacy_unit: str
    assumptions: list[str]
    note: str


class AuditSummary(CamelModel):
    """Counts over the real audit log."""

    total: int
    info: int
    warning: int
    critical: int
    #: True when these are recorded events rather than seeded sample rows.
    recorded: bool
    note: str


class AuditCleared(CamelModel):
    """Result of an administrative audit-log clear."""

    removed: int
    cleared_by: str


class SecurityStatus(CamelModel):
    """Full privacy/security payload for the dashboard page.

    The original fields are retained so existing consumers keep working. The added
    `features`, `authentication`, `transport`, `secure_aggregation` and
    `privacy_accounting` blocks describe what is actually configured and running,
    and `events` now comes from the real audit log rather than seeded sample rows.
    """

    mechanisms_implemented: int
    mechanisms_total: int
    encryption: EncryptionStatus
    differential_privacy: DifferentialPrivacyConfig
    secure_aggregation_enabled: bool
    homomorphic_encryption_enabled: bool
    mechanisms: list[PrivacyMechanism]
    budgets: list[PrivacyBudget]
    events: list[SecurityEvent]

    # --- Added in Stage 5: the real, measured posture -----------------------
    features: list[SecurityFeature] = []
    authentication: AuthStatus | None = None
    transport: TransportStatus | None = None
    secure_aggregation: SecureAggregationStatus | None = None
    privacy_accounting: PrivacyAccountingStatus | None = None
    audit: AuditSummary | None = None
    #: Plain statement of what remains unverified or unimplemented.
    disclaimers: list[str] = []
