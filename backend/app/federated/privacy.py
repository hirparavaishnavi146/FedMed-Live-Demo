"""Differential privacy for federated training: clipping, noise, and accounting.

THE MECHANISM

    Client-side fixed clipping plus Gaussian noise on the aggregate, which is what
    Flower's maintained `DifferentialPrivacyClientSideFixedClipping` implements. Each
    client's update is clipped to a bounded norm, and the server adds Gaussian noise
    scaled by `noise_multiplier` before averaging. Without clipping the noise would
    be meaningless, because an unbounded update can carry arbitrary information.

THE ACCOUNTANT

    `opacus.accountants.create_accountant("rdp")` — Opacus 1.6.0, a maintained
    implementation of the Rényi-DP accountant for the subsampled Gaussian mechanism.
    This module does not implement its own accounting, because a hand-rolled
    accountant is the easiest place to accidentally overstate a privacy guarantee.

    Verified behaviour (tests/test_privacy.py): epsilon rises with the number of
    rounds, falls as noise increases, and rises as delta shrinks.

THE PRIVACY UNIT — READ THIS BEFORE QUOTING AN EPSILON

    The unit being protected is **one client's entire update**, not one patient
    record. Flower's DP treats each client as a single example, so epsilon bounds
    what the server learns about a *hospital's contribution*. It says nothing about
    individuals whose scans are in that hospital's dataset. Presenting this as
    patient-level DP would be false.

ASSUMPTIONS THE BOUND DEPENDS ON

    * The subsampled Gaussian mechanism with Poisson sampling at `sample_rate`.
    * Clipping norm applied per client per round, as configured.
    * Independent noise per round, and an honest server that is not colluding.
    * The accountant's RDP-to-(eps, delta) conversion, which is itself an upper bound.
    * The configured number of rounds is the number actually run. Running more rounds
      than configured invalidates the reported epsilon.

WHAT THIS IS NOT

    Not a privacy guarantee for patients, not verified against a second accountant,
    and not a claim that the resulting model is private in any practical sense.
    See docs/SECURITY.md.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any

#: Documented default target, matching the value the UI shows.
DEFAULT_TARGET_DELTA = 1e-5
DEFAULT_NOISE_MULTIPLIER = 1.0
DEFAULT_CLIPPING_NORM = 1.0


class PrivacyConfigurationError(ValueError):
    """The requested privacy configuration is not usable."""

    code = "invalid_privacy_configuration"


@dataclass(frozen=True)
class PrivacyConfig:
    """Every parameter the privacy bound depends on.

    Recorded with a run so an epsilon can be attributed to the configuration that
    produced it.
    """

    enabled: bool = False
    mechanism: str = "gaussian"
    accountant: str = "rdp"
    noise_multiplier: float = DEFAULT_NOISE_MULTIPLIER
    clipping_norm: float = DEFAULT_CLIPPING_NORM
    #: Fraction of clients sampled per round. None means all clients participate,
    #: which sets sample_rate = 1.0.
    sample_rate: float | None = None
    num_rounds: int = 1
    target_delta: float = DEFAULT_TARGET_DELTA
    #: Optional budget to check the spent epsilon against.
    target_epsilon: float | None = None

    def __post_init__(self) -> None:
        if self.mechanism != "gaussian":
            raise PrivacyConfigurationError(
                f"Only the Gaussian mechanism is implemented; received {self.mechanism!r}."
            )
        if self.accountant != "rdp":
            raise PrivacyConfigurationError(
                f"Only the RDP accountant is implemented; received {self.accountant!r}."
            )
        if self.noise_multiplier <= 0:
            raise PrivacyConfigurationError(
                f"noise_multiplier must be positive, got {self.noise_multiplier}."
            )
        if self.clipping_norm <= 0:
            raise PrivacyConfigurationError(
                f"clipping_norm must be positive, got {self.clipping_norm}."
            )
        if self.sample_rate is not None and not 0.0 < self.sample_rate <= 1.0:
            raise PrivacyConfigurationError(
                f"sample_rate must be in (0, 1], got {self.sample_rate}."
            )
        if self.num_rounds < 1:
            raise PrivacyConfigurationError(
                f"num_rounds must be at least 1, got {self.num_rounds}."
            )
        if not 0.0 < self.target_delta < 1.0:
            raise PrivacyConfigurationError(
                f"target_delta must be in (0, 1), got {self.target_delta}."
            )
        if self.target_epsilon is not None and self.target_epsilon <= 0:
            raise PrivacyConfigurationError(
                f"target_epsilon must be positive when set, got {self.target_epsilon}."
            )

    @property
    def effective_sample_rate(self) -> float:
        """All clients participating is a sampling rate of 1.0, stated explicitly."""
        return 1.0 if self.sample_rate is None else float(self.sample_rate)

    def as_dict(self) -> dict[str, Any]:
        payload = dataclasses.asdict(self)
        payload["effectiveSampleRate"] = self.effective_sample_rate
        payload["privacyUnit"] = "one client's entire update (client-level DP)"
        return payload

    def describe(self) -> str:
        return (
            f"{self.mechanism}, noise={self.noise_multiplier}, clip={self.clipping_norm}, "
            f"q={self.effective_sample_rate}, rounds={self.num_rounds}, "
            f"delta={self.target_delta}"
        )


@dataclass
class PrivacyReport:
    """The accounted privacy expenditure."""

    config: PrivacyConfig
    epsilon: float | None
    delta: float
    steps: int
    within_budget: bool | None
    accountant: str
    available: bool = True
    unavailable_reason: str | None = None
    assumptions: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "epsilon": self.epsilon,
            "delta": self.delta,
            "steps": self.steps,
            "targetEpsilon": self.config.target_epsilon,
            "withinBudget": self.within_budget,
            "accountant": self.accountant,
            "mechanism": "subsampled Gaussian mechanism",
            "available": self.available,
            "unavailableReason": self.unavailable_reason,
            "privacyUnit": "one client's entire update (client-level DP)",
            "config": self.config.as_dict(),
            "configDescription": self.config.describe(),
            "assumptions": self.assumptions,
        }

    def describe(self) -> str:
        if not self.available or self.epsilon is None:
            return f"Not accounted: {self.unavailable_reason}"
        return (
            f"epsilon = {self.epsilon:.4f}, delta = {self.delta:g} after {self.steps} "
            f"round(s) at noise {self.config.noise_multiplier}, q="
            f"{self.config.effective_sample_rate}"
        )


def _accountant():
    """Opacus's RDP accountant, imported lazily so the module loads without it."""
    from opacus.accountants import create_accountant

    return create_accountant(mechanism="rdp")


def account_privacy(config: PrivacyConfig) -> PrivacyReport:
    """Account the epsilon spent by this configuration.

    Returns a report with ``available: False`` and a reason when accounting is not
    possible, rather than a zero or a guess. An epsilon that cannot be computed must
    not be displayed as if it had been.
    """
    if not config.enabled:
        return PrivacyReport(
            config=config,
            epsilon=None,
            delta=config.target_delta,
            steps=0,
            within_budget=None,
            accountant=config.accountant,
            available=False,
            unavailable_reason=(
                "Differential privacy is disabled, so no epsilon has been spent. The "
                "value is absent rather than zero."
            ),
            assumptions=[],
        )

    try:
        accountant = _accountant()
        for _ in range(config.num_rounds):
            accountant.step(
                noise_multiplier=config.noise_multiplier,
                sample_rate=config.effective_sample_rate,
            )
        epsilon = float(accountant.get_epsilon(delta=config.target_delta))
    except ImportError:
        return PrivacyReport(
            config=config,
            epsilon=None,
            delta=config.target_delta,
            steps=config.num_rounds,
            within_budget=None,
            accountant=config.accountant,
            available=False,
            unavailable_reason=(
                "The privacy accountant (opacus) is not installed, so epsilon cannot "
                "be computed. No value is reported rather than an estimate."
            ),
            assumptions=[],
        )
    except Exception as exc:  # noqa: BLE001 - any accounting failure means "no value"
        return PrivacyReport(
            config=config,
            epsilon=None,
            delta=config.target_delta,
            steps=config.num_rounds,
            within_budget=None,
            accountant=config.accountant,
            available=False,
            unavailable_reason=f"Privacy accounting failed: {type(exc).__name__}: {exc}",
            assumptions=[],
        )

    within = (
        None if config.target_epsilon is None else epsilon <= config.target_epsilon
    )

    return PrivacyReport(
        config=config,
        epsilon=epsilon,
        delta=config.target_delta,
        steps=config.num_rounds,
        within_budget=within,
        accountant=config.accountant,
        available=True,
        assumptions=[
            "The unit of privacy is one client's whole update, not one patient record.",
            "Subsampled Gaussian mechanism, Poisson sampling at the stated rate.",
            "Per-client clipping at the stated norm, applied every round.",
            "Noise is independent across rounds; the server is honest and does not collude.",
            "Exactly the configured number of rounds was run. More rounds would mean a "
            "larger epsilon than reported.",
            "The RDP-to-(epsilon, delta) conversion is an upper bound, not an exact value.",
            "This bound covers what the server learns about a hospital's contribution. "
            "It makes no claim about individuals inside that hospital's dataset.",
        ],
    )


def build_dp_strategy(base_strategy: Any, config: PrivacyConfig, num_sampled_clients: int) -> Any:
    """Wrap a Flower strategy with client-side fixed clipping and Gaussian noise.

    Uses `DifferentialPrivacyClientSideFixedClipping` from Flower 1.39. The
    `DPFedAvgFixed`/`DPFedAvgAdaptive` wrappers are also present in this version but
    are documented as deprecated, so they are deliberately not used.
    """
    if not config.enabled:
        raise PrivacyConfigurationError(
            "Refusing to build a DP strategy while privacy is disabled. Enabling "
            "privacy is an explicit decision, not a default."
        )
    if num_sampled_clients < 1:
        raise PrivacyConfigurationError(
            f"num_sampled_clients must be at least 1, got {num_sampled_clients}."
        )

    from flwr.server.strategy import DifferentialPrivacyClientSideFixedClipping

    return DifferentialPrivacyClientSideFixedClipping(
        strategy=base_strategy,
        noise_multiplier=config.noise_multiplier,
        clipping_norm=config.clipping_norm,
        num_sampled_clients=num_sampled_clients,
    )


def privacy_capabilities() -> dict[str, Any]:
    """What the installed libraries actually provide, discovered not assumed."""
    capabilities: dict[str, Any] = {
        "mechanisms": ["gaussian"],
        "accountants": ["rdp"],
        "privacyUnit": "client",
    }

    try:
        import opacus

        capabilities["opacus"] = opacus.__version__
        capabilities["accountantAvailable"] = True
    except ImportError:
        capabilities["opacus"] = None
        capabilities["accountantAvailable"] = False

    try:
        from flwr.server import strategy as flower_strategy

        maintained = [
            name
            for name in (
                "DifferentialPrivacyClientSideFixedClipping",
                "DifferentialPrivacyClientSideAdaptiveClipping",
                "DifferentialPrivacyServerSideFixedClipping",
                "DifferentialPrivacyServerSideAdaptiveClipping",
            )
            if hasattr(flower_strategy, name)
        ]
        deprecated = [
            name
            for name in ("DPFedAvgFixed", "DPFedAvgAdaptive")
            if hasattr(flower_strategy, name)
        ]
        capabilities["flowerClippingWrappers"] = maintained
        capabilities["flowerDeprecatedWrappers"] = deprecated
        capabilities["using"] = (
            "DifferentialPrivacyClientSideFixedClipping"
            if "DifferentialPrivacyClientSideFixedClipping" in maintained
            else None
        )
    except ImportError:
        capabilities["flowerClippingWrappers"] = []
        capabilities["flowerDeprecatedWrappers"] = []
        capabilities["using"] = None

    return capabilities
