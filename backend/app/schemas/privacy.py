"""Schemas for differential-privacy accounting.

Separate from `schemas.security` so the accounting contract is easy to find: an
epsilon is only meaningful together with the configuration and assumptions it was
computed from.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from .common import CamelModel


class PrivacyAccountRequest(CamelModel):
    """A differential-privacy configuration to account."""

    enabled: bool = Field(
        default=True,
        description="Must be true to account anything. Disabled returns no epsilon.",
    )
    noise_multiplier: float = Field(default=1.0, gt=0, le=1000)
    clipping_norm: float = Field(default=1.0, gt=0, le=1e6)
    sample_rate: float | None = Field(
        default=None,
        gt=0,
        le=1.0,
        description="Fraction of clients sampled per round. Omit when all participate.",
    )
    num_rounds: int = Field(default=1, ge=1, le=10000)
    target_delta: float = Field(default=1e-5, gt=0, lt=1)
    target_epsilon: float | None = Field(default=None, gt=0)


class PrivacyAccountResponse(CamelModel):
    """The accounted privacy expenditure, with its assumptions."""

    epsilon: float | None
    delta: float | None
    steps: int
    target_epsilon: float | None = None
    within_budget: bool | None = None
    accountant: str
    mechanism: str
    available: bool
    unavailable_reason: str | None = None
    privacy_unit: str
    privacy_unit_short: str = ""
    config: dict[str, Any] = {}
    config_description: str = ""
    assumptions: list[str] = []
