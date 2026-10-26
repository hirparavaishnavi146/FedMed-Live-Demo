"""Schemas for case intake.

Every response type here carries `interpretationSupported`, because the honest
answer is the same for all of them: this project cannot interpret a scan or a
report, and the field makes that machine-readable rather than leaving it to a note
somebody might not render.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import Field

from .common import CamelModel


class CaseCapabilities(CamelModel):
    """What can be done with each input type. Served so the UI cannot overstate it."""

    mri_volume: dict
    report_pdf: dict
    report_text: dict
    # Explicit alias. The camelCase generator turns "image2d" into "image2D", which
    # would not match the "image2d" key app/cases/intake.py returns for this same
    # entry inside a case summary — one concept with two spellings, and the frontend
    # would have to handle both. The alias pins the wire name to one value.
    image2d: dict = Field(alias="image2d")
    dicom: dict


class CaseFile(CamelModel):
    name: str
    kind: str
    size_bytes: int
    method: str
    note: str
    analysed: bool
    analysis: dict


class CaseSummary(CamelModel):
    case_id: str
    name: str
    created_at: datetime
    file_count: int
    analysed_count: int
    mri_volume_count: int
    mask_count: int
    report_count: int
    # Explicit alias for the same reason as CaseCapabilities.image2d: the generator
    # would produce "image2DCount", which would not match the "image2dCount" key
    # that app/cases/intake.py returns.
    image2d_count: int = Field(alias="image2dCount")
    files: list[CaseFile]

    # False for every case. Stated as a field, not only as prose.
    interpretation_supported: bool
    interpretation_reason: str
    notice: str
    reasons: list[str]
    capabilities: dict
    staged: bool


class CaseList(CamelModel):
    cases: list[CaseSummary] = []
    total: int = 0


class ValidateCaseDirectoryRequest(CamelModel):
    path: str
    name: str | None = None
