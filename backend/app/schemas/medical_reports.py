"""Schemas for medical report analysis.

The output is split into deterministic and generated parts, and they are never
merged: ``findings`` and ``comparison`` come from parsing the text, while
``possibleComplications``, ``clinicalSuggestions`` and the narrative summary come from
the optional AI layer and carry their own availability flag.

``analysisId`` is returned by the analysis endpoint so the result can be fetched again
without re-running the model.
"""

from __future__ import annotations

from datetime import datetime

from .common import CamelModel


class ReportSession(CamelModel):
    session_id: str
    created_at: datetime
    report_count: int
    same_patient_confirmed: bool
    reports: list[dict] = []
    analysis_ids: list[str] = []


class ReportUploadResult(CamelModel):
    session_id: str
    accepted: list[dict] = []
    rejected: list[dict] = []
    report_count: int
    max_reports: int
    note: str


class ExtractedReport(CamelModel):
    report_id: str
    name: str
    status: str
    pages: int | None = None
    characters: int = 0
    headings_found: list[str] = []
    ocr_used: bool = False
    page_texts: list[str] = []
    parsed: dict = {}
    note: str = ""


class ExtractResult(CamelModel):
    session_id: str
    reports: list[ExtractedReport] = []
    ocr: dict = {}
    note: str = ""


class CompareResult(CamelModel):
    session_id: str
    comparison: dict
    report_count: int


class ConfirmSamePatientRequest(CamelModel):
    session_id: str
    confirmed: bool


class AnalyzeRequest(CamelModel):
    session_id: str


class AnalyzeResult(CamelModel):
    """The full analysis. Generated sections state their own availability."""

    analysis_id: str
    session_id: str
    created_at: datetime
    report_count: int
    reports: list[dict] = []
    safety_disclaimer: str

    # Returned before anything else when it is set. When true, the UI must render
    # emergencyNotice above the rest of the analysis.
    urgency: dict
    ai: dict
    summary: dict
    findings: list[dict] = []
    comparison: dict
    diagnosis_status: dict
    clinical_suggestions: dict
    possible_complications: dict
    doctor_ready_summary: dict


class SubsystemHealth(CamelModel):
    """What this feature can actually do right now, stated by the server."""

    available: bool
    ai: dict
    ocr: dict
    max_reports: int
    max_file_bytes: int
    accepted_types: list[str]
    auth_required: bool
    note: str
    #: True when the server has been deliberately configured to serve these routes to
    #: anonymous callers (FEDMED_MEDICAL_REPORTS_PUBLIC_DEMO). The page must say so
    #: prominently rather than quietly accepting uploads from the public.
    public_demo_mode: bool = False
    public_demo_warning: str | None = None
