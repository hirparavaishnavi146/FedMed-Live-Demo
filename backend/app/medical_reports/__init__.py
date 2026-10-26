"""Medical report analysis and clinical decision support.

Kept separate from the federated MRI segmentation pipeline on purpose: the two are
different tasks. Nothing here claims that the federated model can interpret a report
or an X-ray. The connection point is the API, not shared internals.

The package splits into four parts, so a reader can tell measured facts from
generated text:

* ``parsing``        deterministic extraction — quotes and numbers, with provenance
* ``ocr``            optional OCR, honest about being unavailable
* ``analysis_ai``    optional AI layer, refuses rather than inventing
* ``analysis``       orchestration, comparison, urgency, assembly
"""

from . import analysis, analysis_ai, ocr, parsing
from .analysis import (
    MAX_REPORTS,
    SAFETY_DISCLAIMER,
    ReportRecord,
    Session,
    analyse,
    compare,
    diagnosis_status,
    extract_report,
    findings,
    registry,
    urgency,
)
from .parsing import ParsedReport, abnormal_labs, parse_report

__all__ = [
    "MAX_REPORTS",
    "SAFETY_DISCLAIMER",
    "ParsedReport",
    "ReportRecord",
    "Session",
    "abnormal_labs",
    "analysis",
    "analysis_ai",
    "analyse",
    "compare",
    "diagnosis_status",
    "extract_report",
    "findings",
    "ocr",
    "parse_report",
    "parsing",
    "registry",
    "urgency",
]
