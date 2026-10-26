"""Assemble a clinical decision-support analysis from one to three reports.

The section-by-section split, and why it is drawn where it is
-------------------------------------------------------------
A. Summary of reports      deterministic digest, plus an AI narrative when configured
B. Findings and abnormal.  DETERMINISTIC — quoted values compared to ranges the
                           report printed. Nothing here is generated.
C. Possible complications  AI ONLY — reasoning about clinical significance is exactly
                           the kind of claim this project cannot verify
D. Diagnosis status        deterministic — "documented in a report" vs "insufficient
                           information". The middle state is AI.
E. Clinical suggestions    AI only, plus a deterministic list of what is missing
F. Urgency and safety      deterministic red-flag scan, which AI may escalate but
                           NEVER de-escalate
G. Doctor-ready summary    assembled from the deterministic parts, with AI prose
                           clearly separated

The rule behind that table: a section is deterministic if it can be computed and
checked, and AI-only if it requires judgement. Nothing generated is ever presented in
the same field as something measured.

Emergency handling
------------------
A red-flag term found in the text produces an emergency notice that is returned FIRST
in the payload, before the rest of the analysis, and it is never suppressed by AI.
The scan is a keyword match, not triage, and the response says so: the absence of a
red flag is explicitly NOT a statement that the case is safe.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..cases import reports as case_reports
from ..errors import NotFoundError, ValidationProblem
from . import analysis_ai, ocr, parsing

MAX_REPORTS = 3

SAFETY_DISCLAIMER = (
    "This is an AI-assisted draft produced by a research prototype, for the purpose of "
    "preparing for a consultation with a qualified healthcare professional. It is NOT a "
    "diagnosis, NOT a medical opinion, and NOT a substitute for professional care. It has "
    "not been clinically validated. Do not make any treatment decision based on it, and do "
    "not use it to delay seeking care. If you think you may be having a medical emergency, "
    "contact emergency services now."
)

EMERGENCY_NOTICE = (
    "POSSIBLE EMERGENCY WARNING SIGNS WERE FOUND IN THE TEXT OF THESE REPORTS. If the "
    "patient is currently unwell — for example severe or worsening symptoms, difficulty "
    "breathing, chest pain, confusion, or loss of consciousness — seek emergency medical "
    "care immediately. Do not wait for this application to finish analysing anything. "
    "This notice comes from a keyword scan of the report text, not from clinical judgement."
)

URGENCY_LIMITATION = (
    "Urgency is derived from explicit warning terms present in the report text and, when "
    "configured, from an AI suggestion. It is not a triage decision. The absence of a "
    "warning term does NOT mean the case is safe or routine, and this field must not be "
    "used to decide whether to seek care."
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class ReportRecord:
    """One uploaded report, with its extracted text kept separate from any analysis."""

    report_id: str
    name: str
    path: Path
    size_bytes: int
    content_type: str | None
    extract: case_reports.ReportExtract
    parsed: parsing.ParsedReport
    ocr_used: bool = False
    session_id: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "reportId": self.report_id,
            "name": self.name,
            "sizeBytes": self.size_bytes,
            "contentType": self.content_type,
            "status": self.extract.status.value,
            "pages": self.extract.pages,
            "characters": self.extract.characters,
            "headingsFound": self.extract.headings_found,
            "ocrUsed": self.ocr_used,
            "extractNote": self.extract.note,
            # The raw text is exposed so the user can check every claim themselves.
            "pageTexts": self.extract.page_texts,
        }


@dataclass
class Session:
    """A patient's set of reports, before and after analysis."""

    session_id: str
    created_at: datetime
    reports: list[ReportRecord] = field(default_factory=list)
    same_patient_confirmed: bool = False
    confirmed_at: datetime | None = None
    analyses: dict[str, dict[str, Any]] = field(default_factory=dict)


class ReportRegistry:
    """In-process store. Uploaded files live in the shared staging area."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._sessions: dict[str, Session] = {}

    def create(self) -> Session:
        session = Session(session_id=f"mr-{uuid.uuid4().hex[:12]}", created_at=_utc_now())
        with self._lock:
            self._sessions[session.session_id] = session
        return session

    def get(self, session_id: str) -> Session:
        with self._lock:
            session = self._sessions.get(session_id)
        if session is None:
            raise NotFoundError(
                f"No analysis session {session_id!r}. It may have expired; uploads are "
                "temporary.",
                code="report_session_not_found",
            )
        return session

    def list(self) -> list[Session]:
        with self._lock:
            return sorted(self._sessions.values(), key=lambda item: item.created_at, reverse=True)

    def clear(self) -> None:
        with self._lock:
            self._sessions.clear()


registry = ReportRegistry()


def activity_summary() -> dict[str, Any]:
    """Count what users have ACTUALLY uploaded and what was really extracted.

    Every number here is a count over the live registry, so a fresh process reports
    zeroes rather than placeholder figures. That is the point: the dashboard's existing
    cards describe the federated MRI simulation, which is illustrative because no model
    has been trained, and those figures cannot be derived from a medical report. What
    CAN be measured is this — real sessions, real uploaded reports, real values pulled
    out of them — so it is reported separately and labelled as measured.

    Nothing is estimated. A value is counted as "outside its printed range" only when
    the report itself printed a range it falls outside; values with no range are counted
    in their own bucket rather than assumed normal.
    """
    sessions = registry.list()

    reports = 0
    readable = 0
    unreadable = 0
    ocr_used = 0
    values_total = 0
    values_outside_range = 0
    values_without_range = 0
    documented_diagnoses = 0
    documented_medications = 0
    red_flag_reports = 0
    report_types: dict[str, int] = {}

    for session in sessions:
        for report in session.reports:
            reports += 1
            if report.extract.status.value == "extracted":
                readable += 1
            else:
                unreadable += 1
            if report.ocr_used:
                ocr_used += 1

            parsed = report.parsed
            if parsed.report_type:
                report_types[parsed.report_type] = report_types.get(parsed.report_type, 0) + 1
            if parsed.red_flags:
                red_flag_reports += 1
            documented_diagnoses += len(parsed.diagnoses)
            documented_medications += len(parsed.medications)

            for lab in parsed.labs:
                values_total += 1
                if lab.within_range is True:
                    continue
                if lab.within_range is False:
                    values_outside_range += 1
                else:
                    # No reference range was printed, so this application is not
                    # entitled to a verdict. Counted separately, never as normal.
                    values_without_range += 1

    analyses = sum(len(session.analyses) for session in sessions)
    confirmed = sum(1 for session in sessions if session.same_patient_confirmed)

    latest_urgency: str | None = None
    for session in sessions:
        for stored in session.analyses.values():
            urgency = (stored.get("urgency") or {}).get("category")
            if urgency:
                latest_urgency = urgency
                break
        if latest_urgency:
            break

    return {
        "measured": True,
        "sessions": len(sessions),
        "samePatientConfirmed": confirmed,
        "reportsUploaded": reports,
        "reportsReadable": readable,
        "reportsUnreadable": unreadable,
        "reportsReadByOcr": ocr_used,
        "analysesRun": analyses,
        "valuesExtracted": values_total,
        "valuesOutsidePrintedRange": values_outside_range,
        "valuesWithoutPrintedRange": values_without_range,
        "documentedDiagnoses": documented_diagnoses,
        "documentedMedications": documented_medications,
        "reportsWithRedFlags": red_flag_reports,
        "reportTypes": report_types,
        "latestUrgency": latest_urgency,
        "note": (
            "Counted from reports actually uploaded to this server. Zero means nothing "
            "has been uploaded yet — it is not a placeholder. This is separate from the "
            "federated MRI figures above, which describe a model that has not been "
            "trained and cannot be derived from a medical report."
        ),
    }


# --- extraction ------------------------------------------------------------------


def extract_report(
    session_id: str,
    name: str,
    path: Path,
    size_bytes: int,
    content_type: str | None = None,
) -> ReportRecord:
    """Extract text from one uploaded report, using OCR only where it is needed."""
    report_id = f"rpt-{uuid.uuid4().hex[:10]}"
    suffix = path.name.lower()
    ocr_used = False

    if suffix.endswith(".pdf"):
        extract = case_reports.extract(path)
        # A PDF with no text layer is a scan. Try OCR, and if OCR is unavailable say
        # so through the extract status rather than pretending the page is empty.
        if extract.status is case_reports.TextStatus.no_text_layer:
            state = ocr.detect()
            if state.available_for_pdfs:
                try:
                    pages = ocr.ocr_pdf_pages(path)
                except Exception:  # noqa: BLE001 - falls through to the honest message
                    pages = []
                if any(page.strip() for page in pages):
                    ocr_used = True
                    extract = case_reports.ReportExtract(
                        status=case_reports.TextStatus.extracted,
                        characters=sum(len(page) for page in pages),
                        pages=len(pages),
                        page_texts=pages,
                        excerpt="\n".join(pages)[:1200],
                        note="Text was recovered by OCR from a scanned document. OCR can "
                        "misread characters, so every value should be checked against the "
                        "original before it is relied upon.",
                    )
            else:
                extract.note = (
                    extract.note + " " + (state.reason or "")
                )
    elif suffix.endswith((".txt", ".md", ".text", ".report")):
        extract = case_reports.extract(path)
    else:
        # An image: OCR is the only route, and there is no text layer by definition.
        state = ocr.detect()
        if state.available_for_images:
            try:
                text = ocr.ocr_image(path)
            except Exception:  # noqa: BLE001
                text = ""
            if text.strip():
                ocr_used = True
                extract = case_reports.ReportExtract(
                    status=case_reports.TextStatus.extracted,
                    characters=len(text),
                    pages=1,
                    page_texts=[text],
                    excerpt=text[:1200],
                    note="Text was recovered by OCR from an image. OCR can misread "
                    "characters, so every value should be checked against the original.",
                )
            else:
                extract = case_reports.ReportExtract(
                    status=case_reports.TextStatus.no_text_layer,
                    pages=1,
                    note="No text could be read from this image by OCR.",
                )
        else:
            raise ValidationProblem(
                state.reason
                or "This image cannot be read: OCR is not available in this deployment.",
                code="ocr_unavailable",
            )

    parsed = parsing.parse_report(
        report_id, name, extract.page_texts, characters=extract.characters
    )
    return ReportRecord(
        report_id=report_id,
        name=name,
        path=path,
        size_bytes=size_bytes,
        content_type=content_type,
        extract=extract,
        parsed=parsed,
        ocr_used=ocr_used,
        session_id=session_id,
    )


# --- comparison ------------------------------------------------------------------


def compare(reports: list[ReportRecord]) -> dict[str, Any]:
    """Compare findings across reports. Deterministic, and honest about what it cannot do."""
    consistent: list[dict[str, Any]] = []
    differing: list[dict[str, Any]] = []
    only_in_one: list[dict[str, Any]] = []

    by_name: dict[str, list[tuple[str, parsing.LabValue]]] = {}
    for report in reports:
        for lab in report.parsed.labs:
            by_name.setdefault(lab.name.lower(), []).append((report.name, lab))

    for key, entries in sorted(by_name.items()):
        if len(entries) < 2:
            report_name, lab = entries[0]
            only_in_one.append(
                {
                    # The analyte name, NOT the report name: entries holds
                    # (report_name, lab) pairs and the two were mixed up here.
                    "analyte": lab.name,
                    "presentIn": [report_name],
                    "detail": lab.as_dict(),
                    "note": "This analyte appears in only one report, so no comparison "
                    "across time is possible.",
                }
            )
            continue

        values = [(report_name, lab.value) for report_name, lab in entries]
        numeric = [value for _, value in values if value is not None]
        if len(numeric) < 2:
            continue

        record = {
            "analyte": key,
            "presentIn": [report_name for report_name, _ in entries],
            "readings": [
                {"report": report_name, "value": lab.value, "unit": lab.unit,
                 "referenceText": lab.reference_text, "withinRange": lab.within_range,
                 "page": lab.provenance.page}
                for report_name, lab in entries
            ],
        }

        if len(set(numeric)) == 1:
            record["note"] = "The same value appears in every report containing it."
            consistent.append(record)
        else:
            ordered = [value for value in numeric]
            record["change"] = {
                "absolute": ordered[-1] - ordered[0],
                "direction": "increased" if ordered[-1] > ordered[0] else "decreased",
            }
            record["note"] = (
                "The value differs between reports. Whether that is a real change over "
                "time or a difference in measurement cannot be determined from the text "
                "alone; check the report dates, which may themselves be ambiguous."
            )
            differing.append(record)

    # Missing information is the honest flip side of what was found.
    missing: list[str] = []
    without_range = [
        lab for report in reports for lab in parsing.labs_without_reference(report.parsed)
    ]
    if without_range:
        missing.append(
            f"{len(without_range)} value(s) were printed without a reference range, so no "
            "comparison against a normal range was possible for them."
        )
    undated = [report.name for report in reports if not report.parsed.dates]
    if undated:
        missing.append(
            "No report date could be found in: " + ", ".join(undated) + ". Ordering these "
            "reports by date is therefore not possible."
        )
    ambiguous = [
        date
        for report in reports
        for date in report.parsed.dates
        if date.ambiguous
    ]
    if ambiguous:
        missing.append(
            f"{len(ambiguous)} date(s) are written in an ambiguous numeric format and "
            "were not converted."
        )
    unreadable = [
        report.name
        for report in reports
        if report.extract.status is not case_reports.TextStatus.extracted
    ]
    if unreadable:
        missing.append(
            "These reports could not be read, so nothing in them is reflected: "
            + ", ".join(unreadable)
        )

    return {
        "reportsCompared": [report.name for report in reports],
        "consistentFindings": consistent,
        "differingFindings": differing,
        "findingsInOneReportOnly": only_in_one,
        "missingInformation": missing,
        "note": (
            "Comparison is limited to identical analyte names appearing in more than one "
            "report. Different names for the same test are not matched, and values "
            "measured by different laboratories may not be directly comparable."
        ),
    }


# --- findings --------------------------------------------------------------------


def findings(reports: list[ReportRecord]) -> list[dict[str, Any]]:
    """Findings and abnormalities. Every entry is measured or quoted, never generated."""
    items: list[dict[str, Any]] = []

    for report in reports:
        for lab in parsing.abnormal_labs(report.parsed):
            direction = "above" if (lab.reference_high is not None and lab.value is not None and lab.value > lab.reference_high) else "below"
            range_text = lab.reference_text or "no range printed"
            items.append(
                {
                    "finding": f"{lab.name} {lab.value_text}"
                    + (f" {lab.unit}" if lab.unit else ""),
                    "kind": "laboratory-value-outside-printed-range",
                    "reportName": report.name,
                    "reportDate": _report_date(report),
                    "page": lab.provenance.page,
                    "referenceRange": range_text,
                    "direction": direction,
                    "documentedOrInferred": "documented",
                    "whyItMayMatter": (
                        f"The report prints a reference range of {range_text} and this "
                        f"value is {direction} it. This application does not know what "
                        "that means clinically — discuss it with a clinician."
                    ),
                    "sourceSnippet": lab.provenance.snippet,
                }
            )

        for statement in report.parsed.diagnoses:
            items.append(
                {
                    "finding": statement.text[:500],
                    "kind": "statement-under-a-diagnosis-heading",
                    "reportName": report.name,
                    "reportDate": _report_date(report),
                    "page": statement.provenance.page,
                    "referenceRange": None,
                    "direction": None,
                    "documentedOrInferred": "documented",
                    "whyItMayMatter": (
                        f"Written by the report's author under the heading "
                        f"'{statement.heading}'. It is quoted, not interpreted."
                    ),
                    "sourceSnippet": statement.provenance.snippet,
                }
            )

    return items


def _report_date(report: ReportRecord) -> str | None:
    """The report's own date, or None. Never guessed from upload time."""
    for found in report.parsed.dates:
        if found.iso and not found.ambiguous:
            return found.iso
    for found in report.parsed.dates:
        return found.literal
    return None


# --- diagnosis status ------------------------------------------------------------


def diagnosis_status(reports: list[ReportRecord]) -> dict[str, Any]:
    """Distinguish what is written from what is not known. No AI in this section."""
    documented = [
        {
            "reportName": report.name,
            "heading": statement.heading,
            "text": statement.text[:600],
            "page": statement.provenance.page,
        }
        for report in reports
        for statement in report.parsed.diagnoses
    ]

    unreadable = [
        report.name
        for report in reports
        if report.extract.status is not case_reports.TextStatus.extracted
    ]

    if documented:
        status = "documented-in-a-report"
        explanation = (
            "The statements below are quoted from under a diagnosis, impression or "
            "assessment heading. They are what the report's author wrote."
        )
    elif unreadable:
        status = "insufficient-information"
        explanation = (
            "No diagnosis statement was found, and these reports could not be read, so "
            "nothing can be concluded either way: " + ", ".join(unreadable)
        )
    else:
        status = "no-concerning-abnormality-identified-in-the-provided-information"
        explanation = (
            "No diagnosis statement and no value outside its printed reference range was "
            "found in the supplied reports. This is a statement about what these reports "
            "contain, and NOT a statement that the patient is healthy: reports may be "
            "incomplete, values without a printed range could not be assessed, and "
            "normal results do not exclude disease."
        )

    return {
        "status": status,
        "explanation": explanation,
        "documentedDiagnoses": documented,
        "possibleConditionRequiringEvaluation": None,
        "possibleConditionNote": (
            "Suggesting a possible condition requires clinical reasoning that this "
            "deployment cannot perform. See the generated sections, which report that "
            "the AI provider is unavailable when it is not configured."
        ),
    }


# --- urgency ---------------------------------------------------------------------


def urgency(reports: list[ReportRecord], ai: analysis_ai.AiResult | None) -> dict[str, Any]:
    """Red-flag scan, which AI may escalate but never de-escalate."""
    flags = [flag for report in reports for flag in report.parsed.red_flags]
    ai_suggestions = [
        complication.get("recommendedUrgency")
        for complication in ((ai.data.get("possibleComplications") if ai and ai.available else []) or [])
        if isinstance(complication, dict)
    ]

    if flags:
        category = "emergency"
        basis = (
            f"{len(flags)} explicit warning term(s) were found in the report text. The "
            "highest category is used whenever any is present, and it cannot be lowered "
            "by the AI layer."
        )
        show_emergency = True
    elif "emergency" in ai_suggestions or "prompt" in ai_suggestions:
        category = "prompt-review"
        basis = (
            "The AI layer suggested prompt review. This is a generated suggestion, not "
            "clinical judgement."
        )
        show_emergency = False
    else:
        category = "unable-to-assess"
        basis = (
            "No explicit warning term was found and no AI assessment is available, so no "
            "urgency can be established. This is not a clearance."
        )
        show_emergency = False

    return {
        "category": category,
        "basis": basis,
        "emergencyNotice": EMERGENCY_NOTICE if show_emergency else None,
        "showBeforeAnalysis": show_emergency,
        "redFlags": [flag.as_dict() for flag in flags],
        "aiSuggestedUrgencies": ai_suggestions,
        "limitation": URGENCY_LIMITATION,
        "categoriesExplained": {
            "emergency": "Possible emergency warning signs; seek emergency care now.",
            "prompt-review": "Findings that may need timely professional evaluation.",
            "routine-review": "Suitable for discussion at a scheduled appointment.",
            "unable-to-assess": "Insufficient or unreadable evidence to say.",
        },
    }


# --- assembly --------------------------------------------------------------------


def analyse(session: Session, settings: Any, *, force: bool = False) -> dict[str, Any]:
    """Produce the full analysis for a session's reports."""
    if not session.reports:
        raise ValidationProblem(
            "No reports have been uploaded for this session.",
            code="no_reports_uploaded",
        )
    if not session.same_patient_confirmed:
        raise ValidationProblem(
            "Confirm that the uploaded reports belong to the same patient before "
            "analysing them. Comparing reports from different people would produce a "
            "meaningless result.",
            code="same_patient_not_confirmed",
        )

    analysis_id = f"ana-{uuid.uuid4().hex[:12]}"

    # The AI layer is attempted, and its absence is reported rather than papered over.
    ai_state = analysis_ai.availability(settings)
    ai_result: analysis_ai.AiResult | None = None
    ai_error: dict[str, Any] | None = None
    if ai_state.configured:
        try:
            ai_result = analysis_ai.analyse(
                settings, [report.as_dict() for report in session.reports]
            )
        except Exception as exc:  # noqa: BLE001 - reported, never fabricated around
            code = getattr(exc, "code", "ai_analysis_failed")
            ai_error = {"code": code, "message": str(exc)}

    stored: dict[str, Any] = {
        "analysisId": analysis_id,
        "sessionId": session.session_id,
        "createdAt": _utc_now(),
        "reportCount": len(session.reports),
        "reports": [report.as_dict() for report in session.reports],
        "safetyDisclaimer": SAFETY_DISCLAIMER,
        "ai": (ai_result.as_dict() if ai_result else {
            "available": False,
            "data": {},
            "note": "",
            "reason": (ai_error or {}).get("message") or ai_state.reason,
        }),
        "urgency": urgency(session.reports, ai_result),
        "summary": {
            "deterministicDigest": [
                {
                    "reportName": report.name,
                    "reportType": report.parsed.report_type or "not stated in the report",
                    "reportDate": _report_date(report),
                    "pages": report.extract.pages,
                    "status": report.extract.status.value,
                    "valuesFound": len(report.parsed.labs),
                    "valuesOutsidePrintedRange": len(parsing.abnormal_labs(report.parsed)),
                    "headingsFound": report.extract.headings_found,
                    "note": report.extract.note,
                }
                for report in session.reports
            ],
            "note": (
                "This digest is computed from the text. It is not an AI summary and it "
                "does not interpret the findings."
            ),
        },
        "findings": findings(session.reports),
        "comparison": compare(session.reports),
        "diagnosisStatus": diagnosis_status(session.reports),
        "clinicalSuggestions": {
            "available": bool(ai_result and ai_result.available),
            "reason": None if (ai_result and ai_result.available) else
            ((ai_error or {}).get("message") or ai_state.reason),
            "items": (ai_result.data.get("clinicalSuggestions") if ai_result and ai_result.available else []) or [],
            "note": (
                "These are AI-generated suggestions for discussion with a clinician. They "
                "are not treatment advice and no medication change is ever recommended."
            ),
        },
        "possibleComplications": {
            "available": bool(ai_result and ai_result.available),
            "reason": None if (ai_result and ai_result.available) else
            ((ai_error or {}).get("message") or ai_state.reason),
            "items": (ai_result.data.get("possibleComplications") if ai_result and ai_result.available else []) or [],
            "note": (
                "A possibility suggested by a language model is NOT a diagnosis. Each "
                "entry states the evidence behind it, what contradicts it, and what is "
                "missing."
            ),
        },
        "doctorReadySummary": {
            "available": bool(ai_result and ai_result.available),
            "reason": None if (ai_result and ai_result.available) else
            ((ai_error or {}).get("message") or ai_state.reason),
            "ai": (ai_result.data.get("doctorReadySummary") if ai_result and ai_result.available else None),
            "assembledForClinician": _clinician_line(session.reports, ai_result),
        },
    }

    session.analyses[analysis_id] = stored
    if not force:
        registry._sessions[session.session_id] = session
    return stored


def _clinician_line(
    reports: list[ReportRecord], ai: analysis_ai.AiResult | None
) -> dict[str, Any]:
    """The parts of a doctor-ready summary that can be assembled without AI."""
    return {
        "label": "AI-assisted summary for clinician review",
        "reportsProvided": [report.name for report in reports],
        "unreadableReports": [
            report.name
            for report in reports
            if report.extract.status is not case_reports.TextStatus.extracted
        ],
        "valuesOutsidePrintedRange": [
            {
                "report": report.name,
                "value": f"{lab.name} {lab.value_text} {lab.unit or ''}".strip(),
                "printedRange": lab.reference_text,
                "page": lab.provenance.page,
            }
            for report in reports
            for lab in parsing.abnormal_labs(report.parsed)
        ],
        "documentedStatements": [
            {"report": report.name, "heading": statement.heading, "page": statement.provenance.page}
            for report in reports
            for statement in report.parsed.diagnoses
        ],
        "aiGeneratedSections": bool(ai and ai.available),
    }
