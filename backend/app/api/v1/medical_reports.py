"""Medical report analysis endpoints.

Access control
--------------
**Every** route here requires the ``train`` scope, reads included. That differs from
the rest of the dashboard, which leaves reads open, and the difference is deliberate:
this feature handles real medical documents, which are about as sensitive as data
gets. The synthetic dashboard can be browsed anonymously; a patient's report cannot.

Where the files go
------------------
Uploads are processed on this server, written to the shared staging area outside the
repository, and deleted when the session is removed or when staging ages out. They are
never sent anywhere except, if and only if an AI provider is configured, to that
provider as part of the configured analysis request. The health endpoint states
whether an external provider is in use, so that disclosure is explicit rather than
buried.

Nothing here logs report contents. Errors are returned without internal detail.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile

from ...datasets.storage import staging
from ...errors import ConflictError, NotFoundError, ValidationProblem
from ...medical_reports import analysis, analysis_ai, ocr
from ...schemas.common import Envelope, make_meta
from ...schemas.medical_reports import (
    AnalyzeRequest,
    AnalyzeResult,
    CompareResult,
    ConfirmSamePatientRequest,
    ExtractResult,
    ExtractedReport,
    ReportSession,
    ReportUploadResult,
    SubsystemHealth,
)
from ...security import Principal, Scope, require_scope_or_public_demo

router = APIRouter(prefix="/medical-reports", tags=["medical-reports"])

#: The medical-report routes are the one place where authentication may be relaxed,
#: and only by an explicit server-side flag. See require_scope_or_public_demo.
require_train_or_public_demo = require_scope_or_public_demo(Scope.train)

ACCEPTED_TYPES = [".pdf", ".txt", ".md", ".text", ".png", ".jpg", ".jpeg", ".bmp", ".tif"]
# Smaller than the dataset limit on purpose: a single report is a document, not a
# volume series, and a smaller cap limits how much sensitive material can be staged.
MAX_REPORT_BYTES = 25 * 1024 * 1024


def _session_payload(session: analysis.Session) -> ReportSession:
    return ReportSession(
        session_id=session.session_id,
        created_at=session.created_at,
        report_count=len(session.reports),
        same_patient_confirmed=session.same_patient_confirmed,
        reports=[report.as_dict() for report in session.reports],
        analysis_ids=list(session.analyses),
    )


@router.get(
    "/health",
    response_model=Envelope[SubsystemHealth],
    summary="What the report-analysis subsystem can currently do",
)
def health(
    request: Request,
    principal: Principal | None = Depends(require_train_or_public_demo),
) -> Envelope[SubsystemHealth]:
    """State AI and OCR availability, so the UI cannot imply a capability that is absent."""
    # The app's own settings object, not the module-level get_settings() singleton:
    # the singleton is built from the environment, and the two are only guaranteed to
    # agree in production. Reading the wrong one reported publicDemoMode=False while
    # the flag was on.
    settings = request.app.state.settings
    ai_state = analysis_ai.availability(settings)
    ocr_state = ocr.detect()
    public_demo = bool(settings.medical_reports_public_demo)
    return Envelope(
        data=SubsystemHealth(
            available=True,
            ai=ai_state.as_dict(),
            ocr=ocr_state.as_dict(),
            max_reports=analysis.MAX_REPORTS,
            max_file_bytes=MAX_REPORT_BYTES,
            accepted_types=ACCEPTED_TYPES,
            # Not a constant: under public demo mode these routes genuinely do not
            # require a key. Reporting True there would be a false statement about the
            # deployment, which is worse than reporting the uncomfortable truth.
            auth_required=not public_demo,
            note=(
                "Report extraction, comparison and abnormality detection work without an "
                "AI provider. The generated sections — possible complications, clinical "
                "suggestions and the narrative summary — require one, and are reported as "
                "unavailable until FEDMED_AI_BASE_URL and FEDMED_AI_API_KEY are set."
            ),
            public_demo_mode=public_demo,
            public_demo_warning=(
                "PUBLIC DEMO MODE: this server accepts uploads and analysis requests from "
                "anyone, with no API key. Use it only with synthetic examples — never "
                "upload a real patient's report to this deployment. The operator turns "
                "this off with FEDMED_MEDICAL_REPORTS_PUBLIC_DEMO=false."
                if public_demo
                else None
            ),
        ),
        meta=make_meta(),
    )


def _settings():
    from ...core.config import get_settings

    return get_settings()


@router.post(
    "/sessions",
    response_model=Envelope[ReportSession],
    summary="Create an analysis session",
)
def create_session(
    principal: Principal | None = Depends(require_train_or_public_demo),
) -> Envelope[ReportSession]:
    session = analysis.registry.create()
    return Envelope(data=_session_payload(session), meta=make_meta())


@router.get(
    "/sessions/{session_id}",
    response_model=Envelope[ReportSession],
    summary="One session and its reports",
)
def get_session(
    session_id: str,
    principal: Principal | None = Depends(require_train_or_public_demo),
) -> Envelope[ReportSession]:
    return Envelope(data=_session_payload(analysis.registry.get(session_id)), meta=make_meta())


@router.post(
    "/upload",
    response_model=Envelope[ReportUploadResult],
    summary="Upload one to three reports for a patient",
)
async def upload(
    session_id: str = Form(...),
    files: list[UploadFile] = File(...),
    principal: Principal | None = Depends(require_train_or_public_demo),
) -> Envelope[ReportUploadResult]:
    session = analysis.registry.get(session_id)

    remaining = analysis.MAX_REPORTS - len(session.reports)
    if remaining <= 0:
        raise ConflictError(
            f"This session already holds the maximum of {analysis.MAX_REPORTS} reports. "
            "Remove one before adding another.",
            code="report_limit_reached",
        )

    accepted: list[dict] = []
    rejected: list[dict] = []

    for upload_file in files:
        raw_name = upload_file.filename or "unnamed"
        safe_name = raw_name.replace("\\", "/").split("/")[-1].strip() or "unnamed"
        lowered = safe_name.lower()

        if not any(lowered.endswith(suffix) for suffix in ACCEPTED_TYPES):
            rejected.append(
                {
                    "name": safe_name,
                    "reason": (
                        f"Unsupported file type. Accepted: {', '.join(ACCEPTED_TYPES)}."
                    ),
                }
            )
            continue

        # One staging directory per session, so deleting the session deletes the lot.
        session_dir = staging.root / session.session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        target = session_dir / safe_name

        written = 0
        try:
            with target.open("wb") as handle:
                while True:
                    chunk = upload_file.file.read(1024 * 1024)
                    if not chunk:
                        break
                    written += len(chunk)
                    if written > MAX_REPORT_BYTES:
                        handle.close()
                        target.unlink(missing_ok=True)
                        raise ValidationProblem(
                            f"{safe_name} exceeds the "
                            f"{MAX_REPORT_BYTES // 1024 // 1024} MB limit for a single "
                            "report.",
                            code="report_file_too_large",
                        )
                    handle.write(chunk)
        except ValidationProblem:
            session_dir_exists = session_dir.exists() and any(session_dir.iterdir())
            if not session_dir_exists:
                session_dir.rmdir()
            raise

        if written == 0:
            target.unlink(missing_ok=True)
            rejected.append({"name": safe_name, "reason": "The file was empty."})
            continue

        # A second upload of the same name replaces the first, so re-uploading a
        # corrected report does not silently leave both in the analysis.
        session.reports = [item for item in session.reports if item.name != safe_name]

        try:
            record = analysis.extract_report(
                session.session_id, safe_name, target, written, upload_file.content_type
            )
        except ValidationProblem as exc:
            target.unlink(missing_ok=True)
            rejected.append({"name": safe_name, "reason": exc.message})
            continue

        session.reports.append(record)
        accepted.append(
            {
                "reportId": record.report_id,
                "name": record.name,
                "status": record.extract.status.value,
                "pages": record.extract.pages,
                "characters": record.extract.characters,
                "ocrUsed": record.ocr_used,
            }
        )

        if len(session.reports) >= analysis.MAX_REPORTS:
            break

    return Envelope(
        data=ReportUploadResult(
            session_id=session.session_id,
            accepted=accepted,
            rejected=rejected,
            report_count=len(session.reports),
            max_reports=analysis.MAX_REPORTS,
            note=(
                "Text extraction runs on upload. Confirm the reports belong to the same "
                "patient before comparing them."
            ),
        ),
        meta=make_meta(),
    )


@router.post(
    "/extract",
    response_model=Envelope[ExtractResult],
    summary="Extraction and parsed facts for a session's reports",
)
def extract(
    payload: AnalyzeRequest,
    principal: Principal | None = Depends(require_train_or_public_demo),
) -> Envelope[ExtractResult]:
    session = analysis.registry.get(payload.session_id)
    if not session.reports:
        raise ValidationProblem(
            "No reports have been uploaded for this session.", code="no_reports_uploaded"
        )
    return Envelope(
        data=ExtractResult(
            session_id=session.session_id,
            reports=[
                ExtractedReport(
                    report_id=report.report_id,
                    name=report.name,
                    status=report.extract.status.value,
                    pages=report.extract.pages,
                    characters=report.extract.characters,
                    headings_found=report.extract.headings_found,
                    ocr_used=report.ocr_used,
                    page_texts=report.extract.page_texts,
                    parsed=report.parsed.as_dict(),
                    note=report.extract.note,
                )
                for report in session.reports
            ],
            ocr=ocr.detect().as_dict(),
            note=(
                "Values are compared only against reference ranges printed in the report "
                "itself. A value printed without a range is reported without a verdict."
            ),
        ),
        meta=make_meta(),
    )


@router.post(
    "/confirm",
    response_model=Envelope[ReportSession],
    summary="Confirm the reports belong to the same patient",
)
def confirm(
    payload: ConfirmSamePatientRequest,
    principal: Principal | None = Depends(require_train_or_public_demo),
) -> Envelope[ReportSession]:
    session = analysis.registry.get(payload.session_id)
    session.same_patient_confirmed = bool(payload.confirmed)
    if payload.confirmed:
        from datetime import datetime, timezone

        session.confirmed_at = datetime.now(timezone.utc)
    return Envelope(data=_session_payload(session), meta=make_meta())


@router.post(
    "/compare",
    response_model=Envelope[CompareResult],
    summary="Compare findings across the session's reports",
)
def compare(
    payload: AnalyzeRequest,
    principal: Principal | None = Depends(require_train_or_public_demo),
) -> Envelope[CompareResult]:
    session = analysis.registry.get(payload.session_id)
    if not session.reports:
        raise ValidationProblem(
            "No reports have been uploaded for this session.", code="no_reports_uploaded"
        )
    if not session.same_patient_confirmed:
        raise ValidationProblem(
            "Confirm that the uploaded reports belong to the same patient before "
            "comparing them.",
            code="same_patient_not_confirmed",
        )
    return Envelope(
        data=CompareResult(
            session_id=session.session_id,
            comparison=analysis.compare(session.reports),
            report_count=len(session.reports),
        ),
        meta=make_meta(),
    )


@router.post(
    "/analyze",
    response_model=Envelope[AnalyzeResult],
    summary="Analyse a session and return the decision-support draft",
)
def analyze(
    payload: AnalyzeRequest,
    principal: Principal | None = Depends(require_train_or_public_demo),
) -> Envelope[AnalyzeResult]:
    session = analysis.registry.get(payload.session_id)
    stored = analysis.analyse(session, _settings())
    return Envelope(data=AnalyzeResult(**stored), meta=make_meta())


@router.get(
    "/analysis/{analysis_id}",
    response_model=Envelope[AnalyzeResult],
    summary="Fetch a stored analysis",
)
def get_analysis(
    analysis_id: str,
    principal: Principal | None = Depends(require_train_or_public_demo),
) -> Envelope[AnalyzeResult]:
    for session in analysis.registry.list():
        if analysis_id in session.analyses:
            return Envelope(data=AnalyzeResult(**session.analyses[analysis_id]), meta=make_meta())
    raise NotFoundError(
        f"No analysis with id {analysis_id!r}. It may have expired; analyses are held in "
        "memory only.",
        code="analysis_not_found",
    )


@router.delete(
    "/sessions/{session_id}",
    response_model=Envelope[dict],
    summary="Delete a session and purge its uploaded report files",
)
def delete_session(
    session_id: str,
    principal: Principal | None = Depends(require_train_or_public_demo),
) -> Envelope[dict]:
    session = analysis.registry.get(session_id)
    removed = staging.discard(session_id)
    with analysis.registry._lock:  # noqa: SLF001 - intentional purge of this session
        analysis.registry._sessions.pop(session_id, None)
    return Envelope(
        data={
            "sessionId": session_id,
            "deleted": True,
            "filesRemoved": removed,
            "detail": (
                "Session removed and its uploaded report files deleted from staging. "
                "In-memory extraction and analysis are discarded."
            ),
        },
        meta=make_meta(),
    )
