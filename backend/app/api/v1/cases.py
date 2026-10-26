"""Case intake endpoints.

What these endpoints are for
----------------------------
Accepting a mixed set of files — an MRI volume, a written report, a 2D image — and
returning an honest account of each one: what it is, what could be analysed, what
could not, and why.

What they are NOT for
---------------------
They do not diagnose and they do not produce a clinical suggestion. No component of
this project can read a 2D X-ray or a written report, and the only trained model has
seen nothing but synthetic phantoms. Every response therefore carries
`interpretationSupported: false` with the reason, rather than a plausible-sounding
statement that nothing here can support.

Uploads are processed on this server, held in a temporary staging directory, and
are never sent anywhere else. Nothing about a case is written to a public location.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, UploadFile

from ...cases import intake
from ...datasets.storage import staging
from ...schemas.cases import (
    CaseCapabilities,
    CaseList,
    CaseSummary,
    ValidateCaseDirectoryRequest,
)
from ...schemas.common import Envelope, make_meta
from ...security import Principal, require_train

router = APIRouter(prefix="/cases", tags=["cases"])


@router.get(
    "/capabilities",
    response_model=Envelope[CaseCapabilities],
    summary="What can and cannot be done with each input type",
)
def get_capabilities() -> Envelope[CaseCapabilities]:
    """The capability matrix, served so the UI cannot overstate it."""
    return Envelope(data=CaseCapabilities(**intake.capabilities()), meta=make_meta())


@router.get("", response_model=Envelope[CaseList], summary="Analysed cases")
def list_cases() -> Envelope[CaseList]:
    summaries = [intake.summary(case) for case in intake.registry.list()]
    return Envelope(
        data=CaseList(cases=summaries, total=len(summaries)), meta=make_meta()
    )


@router.get(
    "/{case_id}",
    response_model=Envelope[CaseSummary],
    summary="One analysed case",
)
def get_case(case_id: str) -> Envelope[CaseSummary]:
    case = intake.registry.get(case_id)
    return Envelope(data=CaseSummary(**intake.summary(case)), meta=make_meta())


@router.post(
    "/intake",
    response_model=Envelope[CaseSummary],
    summary="Upload a case and receive a summary of what can be done with it",
)
async def intake_case(
    files: list[UploadFile] = File(...),
    name: str | None = Form(default=None),
    principal: Principal = Depends(require_train),
) -> Envelope[CaseSummary]:
    """Stage the upload, classify and analyse it, and summarise the case.

    The staging area is the same one the dataset workflow uses, so uploaded files
    are bounded, sanitised and swept identically. A failed analysis leaves nothing
    behind.
    """
    session = staging.create()
    try:
        for upload in files:
            # allowed_suffixes=None: this workflow exists to receive a PDF or a PNG
            # and explain what it is, so it cannot refuse them at the door. The name
            # sanitising and the size caps still come from the one shared
            # implementation in app/datasets/storage.py.
            session.add_stream(
                upload.filename or "unnamed", upload.file, allowed_suffixes=None
            )
        case = intake.analyse_staged(session.id, name)
    except Exception:
        staging.discard(session.id)
        raise
    return Envelope(data=CaseSummary(**intake.summary(case)), meta=make_meta())


@router.post(
    "/validate-directory",
    response_model=Envelope[CaseSummary],
    summary="Summarise a case directory already on this machine",
)
def intake_directory(
    payload: ValidateCaseDirectoryRequest,
    principal: Principal = Depends(require_train),
) -> Envelope[CaseSummary]:
    case = intake.analyse_directory(payload.path, payload.name)
    return Envelope(data=CaseSummary(**intake.summary(case)), meta=make_meta())


@router.delete(
    "/{case_id}",
    response_model=Envelope[dict],
    summary="Remove a case, deleting its staged files when it was uploaded",
)
def delete_case(
    case_id: str,
    principal: Principal = Depends(require_train),
) -> Envelope[dict]:
    case = intake.registry.get(case_id)
    was_staged = case.staged
    removed = intake.registry.remove(case_id)
    return Envelope(
        data={
            "id": case_id,
            "deleted": removed,
            "detail": (
                "Case and its staged files removed."
                if was_staged
                else "Case removed. The server-side directory was left untouched."
            ),
        },
        meta=make_meta(),
    )
