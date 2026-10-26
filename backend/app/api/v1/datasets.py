"""Dataset management endpoints.

Where uploads are processed
---------------------------
Uploads are handled **on this server**. Files are streamed into a staging directory
on the same machine, validated there, and are **never forwarded to another service**.
Staging is temporary and swept by age; deleting a dataset removes its staged files.
A dataset given as a server-side path is only read — those files are the user's and
are never modified or deleted.

Access control
--------------
Reading the list needs nothing by default (matching the rest of the read-only
dashboard). Everything that changes state — uploading, validating, assigning,
deleting and training — requires the `train` scope. A deployment with no keys
configured refuses those routes with 503 rather than allowing anonymous access.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, Query, UploadFile

from ...datasets import validation as validation_module
from ...datasets.storage import STAGING_TTL_SECONDS, staging
from ...errors import ValidationProblem
from ...schemas.common import Envelope, make_meta
from ...schemas.datasets import (
    AssignDatasetRequest,
    DatasetDeleteResult,
    DatasetList,
    DatasetSummary,
    ImagingJobConfig,
    ImagingJobStatus,
    StartTrainingRequest,
    ValidateDirectoryRequest,
)
from ...security import Principal, require_train
from ...services import dataset_service, imaging_training_service

router = APIRouter(prefix="/datasets", tags=["datasets"])

# Requirement 4: the expected structure, stated by the API rather than only in docs,
# so the page and the server cannot describe it differently.
FORMAT_GUIDE = {
    "formats": list(validation_module.ACCEPTED_SUFFIXES),
    "formatNotes": {
        ".nii.gz": "Compressed NIfTI-1. Preferred. Read by the existing MONAI pipeline.",
        ".nii": "Uncompressed NIfTI-1. Also accepted.",
    },
    "folderStructure": [
        "dataset_folder/",
        "  case_001.nii.gz        <- the image volume",
        "  case_001_seg.nii.gz    <- its mask, same name with _seg",
        "  case_002.nii.gz",
        "  case_002_seg.nii.gz",
    ],
    "rules": [
        "One folder holds the whole dataset. Subfolders are searched too.",
        "Every <name>.nii.gz or <name>.nii needs <name>_seg.nii.gz or <name>_seg.nii "
        "next to it.",
        "Image and mask must be on the SAME voxel grid: identical shape and identical "
        "voxel size. A mismatch is rejected, because the pipeline resizes both "
        "independently and would otherwise train on a meaningless target.",
        "Each case must be a single 3D volume. A 4D file (time series, multiple "
        "channels) must be split into separate volumes first.",
        "Masks must be binary: values 0 and 1 only. Multi-label masks are rejected "
        "because this model performs binary segmentation.",
        "Minimum 8 voxels per axis; maximum 1024. The network down-samples 3 times.",
        "Masks are checked for an empty or nearly-full segmentation and warned about.",
    ],
    "notAccepted": dict(validation_module.REJECTED_SUFFIXES),
    "limits": {
        "maxFileBytes": validation_module.MAX_FILE_BYTES,
        "maxTotalBytes": validation_module.MAX_TOTAL_BYTES,
        "maxCases": validation_module.MAX_CASES,
        "stagingTtlSeconds": STAGING_TTL_SECONDS,
    },
    "privacy": [
        "Uploads are processed on this server and are not sent anywhere else.",
        "Uploaded files are held temporarily in a staging directory and are deleted "
        "when you delete the dataset or when they expire.",
        "Never upload identifiable patient data. Use synthetic phantoms or "
        "appropriately de-identified, approved research data.",
    ],
}


def staging_ttl() -> int:
    """Retained for callers that previously imported it; use the constant instead."""
    return STAGING_TTL_SECONDS


@router.get(
    "/formats",
    response_model=Envelope[dict],
    summary="Expected folder structure, formats and limits",
)
def get_formats() -> Envelope[dict]:
    return Envelope(data=FORMAT_GUIDE, meta=make_meta())


@router.get(
    "/clients",
    response_model=Envelope[list[dict]],
    summary="Hospital clients a dataset can be assigned to",
)
def get_clients() -> Envelope[list[dict]]:
    from ...services import hospital_service

    return Envelope(
        data=[
            {"id": node.id, "name": node.name, "location": node.location}
            for node in hospital_service.list_hospitals()
        ],
        meta=make_meta(),
    )


@router.get(
    "",
    response_model=Envelope[DatasetList],
    summary="Validated datasets",
)
def list_datasets() -> Envelope[DatasetList]:
    summaries = dataset_service.list_summaries()
    return Envelope(
        data=DatasetList(datasets=summaries, total=len(summaries)), meta=make_meta()
    )


@router.post(
    "/validate-directory",
    response_model=Envelope[DatasetSummary],
    summary="Validate a dataset directory already on this machine",
)
def validate_directory(
    payload: ValidateDirectoryRequest,
    principal: Principal = Depends(require_train),
) -> Envelope[DatasetSummary]:
    record = dataset_service.validate_server_directory(payload.path, payload.name)
    return Envelope(data=dataset_service.to_summary(record), meta=make_meta())


@router.post(
    "/upload",
    response_model=Envelope[DatasetSummary],
    summary="Upload and validate a dataset",
)
async def upload_dataset(
    files: list[UploadFile] = File(...),
    name: str | None = Form(default=None),
    principal: Principal = Depends(require_train),
) -> Envelope[DatasetSummary]:
    """Stage the uploaded files, then validate them with the same validator.

    There is one validator for both paths, so an uploaded dataset is held to
    exactly the rules a server-side directory is.
    """
    session = staging.create()
    try:
        for upload in files:
            session.add_stream(upload.filename or "", upload.file)
        record = dataset_service.validate_staged_session(session.id, name)
    except Exception:
        # A failed upload leaves nothing behind.
        staging.discard(session.id)
        raise
    return Envelope(data=dataset_service.to_summary(record), meta=make_meta())


@router.get(
    "/{dataset_id}",
    response_model=Envelope[DatasetSummary],
    summary="One dataset, with per-case detail",
)
def get_dataset(dataset_id: str) -> Envelope[DatasetSummary]:
    return Envelope(data=dataset_service.get_summary(dataset_id), meta=make_meta())


@router.post(
    "/{dataset_id}/assign",
    response_model=Envelope[DatasetSummary],
    summary="Assign a validated dataset to a hospital client",
)
def assign_dataset(
    dataset_id: str,
    payload: AssignDatasetRequest,
    principal: Principal = Depends(require_train),
) -> Envelope[DatasetSummary]:
    record = dataset_service.assign(dataset_id, payload.hospital_id)
    return Envelope(data=dataset_service.to_summary(record), meta=make_meta())


@router.delete(
    "/{dataset_id}",
    response_model=Envelope[DatasetDeleteResult],
    summary="Remove a dataset, deleting its staged files when it was uploaded",
)
def delete_dataset(
    dataset_id: str,
    principal: Principal = Depends(require_train),
) -> Envelope[DatasetDeleteResult]:
    from ...errors import NotFoundError

    record = dataset_service.registry.get(dataset_id)
    was_staged = record.staged
    staged_files = 0
    if was_staged:
        session = staging.session(record.extra.get("sessionId") or record.path.name)
        staged_files = len(session.files) if session is not None else 0
    _, discarded = dataset_service.delete(dataset_id)
    if not discarded and was_staged:
        raise ValidationProblem(
            "The dataset record was removed but its staged files could not be "
            "deleted. Remove them manually from the staging directory.",
            code="staging_delete_failed",
        )
    return Envelope(
        data=DatasetDeleteResult(
            id=dataset_id,
            deleted=True,
            # The number of uploaded files actually removed from disk. A
            # server-side directory is never touched, so it reports 0.
            files_removed=staged_files if discarded else 0,
            detail=(
                "Record and staged files removed."
                if was_staged
                else "Record removed. The server-side directory was left untouched."
            ),
        ),
        meta=make_meta(),
    )


# --- the training job --------------------------------------------------------
@router.post(
    "/training/start",
    response_model=Envelope[ImagingJobStatus],
    summary="Start a real 3D U-Net training job",
)
def start_training(
    payload: StartTrainingRequest,
    principal: Principal = Depends(require_train),
) -> Envelope[ImagingJobStatus]:
    status = imaging_training_service.job.start(payload.dataset_id, payload.config)
    return Envelope(data=status, meta=make_meta())


@router.get(
    "/training/status",
    response_model=Envelope[ImagingJobStatus],
    summary="Training job status, progress and measured metrics",
)
def training_status() -> Envelope[ImagingJobStatus]:
    status = imaging_training_service.job.status()
    available, reason = imaging_training_service.torch_available()
    status.available = available
    status.unavailable_reason = reason
    return Envelope(data=status, meta=make_meta())


@router.post(
    "/training/stop",
    response_model=Envelope[ImagingJobStatus],
    summary="Stop the running training job",
)
def stop_training(
    principal: Principal = Depends(require_train),
) -> Envelope[ImagingJobStatus]:
    return Envelope(data=imaging_training_service.job.request_stop(), meta=make_meta())


@router.post(
    "/training/reset",
    response_model=Envelope[ImagingJobStatus],
    summary="Clear a finished training job",
)
def reset_training(
    principal: Principal = Depends(require_train),
) -> Envelope[ImagingJobStatus]:
    return Envelope(data=imaging_training_service.job.reset(), meta=make_meta())
