"""Dataset registry: validated datasets, and which hospital client they belong to.

State lives in this process, like the other services here. Nothing is written to a
database, because the deployment's filesystem is ephemeral and a dataset's real home
is the machine it was validated on.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..datasets import validation as validation_module
from ..datasets.storage import staging
from ..errors import ConflictError, NotFoundError, ValidationProblem
from ..schemas.datasets import (
    DatasetCase,
    DatasetSource,
    DatasetSummary,
    ValidationLimits,
)
from . import hospital_service


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class DatasetRecord:
    """One validated dataset."""

    id: str
    name: str
    source: DatasetSource
    path: Path
    created_at: datetime
    validation: validation_module.DatasetValidation
    staged: bool = False
    assigned_hospital_id: str | None = None
    assigned_hospital_name: str | None = None
    assigned_at: datetime | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class DatasetRegistry:
    """In-memory registry. Guarded by a lock because jobs read it from a thread."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._records: dict[str, DatasetRecord] = {}

    def register(
        self,
        validation: validation_module.DatasetValidation,
        *,
        name: str,
        source: DatasetSource,
        staged: bool = False,
    ) -> DatasetRecord:
        dataset_id = f"ds-{uuid.uuid4().hex[:12]}"
        record = DatasetRecord(
            id=dataset_id,
            name=name,
            source=source,
            path=Path(validation.path),
            created_at=_utc_now(),
            validation=validation,
            staged=staged,
        )
        with self._lock:
            self._records[dataset_id] = record
        return record

    def get(self, dataset_id: str) -> DatasetRecord:
        with self._lock:
            record = self._records.get(dataset_id)
        if record is None:
            raise NotFoundError(
                f"No dataset with id {dataset_id!r}.",
                code="dataset_not_found",
            )
        return record

    def list(self) -> list[DatasetRecord]:
        with self._lock:
            return sorted(
                self._records.values(), key=lambda record: record.created_at, reverse=True
            )

    def assign(self, dataset_id: str, hospital_id: str) -> DatasetRecord:
        """Point a dataset at a hospital client.

        Refuses an invalid dataset: assigning one that cannot be read would only
        move the failure to the moment training starts.
        """
        record = self.get(dataset_id)
        if not record.validation.valid:
            raise ValidationProblem(
                f"Dataset {record.name!r} did not pass validation, so it cannot be "
                "assigned. Fix the reported errors and validate it again.",
                code="dataset_not_valid",
            )

        hospital = hospital_service.get_hospital(hospital_id)

        with self._lock:
            # One dataset per client: a client trains on one local dataset.
            for other in self._records.values():
                if other.id != dataset_id and other.assigned_hospital_id == hospital.id:
                    other.assigned_hospital_id = None
                    other.assigned_hospital_name = None
                    other.assigned_at = None
            record.assigned_hospital_id = hospital.id
            record.assigned_hospital_name = hospital.name
            record.assigned_at = _utc_now()
        return record

    def remove(self, dataset_id: str) -> tuple[int, bool]:
        """Delete a dataset. Returns (files removed, staging discarded)."""
        record = self.get(dataset_id)
        with self._lock:
            self._records.pop(dataset_id, None)

        removed = 0
        discarded = False
        if record.staged:
            # Uploaded data is ours to delete, and it is deleted from disk too.
            # The staging directory is named after the upload session, not the
            # dataset id, so the session id recorded at validation time is used.
            session_name = record.extra.get("sessionId") or record.path.name
            discarded = staging.discard(session_name)
        # A server-side directory is the user's, not ours. Only the record goes.
        return removed, discarded

    def clear(self) -> None:
        with self._lock:
            self._records.clear()


registry = DatasetRegistry()


def _case_to_schema(case: validation_module.CaseReport) -> DatasetCase:
    return DatasetCase(**case.as_dict())


def to_summary(record: DatasetRecord) -> DatasetSummary:
    """Build the API shape for one dataset."""
    validation = record.validation
    # The limits a dataset was checked against, read from the module that enforces
    # them, so the UI cannot drift from the real values.
    constants = validation_module
    payload = {
        "id": record.id,
        "name": record.name,
        "source": record.source,
        "path": str(record.path),
        "valid": validation.valid,
        "caseCount": validation.case_count,
        "validCaseCount": validation.valid_case_count,
        "errorCount": validation.error_count,
        "warningCount": validation.warning_count,
        "totalBytes": validation.total_bytes,
        "modality": validation.modality,
        "modalitySource": validation.modality_source,
        "createdAt": record.created_at,
        "issues": [issue.as_dict() for issue in validation.issues],
        "cases": [_case_to_schema(case) for case in validation.cases],
        "limits": ValidationLimits(
            max_file_bytes=constants.MAX_FILE_BYTES,
            max_total_bytes=constants.MAX_TOTAL_BYTES,
            max_cases=constants.MAX_CASES,
            min_voxels=constants.MIN_VOXELS,
            max_voxels=constants.MAX_VOXELS,
            accepted_suffixes=list(constants.ACCEPTED_SUFFIXES),
        ),
        "assignedHospitalId": record.assigned_hospital_id,
        "assignedHospitalName": record.assigned_hospital_name,
        "assignedAt": record.assigned_at,
        # Ready to train means validated AND assigned. Not merely uploaded.
        "readyToTrain": validation.valid and bool(record.assigned_hospital_id),
    }
    return DatasetSummary(**payload)


def validate_server_directory(path: str, name: str | None = None) -> DatasetRecord:
    """Validate a directory already present on this machine and register it."""
    resolved = Path(path).expanduser()
    result = validation_module.validate_directory(resolved)
    return registry.register(
        result,
        name=name or resolved.name or str(resolved),
        source=DatasetSource.directory,
        staged=False,
    )


def validate_staged_session(session_id: str, name: str | None = None) -> DatasetRecord:
    """Validate what an upload staged, and register the result."""
    session = staging.session(session_id)
    if session is None:
        raise NotFoundError(
            f"No upload session {session_id!r}. It may have expired; staging is "
            "temporary.",
            code="upload_session_not_found",
        )
    result = validation_module.validate_directory(session.path)
    record = registry.register(
        result,
        name=name or session_id,
        source=DatasetSource.upload,
        staged=True,
    )
    # The staging directory name is the session id; keep it for later deletion.
    record.extra["sessionId"] = session_id
    return record


def assign(dataset_id: str, hospital_id: str) -> DatasetRecord:
    return registry.assign(dataset_id, hospital_id)


def list_summaries() -> list[DatasetSummary]:
    return [to_summary(record) for record in registry.list()]


def get_summary(dataset_id: str) -> DatasetSummary:
    return to_summary(registry.get(dataset_id))


def delete(dataset_id: str) -> tuple[int, bool]:
    """Remove a dataset from the registry, and its staged files when it was uploaded."""
    return registry.remove(dataset_id)
