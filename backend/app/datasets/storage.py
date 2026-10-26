"""Safe staging for uploaded datasets.

Where uploads live, and what happens to them
--------------------------------------------
Uploads are handled **on the same server** that serves the API. They are written to
a staging directory and are **never sent to any other service**. The staging root
is resolved outside the repository by default, so an uploaded patient scan can never
be committed to Git by accident, and it is swept on age so a dataset does not sit on
disk indefinitely.

Everything here assumes the uploaded bytes are hostile:

* the filename from the client is discarded for path purposes and rebuilt from
  sanitised characters — ``..\\..\\evil.nii.gz`` becomes ``evil.nii.gz``;
* the extension must be an accepted NIfTI suffix, checked on the sanitised name;
* each file and the whole session are size-capped while streaming, so a client
  cannot exhaust the disk by simply not stopping;
* nothing is executed, imported, deserialised or shelled out to.
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO

from ..errors import ValidationProblem
from .validation import ACCEPTED_SUFFIXES, MAX_FILE_BYTES, MAX_TOTAL_BYTES

# Uploads are temporary. A session older than this is removed by the sweeper.
STAGING_TTL_SECONDS: int = 6 * 60 * 60

_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")
_CHUNK = 1024 * 1024

# Environment override, so a deployment can point staging at a mounted volume.
ROOT_ENV_VAR = "FEDMED_STAGING_DIR"


def staging_root() -> Path:
    """The staging directory, outside the repository unless configured otherwise.

    Defaults to a ``fedmed-staging`` directory in the system temp area rather than
    anything under the project tree, so nothing here can be picked up by Git.
    """
    configured = os.environ.get(ROOT_ENV_VAR)
    root = Path(configured) if configured else Path(tempfile.gettempdir()) / "fedmed-staging"
    root.mkdir(parents=True, exist_ok=True)
    return root


def safe_filename(raw: str, *, allowed_suffixes: tuple[str, ...] | None = ACCEPTED_SUFFIXES) -> str:
    """Turn a client-supplied name into a safe basename with an accepted suffix.

    Directory components are dropped rather than sanitised: the only legitimate
    place for an uploaded file is directly inside its own session directory.

    `allowed_suffixes=None` allows any extension while keeping every other rule.
    """
    if not raw or not raw.strip():
        raise ValidationProblem("An uploaded file had an empty name.", code="upload_name_missing")

    # Take the basename under both separators, then discard anything left.
    name = raw.replace("\\", "/").split("/")[-1].strip()
    name = _SAFE_NAME.sub("_", name).lstrip("._-")

    if not name:
        raise ValidationProblem(
            f"The uploaded filename {raw!r} contains no usable characters.",
            code="upload_name_invalid",
        )

    if allowed_suffixes is not None:
        lowered = name.lower()
        if not any(lowered.endswith(suffix) for suffix in allowed_suffixes):
            accepted = " or ".join(allowed_suffixes)
            raise ValidationProblem(
                f"{name} is not a supported file type. Upload NIfTI files ({accepted}) only.",
                code="upload_unsupported_format",
            )
    return name


@dataclass
class StagingSession:
    """One upload batch: a directory of files that together form a dataset."""

    id: str
    path: Path
    created_at: float = field(default_factory=time.time)
    total_bytes: int = 0
    files: list[str] = field(default_factory=list)

    def add_stream(
        self,
        raw_name: str,
        stream: BinaryIO,
        *,
        allowed_suffixes: tuple[str, ...] | None = ACCEPTED_SUFFIXES,
    ) -> str:
        """Write one uploaded file, enforcing the name rules and size caps.

        Returns the name written. Raises ValidationProblem on a bad name, an
        unsupported type, or an exceeded limit.

        `allowed_suffixes=None` accepts any type. The case-intake workflow needs
        that, because its whole purpose is to receive a PDF or a 2D image and
        explain what it is — refusing them at the door would make the classification
        impossible. The name sanitising and the size caps still apply, so the limits
        live in one place either way.
        """
        name = safe_filename(raw_name, allowed_suffixes=allowed_suffixes)
        target = self.path / name

        # Two uploads of the same name replace each other rather than both landing,
        # which keeps a re-upload of one corrected case idempotent.
        written = 0
        try:
            with target.open("wb") as handle:
                while True:
                    chunk = stream.read(_CHUNK)
                    if not chunk:
                        break
                    written += len(chunk)
                    if written > MAX_FILE_BYTES:
                        handle.close()
                        target.unlink(missing_ok=True)
                        raise ValidationProblem(
                            f"{name} exceeds the "
                            f"{MAX_FILE_BYTES // 1024 // 1024} MB per-file limit.",
                            code="upload_file_too_large",
                        )
                    if self.total_bytes + written > MAX_TOTAL_BYTES:
                        handle.close()
                        target.unlink(missing_ok=True)
                        raise ValidationProblem(
                            f"This upload would exceed the "
                            f"{MAX_TOTAL_BYTES // 1024 // 1024 // 1024} GB dataset "
                            "limit. Split the dataset before uploading.",
                            code="upload_total_too_large",
                        )
                    handle.write(chunk)
        except ValidationProblem:
            raise
        except OSError as exc:
            target.unlink(missing_ok=True)
            raise ValidationProblem(
                f"{name} could not be written to the staging area: {exc}",
                code="upload_write_failed",
            ) from exc

        if written == 0:
            target.unlink(missing_ok=True)
            raise ValidationProblem(
                f"{name} was uploaded empty.", code="upload_empty_file"
            )

        self.total_bytes += written
        if name not in self.files:
            self.files.append(name)
        return name


class StagingArea:
    """Creates, looks up and sweeps staging sessions."""

    def __init__(self, root: Path | None = None) -> None:
        self._root = Path(root) if root is not None else staging_root()
        self._root.mkdir(parents=True, exist_ok=True)

    @property
    def root(self) -> Path:
        return self._root

    def create(self) -> StagingSession:
        session_id = f"ds-{uuid.uuid4().hex[:12]}"
        path = self._root / session_id
        path.mkdir(parents=True, exist_ok=False)
        return StagingSession(id=session_id, path=path)

    def session(self, session_id: str) -> StagingSession | None:
        """Re-open an existing session, refusing anything that is not a plain id.

        The id is validated rather than joined blindly, so a crafted value cannot
        walk out of the staging root.
        """
        if not re.fullmatch(r"ds-[0-9a-f]{12}", session_id or ""):
            return None
        path = self._root / session_id
        if not path.is_dir():
            return None
        files = sorted(entry.name for entry in path.iterdir() if entry.is_file())
        total = sum((path / name).stat().st_size for name in files)
        return StagingSession(id=session_id, path=path, total_bytes=total, files=files)

    def discard(self, session_id: str) -> bool:
        """Delete a session and everything in it."""
        session = self.session(session_id)
        if session is None:
            return False
        shutil.rmtree(session.path, ignore_errors=True)
        return not session.path.exists()

    def sweep_expired(self, max_age_seconds: int = STAGING_TTL_SECONDS) -> list[str]:
        """Remove sessions older than ``max_age_seconds``. Returns the ids removed."""
        removed: list[str] = []
        if not self._root.exists():
            return removed
        for entry in self._root.iterdir():
            if not entry.is_dir():
                continue
            try:
                age = time.time() - entry.stat().st_mtime
            except OSError:  # pragma: no cover - racing deletion
                continue
            if age > max_age_seconds:
                shutil.rmtree(entry, ignore_errors=True)
                removed.append(entry.name)
        return removed


# A single shared area, like the other services in this project.
staging = StagingArea()
