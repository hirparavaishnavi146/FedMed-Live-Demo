"""Classify an uploaded file by its CONTENT, not by its extension.

Why content and not the name
----------------------------
An extension is a claim, not a fact: a scanned report is routinely saved as
``scan.pdf``, and a JPEG is routinely saved as ``.dcm`` by the wrong export
setting. Treating inputs as interchangeable because their names look plausible is
exactly the failure the brief warns about, so every file is identified from its
first bytes and the extension is recorded separately as a hint.

Nothing here interprets anything. This module only answers "what kind of file is
this", which is the precondition for saying honestly what can and cannot be done
with it.
"""

from __future__ import annotations

import gzip
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class FileKind(str, Enum):
    """What a file is. Deliberately narrow: only what this project can act on."""

    mri_volume = "mri-volume"
    report_pdf = "report-pdf"
    report_text = "report-text"
    image_2d = "2d-image"
    dicom = "dicom"
    archive = "archive"
    unsupported = "unsupported"


# A NIfTI-1 volume starts with sizeof_hdr == 348, little- or big-endian.
_NIFTI_LE = b"\x5c\x01\x00\x00"
_NIFTI_BE = b"\x00\x00\x01\x5c"

_MAGIC = (
    (b"%PDF", FileKind.report_pdf),
    (b"\x89PNG\r\n\x1a\n", FileKind.image_2d),
    (b"\xff\xd8\xff", FileKind.image_2d),
    (b"BM", FileKind.image_2d),
    (b"II*\x00", FileKind.image_2d),
    (b"MM\x00*", FileKind.image_2d),
    (b"\x1f\x8b", None),  # gzip: may be a .nii.gz, but the name decides
    (b"PK\x03\x04", FileKind.archive),
)

_TEXT_SUFFIXES = {".txt", ".md", ".text", ".report"}
_EXT_HINTS = {
    ".nii": FileKind.mri_volume,
    ".nii.gz": FileKind.mri_volume,
    ".dcm": FileKind.unsupported,  # DICOM is refused; see below
    ".dicom": FileKind.unsupported,
}


@dataclass
class Classified:
    """A file's real kind, plus how confident that identification is."""

    kind: FileKind
    method: str
    note: str
    suffix: str

    @property
    def is_analysable_image(self) -> bool:
        return self.kind is FileKind.mri_volume


def _looks_like_text(sample: bytes) -> bool:
    """True when the bytes decode as text with no NULs.

    A radiology report exported as .txt or .md is text; a DICOM or a compressed
    volume is not. NUL bytes and undecodable runs rule text out.
    """
    if not sample:
        return False
    if b"\x00" in sample[:2048]:
        return False
    try:
        sample.decode("utf-8")
        return True
    except UnicodeDecodeError:
        # Allow a high ratio of valid text rather than demanding a clean decode.
        try:
            sample.decode("cp1252")
        except UnicodeDecodeError:
            return False
        printable = sum(1 for b in sample if 32 <= b < 127 or b in (9, 10, 13))
        return printable / max(1, len(sample)) > 0.85


def classify(path: Path) -> Classified:
    """Identify one file. Reads at most a few kilobytes."""
    path = Path(path)
    name = path.name.lower()
    suffix = ".nii.gz" if name.endswith(".nii.gz") else path.suffix.lower()

    with path.open("rb") as handle:
        head = handle.read(512)

    # --- DICOM: "DICM" at byte 128. Refused, and named as DICOM, not as .dcm. ---
    if len(head) >= 132 and head[128:132] == b"DICM":
        return Classified(
            FileKind.dicom,
            method="magic bytes (DICM at offset 128)",
            suffix=suffix,
            note=(
                "This is a DICOM file. This project cannot read DICOM series: a "
                "series is many files describing one volume, and reading it needs a "
                "DICOM library that is not installed. Convert the series to a single "
                "NIfTI volume first."
            ),
        )

    # --- NIfTI, by header magic, taking the .gz container into account. --------
    if head[:2] == b"\x1f\x8b":
        try:
            with gzip.open(path, "rb") as handle:
                inner = handle.read(512)
        except OSError:
            inner = b""
        if inner[:4] in (_NIFTI_LE, _NIFTI_BE):
            return Classified(
                FileKind.mri_volume,
                method="gzip container, NIfTI-1 header (sizeof_hdr = 348)",
                suffix=suffix,
                note="Compressed NIfTI-1 volume.",
            )
        return Classified(
            FileKind.unsupported,
            method="gzip container with no NIfTI header",
            suffix=suffix,
            note=(
                "This is a gzip archive but does not contain a NIfTI volume. "
                "Archives are not analysed."
            ),
        )

    if head[:4] in (_NIFTI_LE, _NIFTI_BE):
        return Classified(
            FileKind.mri_volume,
            method="NIfTI-1 header (sizeof_hdr = 348)",
            suffix=suffix,
            note="Uncompressed NIfTI-1 volume.",
        )

    # --- Document and image containers, by magic. ---------------------------
    for magic, kind in _MAGIC:
        if kind is None:
            continue
        if head.startswith(magic):
            if kind is FileKind.report_pdf:
                return Classified(
                    FileKind.report_pdf,
                    method="magic bytes (%PDF)",
                    suffix=suffix,
                    note="PDF document. Text is extracted if the file has a text layer.",
                )
            if kind is FileKind.image_2d:
                label = {
                    b"\x89PNG\r\n\x1a\n": "PNG",
                    b"\xff\xd8\xff": "JPEG",
                    b"BM": "BMP",
                    b"II*\x00": "TIFF",
                    b"MM\x00*": "TIFF",
                }.get(magic, "image")
                return Classified(
                    FileKind.image_2d,
                    method=f"magic bytes ({label})",
                    suffix=suffix,
                    note=(
                        f"{label} image. This is a 2D picture. No model in this "
                        "project can interpret it; see capabilities()."
                    ),
                )
            return Classified(
                FileKind.archive,
                method="magic bytes (ZIP)",
                suffix=suffix,
                note="Archive. Extract it and upload the individual files.",
            )

    # --- Plain text reports. -------------------------------------------------
    if suffix in _TEXT_SUFFIXES or _looks_like_text(head):
        return Classified(
            FileKind.report_text,
            method="decodes as text",
            suffix=suffix,
            note="Text document. Headings and their sections are extracted verbatim.",
        )

    hint = _EXT_HINTS.get(suffix)
    if hint is not None:
        return Classified(
            FileKind.unsupported,
            method=f"extension only ({suffix})",
            suffix=suffix,
            note=(
                f"The name suggests {suffix}, but the content is not a NIfTI volume. "
                "Convert it to .nii.gz or .nii."
            ),
        )

    return Classified(
        FileKind.unsupported,
        method="no recognised signature",
        suffix=suffix,
        note=(
            "Not recognised as a NIfTI volume, a PDF, a text report or a 2D image. "
            "Nothing is assumed about an unknown file."
        ),
    )
