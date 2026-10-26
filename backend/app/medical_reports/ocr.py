"""Optional OCR for scanned pages, with an honest account of what is missing.

OCR is genuinely unavailable in this environment, and the code says so precisely
rather than failing vaguely or silently returning nothing.

Two separate things are needed, and they fail independently:

1. **An OCR engine.** ``pytesseract`` is a Python wrapper; it needs the Tesseract
   *binary* installed on the machine. Tesseract is a system program, not a pip
   package, so it cannot be added by installing a requirement — the host has to have
   it. Installing a system binary is not something this project should do silently.
2. **A PDF rasteriser**, only for OCR-ing a PDF: Tesseract reads an image, so a PDF
   page must be rendered to one first. That needs something like PyMuPDF or
   pdf2image with Poppler. Neither is installed.

So: a standalone image needs (1); a scanned PDF needs both. Each failure is reported
with the specific missing piece, because "OCR failed" is not actionable while "the
Tesseract binary is not installed" is.

Nothing here ever guesses at text it cannot read. A page that cannot be OCR-ed is
reported as unreadable, which is different from empty.
"""

from __future__ import annotations

import importlib.util
import shutil
from dataclasses import dataclass
from pathlib import Path

from ..errors import ApiError

# How much OCR text to keep per page, so a pathological image cannot flood a response.
MAX_OCR_CHARS_PER_PAGE = 20_000

INSTALL_HINT = (
    "To enable OCR, install the Tesseract binary for your platform and the Python "
    "wrapper (pip install pytesseract). For PDF pages, also install a rasteriser "
    "(pip install pymupdf) or Poppler for pdf2image. Until then, scanned documents "
    "cannot be read and are reported as unreadable rather than as empty."
)


@dataclass
class OcrAvailability:
    engine: bool
    pdf_rasteriser: bool
    engine_detail: str
    rasteriser_detail: str

    @property
    def available_for_images(self) -> bool:
        return self.engine

    @property
    def available_for_pdfs(self) -> bool:
        return self.engine and self.pdf_rasteriser

    @property
    def reason(self) -> str | None:
        if self.available_for_pdfs:
            return None
        missing: list[str] = []
        if not self.engine:
            missing.append("the Tesseract OCR engine")
        if not self.pdf_rasteriser:
            missing.append("a PDF-to-image rasteriser")
        return (
            "Optical character recognition is not available in this deployment: "
            f"{' and '.join(missing)} not installed. " + INSTALL_HINT
        )

    def as_dict(self) -> dict:
        return {
            "availableForImages": self.available_for_images,
            "availableForPdfs": self.available_for_pdfs,
            "engineInstalled": self.engine,
            "engineDetail": self.engine_detail,
            "pdfRasteriserInstalled": self.pdf_rasteriser,
            "rasteriserDetail": self.rasteriser_detail,
            "reason": self.reason,
            "installHint": INSTALL_HINT,
        }


def detect() -> OcrAvailability:
    """Check both requirements without importing anything heavy."""
    engine = importlib.util.find_spec("pytesseract") is not None
    binary = shutil.which("tesseract")
    if engine and binary:
        engine_detail = f"pytesseract installed; tesseract binary at {binary}"
        engine_ok = True
    elif engine and not binary:
        engine_detail = (
            "pytesseract is installed but the tesseract binary was not found on PATH, "
            "so no OCR can run."
        )
        engine_ok = False
    elif binary and not engine:
        engine_detail = (
            f"the tesseract binary is at {binary} but the pytesseract wrapper is not "
            "installed."
        )
        engine_ok = False
    else:
        engine_detail = "neither pytesseract nor the tesseract binary is installed."
        engine_ok = False

    rasteriser = importlib.util.find_spec("fitz") is not None or importlib.util.find_spec("pdf2image") is not None
    rasteriser_detail = (
        "a PDF rasteriser is installed (PyMuPDF or pdf2image)."
        if rasteriser
        else "no PDF rasteriser is installed (needs PyMuPDF or pdf2image with Poppler)."
    )

    return OcrAvailability(
        engine=engine_ok,
        pdf_rasteriser=rasteriser,
        engine_detail=engine_detail,
        rasteriser_detail=rasteriser_detail,
    )


def ocr_image(path: Path) -> str:
    """Read text from a standalone image.

    Raises ApiError when OCR is unavailable, because returning an empty string would
    be indistinguishable from "the image contains no text".
    """
    availability = detect()
    if not availability.available_for_images:
        raise ApiError(
            availability.reason or "OCR is not available.",
            code="ocr_unavailable",
            status_code=503,
        )

    import pytesseract  # imported after the availability check
    from PIL import Image

    with Image.open(path) as image:
        try:
            text = pytesseract.image_to_string(image)
        except Exception as exc:  # noqa: BLE001
            raise ApiError(
                f"OCR failed on {path.name}: {type(exc).__name__}.",
                code="ocr_failed",
                status_code=422,
            ) from exc
    return text.strip()[:MAX_OCR_CHARS_PER_PAGE]


def ocr_pdf_pages(path: Path) -> list[str]:
    """Render each PDF page to an image and OCR it.

    Raises ApiError when either requirement is missing.
    """
    availability = detect()
    if not availability.available_for_pdfs:
        raise ApiError(
            availability.reason or "OCR is not available for PDFs.",
            code="ocr_unavailable",
            status_code=503,
        )

    import pytesseract

    pages: list[str] = []
    if importlib.util.find_spec("fitz") is not None:
        import fitz

        with fitz.open(str(path)) as document:
            for page in document:
                pixmap = page.get_pixmap(dpi=200)
                try:
                    text = pytesseract.image_to_string(pixmap.tobytes("png"))
                except Exception as exc:  # noqa: BLE001
                    raise ApiError(
                        f"OCR failed on a page of {path.name}: {type(exc).__name__}.",
                        code="ocr_failed",
                        status_code=422,
                    ) from exc
                pages.append(text.strip()[:MAX_OCR_CHARS_PER_PAGE])
        return pages

    from pdf2image import convert_from_path

    for image in convert_from_path(str(path), dpi=200):
        try:
            text = pytesseract.image_to_string(image)
        except Exception as exc:  # noqa: BLE001
            raise ApiError(
                f"OCR failed on a page of {path.name}: {type(exc).__name__}.",
                code="ocr_failed",
                status_code=422,
            ) from exc
        pages.append(text.strip()[:MAX_OCR_CHARS_PER_PAGE])
    return pages
