"""Extract text from a report. It does NOT interpret it.

Scope, stated plainly
---------------------
This module pulls the characters out of a PDF or text file and finds the headings
that are literally present. It does not summarise, classify, score or diagnose. A
radiology report is a document written by a clinician; rephrasing it into an
app-generated "impression" would be inventing clinical content, and the honest
thing is to show the clinician's own words under the heading they wrote.

The scanned-PDF case matters
----------------------------
Many reports are photographs of paper. Those have no text layer, so PDF extraction
returns nothing at all. Reporting "no findings" for such a file would be a serious
false statement: the findings are there, the software simply cannot read them. That
case is reported as `no_text_layer` with an explicit note that OCR is not
implemented, so the reader knows the difference between "nothing found" and
"nothing readable".
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

# How much text to keep. A report is far smaller than this; the cap exists so a
# pathological file cannot bloat a response.
MAX_CHARS = 40_000
MAX_SECTION_CHARS = 4_000

# Headings that radiology reports actually use. Matched only when a line is short
# and starts with one of these, so a sentence containing the word "finding" is not
# mistaken for a heading.
HEADINGS = (
    "clinical history",
    "history",
    "indication",
    "technique",
    "technique/procedure",
    "procedure",
    "comparison",
    "findings",
    "observation",
    "observations",
    "impression",
    "conclusion",
    "conclusions",
    "assessment",
    "diagnosis",
    "recommendation",
    "recommendations",
    "comment",
    "addendum",
    "result",
    "results",
)


class TextStatus(str, Enum):
    extracted = "extracted"
    no_text_layer = "no-text-layer"
    unsupported = "unsupported"
    failed = "failed"


@dataclass
class ReportSection:
    heading: str
    text: str


@dataclass
class ReportExtract:
    status: TextStatus
    characters: int = 0
    pages: int | None = None
    sections: list[ReportSection] = field(default_factory=list)
    headings_found: list[str] = field(default_factory=list)
    excerpt: str = ""
    note: str = ""
    # Text keyed by page, so a claim can cite the page it came from. One entry per
    # page for a PDF; a text file is a single "page". Empty when nothing was read.
    page_texts: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "status": self.status.value,
            "characters": self.characters,
            "pages": self.pages,
            "headingsFound": self.headings_found,
            "sections": [{"heading": s.heading, "text": s.text} for s in self.sections],
            "excerpt": self.excerpt,
            "note": self.note,
            "pageTexts": self.page_texts,
        }


def _split_sections(text: str) -> tuple[list[ReportSection], list[str]]:
    """Find heading lines and the text that follows each one.

    A line is treated as a heading only when it is short and matches a known
    heading, optionally followed by a colon. Prose is never split on the word
    alone, because "the finding was" is not a heading.
    """
    lines = text.splitlines()
    marks: list[tuple[int, str]] = []
    pattern = re.compile(
        r"^\s*(" + "|".join(re.escape(h) for h in HEADINGS) + r")\s*:?\s*$",
        re.IGNORECASE,
    )

    for index, line in enumerate(lines):
        stripped = line.strip()
        # Strip common markdown emphasis the exporters add.
        candidate = stripped.strip("*#_ ").strip()
        if not candidate or len(candidate) > 60:
            continue
        match = pattern.match(candidate)
        if match:
            # Strip punctuation so a heading is "Findings", not "Findings:".
            heading = candidate.rstrip(": ").strip().title()
            marks.append((index, heading))

    if not marks:
        return [], []

    sections: list[ReportSection] = []
    for position, (index, heading) in enumerate(marks):
        end = marks[position + 1][0] if position + 1 < len(marks) else len(lines)
        body = "\n".join(lines[index + 1 : end]).strip()
        sections.append(ReportSection(heading=heading, text=body[:MAX_SECTION_CHARS]))
    return sections, [heading for _, heading in marks]


def _extract_pdf(path: Path) -> ReportExtract:
    try:
        from pypdf import PdfReader
    except ImportError:
        return ReportExtract(
            status=TextStatus.unsupported,
            note="PDF support is not installed in this deployment (pypdf is missing).",
        )

    try:
        reader = PdfReader(str(path))
    except Exception as exc:  # noqa: BLE001 - reported as status, not raised
        return ReportExtract(
            status=TextStatus.failed,
            note=f"The PDF could not be opened: {type(exc).__name__}: {exc}",
        )

    pages = len(reader.pages)
    collected: list[str] = []
    for page in reader.pages:
        try:
            collected.append(page.extract_text() or "")
        except Exception:  # noqa: BLE001 - one bad page must not lose the rest
            collected.append("")

    text = "\n".join(collected).strip()
    if len(text) < 20:
        # A scanned report: the pixels are there, the characters are not.
        return ReportExtract(
            status=TextStatus.no_text_layer,
            pages=pages,
            page_texts=collected,
            note=(
                f"This PDF has {pages} page(s) but contains no extractable text, so it "
                "is almost certainly a scan or a photograph of a document. The report "
                "content is present but not readable as text, and OCR is NOT "
                "implemented here. Nothing can be quoted from it, and this is not a "
                "statement that the report is empty."
            ),
        )

    text = text[:MAX_CHARS]
    sections, headings = _split_sections(text)
    return ReportExtract(
        status=TextStatus.extracted,
        characters=len(text),
        pages=pages,
        page_texts=[page[:MAX_CHARS] for page in collected],
        sections=sections,
        headings_found=headings,
        excerpt=text[:1200],
        note=(
            "Text was extracted verbatim. Sections are shown under the headings the "
            "document itself uses; no wording has been generated, summarised or "
            "reinterpreted."
            if headings
            else (
                "Text was extracted verbatim, but no standard report headings "
                "(Findings, Impression, ...) were found, so the opening of the "
                "document is shown instead."
            )
        ),
    )


def _extract_text_file(path: Path) -> ReportExtract:
    raw = path.read_bytes()
    for encoding in ("utf-8", "cp1252", "latin-1"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:  # pragma: no cover - latin-1 decodes anything
        return ReportExtract(
            status=TextStatus.failed, note="The file could not be decoded as text."
        )

    text = text.strip()[:MAX_CHARS]
    if not text:
        return ReportExtract(
            status=TextStatus.no_text_layer,
            note="The file is empty, so there is nothing to quote.",
        )

    sections, headings = _split_sections(text)
    return ReportExtract(
        status=TextStatus.extracted,
        characters=len(text),
        pages=None,
        page_texts=[text],
        sections=sections,
        headings_found=headings,
        excerpt=text[:1200],
        note=(
            "Text was extracted verbatim. Sections are shown under the headings the "
            "document itself uses; no wording has been generated, summarised or "
            "reinterpreted."
            if headings
            else "Text was extracted verbatim; no standard report headings were found."
        ),
    )


def extract(path: Path) -> ReportExtract:
    """Extract text from a report file, dispatching on the real container."""
    path = Path(path)
    name = path.name.lower()
    if name.endswith(".pdf"):
        return _extract_pdf(path)
    return _extract_text_file(path)
