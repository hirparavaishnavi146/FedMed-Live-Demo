"""Deterministic extraction of medical facts from report text.

What this module is
-------------------
A parser. It finds laboratory values, dates, report types, and explicitly written
diagnoses and medications, and it records exactly where each one came from. Every
fact it reports is a quotation or a number that was literally present in the text.

What it deliberately refuses to do
----------------------------------
It does not interpret, summarise, or reason about clinical significance. That is the
job of the optional AI layer, and mixing the two would make it impossible to tell a
measured fact from a generated sentence.

Two rules that matter more than the rest
----------------------------------------
1. **A reference range is only ever the one printed in the report.** If a report
   gives a value with no range, the result carries ``reference_range=None`` and
   ``within_range=None``. It is never filled in from a built-in table of "normal"
   values, because an invented range turns an ordinary number into a false alarm (or
   worse, a false reassurance). The reader sees "no reference range given in the
   report" rather than a judgement the software is not qualified to make.
2. **An ambiguous date stays ambiguous.** ``01/02/2026`` is not silently read as
   January 2nd. The literal text is kept and the ambiguity is flagged, because a
   wrong report order would corrupt every "changed over time" comparison.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from typing import Any

# --- vocabulary ------------------------------------------------------------------

# Terms that, when they appear in a report, are explicit warnings from the clinician
# who wrote it. This list is deliberately small and literal: it is a red-flag SCAN,
# not a triage algorithm, and the analysis states that limitation wherever it is used.
RED_FLAG_TERMS: tuple[str, ...] = (
    "acute",
    "critical",
    "critically",
    "emergency",
    "immediately",
    "urgent",
    "urgently",
    "life-threatening",
    "life threatening",
    "anaphyla",
    "haemorrhage",
    "hemorrhage",
    "haemodynamic",
    "hemodynamic",
    "sepsis",
    "septic",
    "meningitis",
    "embolism",
    "infarction",
    "infarct",
    "occlusion",
    "mass effect",
    "midline shift",
    "raised intracranial",
    "increased intracranial",
    "fracture",
    "pneumothorax",
    "aortic dissection",
    "cardiac arrest",
    "stroke",
    "malignan",
    "suspicious for malignancy",
    "metastas",
    "necrosis",
    "abscess",
    "obstruction",
    "perforation",
    "tamponade",
    "pulmonary oedema",
    "pulmonary edema",
)

# Sections whose content is quoted as an explicitly documented diagnosis.
DIAGNOSIS_HEADINGS = (
    "diagnosis",
    "diagnoses",
    "impression",
    "assessment",
    "conclusion",
    "conclusions",
)
MEDICATION_HEADINGS = ("medications", "medication", "rx", "drugs", "current medications", "treatment")
HISTORY_HEADINGS = ("clinical history", "history", "indication", "reason for referral", "presenting complaint")
FINDINGS_HEADINGS = ("findings", "observation", "observations", "results", "result", "report")

# Report type is decided by the words the report uses about itself.
REPORT_TYPE_HINTS: tuple[tuple[str, str], ...] = (
    ("mri", "MRI"),
    ("magnetic resonance", "MRI"),
    ("ct ", "CT"),
    ("computed tomography", "CT"),
    ("x-ray", "X-ray"),
    ("xray", "X-ray"),
    ("radiograph", "X-ray"),
    ("ultrasound", "Ultrasound"),
    ("sonograph", "Ultrasound"),
    ("ecg", "ECG"),
    ("ekg", "ECG"),
    ("electrocardiogram", "ECG"),
    ("electroencephalogram", "EEG"),
    ("eeg", "EEG"),
    ("biopsy", "Biopsy"),
    ("histology", "Histopathology"),
    ("histopathology", "Histopathology"),
    ("cytology", "Cytology"),
    ("haematology", "Haematology"),
    ("hematology", "Haematology"),
    ("full blood count", "Haematology"),
    ("complete blood count", "Haematology"),
    ("cbc", "Haematology"),
    ("lipid", "Lipid profile"),
    ("liver function", "Liver function"),
    ("renal function", "Renal function"),
    ("kidney function", "Renal function"),
    ("thyroid", "Thyroid function"),
    ("urinalysis", "Urinalysis"),
    ("microbiology", "Microbiology"),
    ("culture", "Microbiology"),
    ("pathology", "Pathology"),
)

# --- patterns --------------------------------------------------------------------

# Labels that look like a result row but are document metadata. Parsec as a "value"
# otherwise: "Date: 2026-01-15" would become a laboratory value named "Date".
NON_ANALYTE_LABELS: frozenset[str] = frozenset(
    {
        "date",
        "dates",
        "time",
        "page",
        "ref",
        "reference",
        "sample",
        "specimen",
        "patient",
        "name",
        "id",
        "nhs",
        "mrn",
        "hospital",
        "ward",
        "clinician",
        "consultant",
        "reported by",
        "requested by",
        "lab number",
        "accession",
        "telephone",
        "tel",
        "fax",
        "address",
        "dob",
        "sex",
        "age",
        "address",
        "version",
        "printed",
        "received",
        "collected",
        "final",
        "amended",
    }
)

# A number, allowing for thousands separators and a decimal comma.
_NUMBER = r"[-+]?\d{1,3}(?:[,\s]\d{3})*(?:[.,]\d+)?|[-+]?\d+(?:[.,]\d+)?"

# Reference ranges as they are actually printed.
_RANGE_PATTERNS = (
    # 13.0 - 17.0   |   13 - 17   |  13.0–17.0
    re.compile(rf"(?P<low>{_NUMBER})\s*(?:-|–|—|to)\s*(?P<high>{_NUMBER})", re.IGNORECASE),
    # <= 5 / < 5 / up to 5
    re.compile(rf"(?:<=|<|≤|up to|below|less than)\s*(?P<high>{_NUMBER})", re.IGNORECASE),
    # >= 2 / > 2 / at least 2
    re.compile(rf"(?:>=|>|≥|at least|above|greater than)\s*(?P<low>{_NUMBER})", re.IGNORECASE),
)

_FLAG_PATTERN = re.compile(r"(?<![A-Za-z])(H|L|HH|LL|HIGH|LOW|NORMAL|ABNORMAL|\*|↑|↓)(?![A-Za-z])", re.IGNORECASE)

_UNIT_PATTERN = re.compile(
    r"(?<![A-Za-z])("
    r"mg/dL|mg/dl|g/dL|g/dl|g/L|mmol/L|umol/L|µmol/L|mol/L|"
    r"ng/mL|ng/ml|pg/mL|pg/ml|µg/L|ug/L|mcg/L|"
    r"IU/L|U/L|u/L|mIU/L|mIU/mL|"
    r"10\^?\d+/L|10\*\*\d+/L|x10\^?\d+/L|"
    r"%|fL|pg|mm/h|mmHg|bpm|kg/m2|mL/min|mL/min/1\.73m2|"
    r"s|sec|min|hours|days|"
    r"copies/mL|CFU/mL"
    r")(?![A-Za-z])"
)

_DATE_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    # Unambiguous: ISO first, because it cannot be misread.
    (re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b"), "iso"),
    (re.compile(r"\b(\d{4})/(\d{2})/(\d{2})\b"), "iso-slash"),
    # Month names are unambiguous about which field is the day.
    (re.compile(r"\b(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})\b", re.IGNORECASE), "dmy-name"),
    (re.compile(r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s+(\d{4})\b", re.IGNORECASE), "mdy-name"),
    # Ambiguous numerically, so these are flagged rather than converted.
    (re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b"), "ambiguous-slash"),
    (re.compile(r"\b(\d{1,2})\.(\d{1,2})\.(\d{4})\b"), "ambiguous-dot"),
)

_MONTHS = {
    name: index
    for index, name in enumerate(
        [
            "january", "february", "march", "april", "may", "june",
            "july", "august", "september", "october", "november", "december",
        ],
        start=1,
    )
}


# --- models ----------------------------------------------------------------------


@dataclass
class Provenance:
    """Where a fact came from. Every extracted fact carries one."""

    report_id: str
    report_name: str
    page: int | None
    line: int | None
    snippet: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "reportId": self.report_id,
            "reportName": self.report_name,
            "page": self.page,
            "line": self.line,
            "snippet": self.snippet,
        }


@dataclass
class LabValue:
    """One laboratory value as printed, with the report's own reference range."""

    name: str
    value: float | None
    value_text: str
    unit: str | None
    reference_text: str | None
    reference_low: float | None
    reference_high: float | None
    reported_flag: str | None
    within_range: bool | None
    provenance: Provenance
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "valueText": self.value_text,
            "unit": self.unit,
            "referenceText": self.reference_text,
            "referenceLow": self.reference_low,
            "referenceHigh": self.reference_high,
            "reportedFlag": self.reported_flag,
            "withinRange": self.within_range,
            "note": self.note,
            "provenance": self.provenance.as_dict(),
        }


@dataclass
class RedFlag:
    """A literal warning term found in the text, quoted in place."""

    term: str
    provenance: Provenance

    def as_dict(self) -> dict[str, Any]:
        return {"term": self.term, "provenance": self.provenance.as_dict()}


@dataclass
class DocumentedStatement:
    """A statement the report makes under one of its own headings."""

    heading: str
    text: str
    provenance: Provenance

    def as_dict(self) -> dict[str, Any]:
        return {
            "heading": self.heading,
            "text": self.text,
            "provenance": self.provenance.as_dict(),
        }


@dataclass
class ReportDate:
    literal: str
    iso: str | None
    ambiguous: bool
    note: str
    provenance: Provenance

    def as_dict(self) -> dict[str, Any]:
        return {
            "literal": self.literal,
            "iso": self.iso,
            "ambiguous": self.ambiguous,
            "note": self.note,
            "provenance": self.provenance.as_dict(),
        }


@dataclass
class ParsedReport:
    """Everything deterministically extracted from one report."""

    report_id: str
    name: str
    report_type: str | None
    report_type_evidence: str | None
    dates: list[ReportDate] = field(default_factory=list)
    labs: list[LabValue] = field(default_factory=list)
    diagnoses: list[DocumentedStatement] = field(default_factory=list)
    medications: list[DocumentedStatement] = field(default_factory=list)
    history: list[DocumentedStatement] = field(default_factory=list)
    impressions: list[DocumentedStatement] = field(default_factory=list)
    red_flags: list[RedFlag] = field(default_factory=list)
    page_count: int = 0
    characters: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "reportId": self.report_id,
            "name": self.name,
            "reportType": self.report_type,
            "reportTypeEvidence": self.report_type_evidence,
            "dates": [item.as_dict() for item in self.dates],
            "labs": [item.as_dict() for item in self.labs],
            "diagnoses": [item.as_dict() for item in self.diagnoses],
            "medications": [item.as_dict() for item in self.medications],
            "history": [item.as_dict() for item in self.history],
            "impressions": [item.as_dict() for item in self.impressions],
            "redFlags": [item.as_dict() for item in self.red_flags],
            "pageCount": self.page_count,
            "characters": self.characters,
        }


# --- helpers ---------------------------------------------------------------------


def _to_float(text: str) -> float | None:
    """Parse a printed number, tolerating a decimal comma and thousands separators."""
    cleaned = text.strip().replace(" ", "")
    # A single comma with <=2 trailing digits is a decimal comma ("10,2").
    if re.fullmatch(r"[-+]?\d{1,3}(?:\.\d{3})*,\d{1,2}", cleaned):
        cleaned = cleaned.replace(".", "").replace(",", ".")
    else:
        cleaned = cleaned.replace(",", "")
    try:
        return float(cleaned)
    except ValueError:
        return None


def _find_range(segment: str) -> tuple[str | None, float | None, float | None]:
    """The reference range printed in this segment, if there is one.

    Returns the range as it appeared plus its numeric bounds. Bounds are None when
    the report did not print them, and the caller must not substitute any.
    """
    for pattern in _RANGE_PATTERNS:
        match = pattern.search(segment)
        if not match:
            continue
        literal = match.group(0).strip()
        low_text = match.groupdict().get("low")
        high_text = match.groupdict().get("high")
        # A two-sided range whose left number is larger than the right is almost
        # certainly a date or an identifier, not a range.
        low = _to_float(low_text) if low_text else None
        high = _to_float(high_text) if high_text else None
        if low is not None and high is not None and low > high:
            continue
        return literal, low, high
    return None, None, None


def _classify_flag(flag: str | None) -> str | None:
    if not flag:
        return None
    upper = flag.strip().upper()
    if upper in {"H", "HH", "HIGH", "↑"}:
        return "high"
    if upper in {"L", "LL", "LOW", "↓"}:
        return "low"
    if upper == "NORMAL":
        return "normal"
    if upper in {"ABNORMAL", "*"}:
        return "abnormal"
    return None


def _looks_like_a_value_name(name: str) -> bool:
    """Reject obvious non-analyte "names" so the lab list stays meaningful."""
    stripped = name.strip(" .:-|")
    if len(stripped) < 2 or len(stripped) > 60:
        return False
    if not any(character.isalpha() for character in stripped):
        return False
    # Mostly digits means it is a number row, not an analyte name.
    digits = sum(1 for character in stripped if character.isdigit())
    if digits > len(stripped) / 2:
        return False
    # Metadata labels are not analytes. Without this, a line like
    # "Date: 2026-01-15" is reported as a laboratory value called "Date" with a
    # value of 2026, which would appear in the findings as a spurious abnormality.
    cleaned = re.sub(r"[^a-z ]", "", stripped.lower()).strip()
    if cleaned in NON_ANALYTE_LABELS:
        return False
    # A trailing colon means a key/value line ("Patient: John"), not a result row.
    if name.rstrip().endswith(":"):
        return False
    return True


def _parse_dates(text: str, base: Provenance) -> list[ReportDate]:
    found: list[ReportDate] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        for pattern, kind in _DATE_PATTERNS:
            match = pattern.search(line)
            if not match:
                continue
            literal = match.group(0)
            provenance = Provenance(
                report_id=base.report_id,
                report_name=base.report_name,
                page=base.page,
                line=line_number,
                snippet=line.strip()[:200],
            )
            if kind in {"iso", "iso-slash"}:
                year, month, day = int(match.group(1)), int(match.group(2)), int(match.group(3))
                try:
                    iso = date(year, month, day).isoformat()
                except ValueError:
                    continue
                found.append(
                    ReportDate(literal, iso, False, "Unambiguous ISO-ordered date.", provenance)
                )
            elif kind == "dmy-name":
                day = int(match.group(1))
                month = _MONTHS[match.group(2).lower()]
                year = int(match.group(3))
                try:
                    iso = date(year, month, day).isoformat()
                except ValueError:
                    continue
                found.append(
                    ReportDate(literal, iso, False, "Day-month-year with a month name.", provenance)
                )
            elif kind == "mdy-name":
                month = _MONTHS[match.group(1).lower()]
                day = int(match.group(2))
                year = int(match.group(3))
                try:
                    iso = date(year, month, day).isoformat()
                except ValueError:
                    continue
                found.append(
                    ReportDate(literal, iso, False, "Month-day-year with a month name.", provenance)
                )
            else:
                first, second = int(match.group(1)), int(match.group(2))
                # Day-first is only certain when a field exceeds 12.
                if first > 12 and second <= 12:
                    note = "Read as day-month-year (the first field exceeds 12)."
                    iso = f"{int(match.group(3)):04d}-{second:02d}-{first:02d}"
                    ambiguous = False
                elif second > 12 and first <= 12:
                    note = "Read as month-day-year (the second field exceeds 12)."
                    iso = f"{int(match.group(3)):04d}-{first:02d}-{second:02d}"
                    ambiguous = False
                else:
                    note = (
                        "AMBIGUOUS: both fields are 12 or less, so day-first and "
                        "month-first cannot be told apart from the text. Recorded as "
                        "written; do not rely on the ordering of reports by date until "
                        "this is confirmed."
                    )
                    iso = None
                    ambiguous = True
                found.append(ReportDate(literal, iso, ambiguous, note, provenance))
            break
    return found


def _extract_labs(text: str, base: Provenance) -> list[LabValue]:
    """Find laboratory values, using only reference ranges present in the text."""
    labs: list[LabValue] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if len(line) < 4 or len(line) > 300:
            continue

        unit_match = _UNIT_PATTERN.search(line)
        # The value is the first number that is not part of a range on this line.
        value_match = re.search(rf"(?<![\d.])(?P<value>{_NUMBER})(?![\d.])", line)
        if not value_match:
            continue

        name = line[: value_match.start()]
        if not _looks_like_a_value_name(name):
            continue

        value_text = value_match.group("value")
        value = _to_float(value_text)
        if value is None:
            continue

        tail = line[value_match.end() :]
        reference_text, low, high = _find_range(tail) or _find_range(line)
        flag_match = _FLAG_PATTERN.search(tail)
        unit = unit_match.group(0) if unit_match else None

        # Compute the verdict ONLY from a range the report printed.
        within: bool | None = None
        note = ""
        if low is not None or high is not None:
            if low is not None and high is not None:
                within = low <= value <= high
            elif high is not None:
                within = value <= high
            else:
                within = value >= low  # type: ignore[operator]
        else:
            note = (
                "No reference range was printed with this value, so no comparison was "
                "made. This is not a statement that the value is normal."
            )

        labs.append(
            LabValue(
                name=name.strip(" .:-|\t")[:80],
                value=value,
                value_text=value_text,
                unit=unit,
                reference_text=reference_text,
                reference_low=low,
                reference_high=high,
                reported_flag=_classify_flag(flag_match.group(0) if flag_match else None),
                within_range=within,
                note=note,
                provenance=Provenance(
                    report_id=base.report_id,
                    report_name=base.report_name,
                    page=base.page,
                    line=line_number,
                    snippet=line[:200],
                ),
            )
        )
    return labs


def _extract_statements(
    text: str, base: Provenance, headings: tuple[str, ...]
) -> list[DocumentedStatement]:
    """Quote the text under any of the given headings, verbatim."""
    lines = text.splitlines()
    marker = re.compile(
        r"^\s*\**\s*(" + "|".join(re.escape(item) for item in headings) + r")\s*:?\s*\**\s*$",
        re.IGNORECASE,
    )
    starts: list[tuple[int, str]] = []
    for index, line in enumerate(lines):
        candidate = line.strip().strip("*#_ ").strip()
        if not candidate or len(candidate) > 60:
            continue
        if marker.match(candidate):
            starts.append((index, candidate.rstrip(": ").strip().title()))

    statements: list[DocumentedStatement] = []
    for position, (index, heading) in enumerate(starts):
        end = starts[position + 1][0] if position + 1 < len(starts) else len(lines)
        body = "\n".join(lines[index + 1 : end]).strip()
        if not body:
            continue
        statements.append(
            DocumentedStatement(
                heading=heading,
                text=body[:4000],
                provenance=Provenance(
                    report_id=base.report_id,
                    report_name=base.report_name,
                    page=base.page,
                    line=index + 2,
                    snippet=body[:200],
                ),
            )
        )
    return statements


def _find_red_flags(text: str, base: Provenance) -> list[RedFlag]:
    """Literal warning terms, quoted with the sentence they appeared in."""
    found: list[RedFlag] = []
    lowered = text.lower()
    for term in RED_FLAG_TERMS:
        position = lowered.find(term)
        if position < 0:
            continue
        line_start = lowered.rfind("\n", 0, position) + 1
        line_end = lowered.find("\n", position)
        if line_end < 0:
            line_end = len(text)
        line = text[line_start:line_end]
        line_number = text.count("\n", 0, position) + 1
        found.append(
            RedFlag(
                term=term,
                provenance=Provenance(
                    report_id=base.report_id,
                    report_name=base.report_name,
                    page=base.page,
                    line=line_number,
                    snippet=line.strip()[:200],
                ),
            )
        )
    return found


def _detect_report_type(text: str) -> tuple[str | None, str | None]:
    """The report type, decided by the words the report uses about itself."""
    head = text[:3000].lower()
    for needle, label in REPORT_TYPE_HINTS:
        position = head.find(needle)
        if position >= 0:
            return label, head[max(0, position - 30) : position + 60].strip().replace("\n", " ")
    return None, None


# --- entry point -----------------------------------------------------------------


def parse_report(
    report_id: str,
    name: str,
    page_texts: list[str],
    *,
    characters: int = 0,
) -> ParsedReport:
    """Parse one report's text into structured, traceable facts.

    ``page_texts`` holds one entry per page so each fact can cite its page. A text
    file passes a single entry.
    """
    whole = "\n".join(page_texts)
    base = Provenance(report_id=report_id, report_name=name, page=1, line=None, snippet="")

    report = ParsedReport(
        report_id=report_id,
        name=name,
        report_type=None,
        report_type_evidence=None,
        page_count=len(page_texts) or 1,
        characters=characters or len(whole),
    )
    report.report_type, report.report_type_evidence = _detect_report_type(whole)

    seen_labs: set[tuple[str, int | None, int | None]] = set()
    for page_index, page_text in enumerate(page_texts or [whole], start=1):
        page_base = Provenance(
            report_id=report_id, report_name=name, page=page_index, line=None, snippet=""
        )
        for lab in _extract_labs(page_text, page_base):
            key = (lab.name.lower(), lab.provenance.page, lab.provenance.line)
            if key in seen_labs:
                continue
            seen_labs.add(key)
            report.labs.append(lab)

        report.dates.extend(_parse_dates(page_text, page_base))
        report.diagnoses.extend(_extract_statements(page_text, page_base, DIAGNOSIS_HEADINGS))
        report.medications.extend(_extract_statements(page_text, page_base, MEDICATION_HEADINGS))
        report.history.extend(_extract_statements(page_text, page_base, HISTORY_HEADINGS))
        report.impressions.extend(_extract_statements(page_text, page_base, FINDINGS_HEADINGS))
        report.red_flags.extend(_find_red_flags(page_text, page_base))

    # A value that is out of range in any page is the interesting one, so order by
    # out-of-range first, then by the report's own order.
    report.labs.sort(key=lambda item: (item.within_range is not False, item.name.lower()))
    report.dates.sort(key=lambda item: (item.iso is None, item.iso or ""))
    return report


def abnormal_labs(report: ParsedReport) -> list[LabValue]:
    """Values the report's own reference range places outside that range.

    Values with no printed range are excluded, because calling them abnormal would
    require a range the report did not give.
    """
    return [lab for lab in report.labs if lab.within_range is False]


def labs_without_reference(report: ParsedReport) -> list[LabValue]:
    """Values the report printed without any reference range."""
    return [lab for lab in report.labs if lab.reference_text is None]
