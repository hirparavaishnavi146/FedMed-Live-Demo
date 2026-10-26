"""Case intake: classify mixed uploads and report what can and cannot be done.

This package deliberately contains no diagnosis logic. Its job is to say what each
file IS and what the project can genuinely do with it, so that a "suggestion" is
never fabricated from a model that cannot support one.
"""

from .classify import FileKind, classify as classify_file
from .intake import (
    MODEL_PROVENANCE,
    NON_DIAGNOSTIC_NOTICE,
    CaseFile,
    CaseRecord,
    CaseRegistry,
    analyse_directory,
    analyse_staged,
    capabilities,
    registry,
    segment,
    summary,
)
from .reports import ReportExtract, ReportSection, TextStatus, extract

__all__ = [
    "MODEL_PROVENANCE",
    "NON_DIAGNOSTIC_NOTICE",
    "CaseFile",
    "CaseRecord",
    "CaseRegistry",
    "FileKind",
    "ReportExtract",
    "ReportSection",
    "TextStatus",
    "analyse_directory",
    "analyse_staged",
    "capabilities",
    "classify_file",
    "extract",
    "registry",
    "segment",
    "summary",
]
