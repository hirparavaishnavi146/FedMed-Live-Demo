"""Assemble a case summary from mixed uploaded files.

What this produces, and what it must never be mistaken for
---------------------------------------------------------
A **case summary**: an account of what was received, what could be analysed, what
could not, and why. It is not a diagnosis and it does not contain an
interpretation.

That boundary is enforced by the code, not just by a label. The only model here is
a 3D U-Net trained on procedurally generated synthetic phantoms; its output on a
real scan is not clinically meaningful, and a 2D X-ray or a PDF report is outside
its input domain entirely. So instead of manufacturing a "suggestion" from models
that cannot support one, every input type is mapped to what the project can
genuinely do with it, and the rest is stated as not possible.
"""

from __future__ import annotations

import importlib
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..datasets import validation as validation_module
from ..datasets.storage import staging
from ..errors import ApiError, NotFoundError
from . import classify as classify_module
from . import reports

# Where a trained checkpoint lives. The same directory the training job writes to.
CHECKPOINT_DIR = Path("experiments/imaging/checkpoints")
EXPERIMENT_FILE = Path("experiments/imaging/experiment.json")

NON_DIAGNOSTIC_NOTICE = (
    "This is a technical case summary produced by a research prototype. It is NOT a "
    "diagnosis, NOT a clinical interpretation, and NOT a medical opinion. No output "
    "here has been validated for clinical use, and no finding should be read from it."
)

MODEL_PROVENANCE = (
    "The only model in this project is a 3D U-Net trained exclusively on "
    "procedurally generated synthetic phantoms. Its output on a real MRI volume is "
    "not meaningful, because it has never seen one."
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class CaseFile:
    name: str
    kind: str
    size_bytes: int
    method: str
    note: str
    analysis: dict[str, Any] = field(default_factory=dict)
    analysed: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "sizeBytes": self.size_bytes,
            "method": self.method,
            "note": self.note,
            "analysed": self.analysed,
            "analysis": self.analysis,
        }


@dataclass
class CaseRecord:
    id: str
    name: str
    created_at: datetime
    root: Path
    staged: bool
    files: list[CaseFile] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)


# --- segmentation -----------------------------------------------------------------


def _checkpoint_config() -> dict[str, Any] | None:
    """The architecture the saved checkpoint expects, or None when there is none.

    The checkpoint records a parameter count but not the feature widths, so the
    widths are read from the experiment that produced it. A mismatch is refused
    rather than loaded, because silently loading wrong-shaped weights is how a
    model appears to work while predicting noise.
    """
    best = CHECKPOINT_DIR / "best.pt"
    if not best.is_file():
        return None
    features = (8, 16, 32)
    target_shape = (32, 32, 32)
    parameter_count = None
    try:
        import json

        if EXPERIMENT_FILE.is_file():
            experiment = json.loads(EXPERIMENT_FILE.read_text(encoding="utf-8"))
            config = experiment.get("experiment") or {}
            if config.get("features"):
                features = tuple(int(value) for value in config["features"])
            if config.get("targetShape"):
                target_shape = tuple(int(value) for value in config["targetShape"])
            parameter_count = (experiment.get("model") or {}).get("parameterCount")
    except Exception:  # noqa: BLE001 - fall back to the defaults above
        pass
    return {
        "path": best,
        "features": features,
        "target_shape": target_shape,
        "parameter_count": parameter_count,
    }


def segment(image_path: Path, label_path: Path | None) -> dict[str, Any]:
    """Segment one volume with the trained U-Net. Real computation, real limits.

    Returns a result that always carries the model's provenance, so a mask can
    never be read as a clinical finding. When a reference mask is supplied the Dice
    is a genuine measurement against it.
    """
    try:
        import torch
    except ImportError:
        return {
            "available": False,
            "reason": (
                "PyTorch is not installed in this deployment, so inference cannot run "
                "here. The dashboard stays usable."
            ),
        }

    config = _checkpoint_config()
    if config is None:
        return {
            "available": False,
            "reason": (
                "No trained checkpoint was found in "
                f"{CHECKPOINT_DIR}. Train the U-Net first (Datasets page, or "
                "python -m app.imaging.pipeline); an untrained model would produce a "
                "meaningless mask, so none is produced."
            ),
        }

    preprocess = importlib.import_module("..imaging.preprocess", __package__)
    train_module = importlib.import_module("..imaging.train", __package__)
    unet_module = importlib.import_module("..imaging.unet", __package__)
    metrics = importlib.import_module("..imaging.metrics", __package__)

    preprocess_config = preprocess.PreprocessConfig(target_shape=tuple(config["target_shape"]))
    processed = preprocess.preprocess_case(image_path, label_path, preprocess_config)
    image, reference = preprocess.as_network_input(processed)

    model = unet_module.build_unet(seed=0, features=tuple(config["features"]))
    try:
        train_module.load_checkpoint(config["path"], model)
    except Exception as exc:  # noqa: BLE001
        return {
            "available": False,
            "reason": (
                "The checkpoint could not be loaded into the model, so no inference "
                f"was run: {type(exc).__name__}: {exc}"
            ),
        }

    model.eval()
    with torch.no_grad():
        logits = model(image.unsqueeze(0))

    threshold = 0.5
    probabilities = metrics.to_probabilities(logits, from_logits=True)
    mask = (probabilities >= threshold).float()

    voxels = int(mask.sum().item())
    total = int(mask.numel())
    spacing = processed.get("spacing") or (1.0, 1.0, 1.0)
    voxel_volume = float(spacing[0] * spacing[1] * spacing[2])

    result: dict[str, Any] = {
        "available": True,
        "checkpoint": str(config["path"]),
        "architecture": "UNet3D",
        "features": list(config["features"]),
        "parameters": config["parameter_count"],
        "threshold": threshold,
        "inputShape": list(processed.get("shape") or ()),
        "voxelSpacingMm": [round(float(value), 4) for value in spacing],
        "foregroundVoxels": voxels,
        "foregroundFraction": voxels / total if total else 0.0,
        "volumeMm3": voxels * voxel_volume,
        "referenceMaskSupplied": label_path is not None,
        "diceAgainstReference": None,
        "trainedOn": "procedurally generated synthetic phantoms",
        "meaningfulOnRealData": False,
        "provenance": MODEL_PROVENANCE,
    }

    if reference is not None:
        # A real measurement against the mask the user supplied. Its value is only
        # as meaningful as the model is on that data, which `meaningfulOnRealData`
        # states separately.
        result["diceAgainstReference"] = float(
            metrics.dice_score(logits, reference.unsqueeze(0), threshold=threshold, from_logits=True)
        )

    return result


# --- per-file analysis ------------------------------------------------------------


def _analyse_mri(case: CaseRecord, path: Path) -> dict[str, Any]:
    """Validate the volume (reusing the dataset validator) and try to segment it."""
    analysis: dict[str, Any] = {}

    # Reuse the one validator rather than a second set of geometry rules.
    try:
        validation = validation_module.validate_directory(path.parent)
    except Exception as exc:  # noqa: BLE001
        analysis["validation"] = {
            "available": False,
            "reason": f"The volume could not be validated: {type(exc).__name__}: {exc}",
        }
        return analysis

    case_report = next(
        (item for item in validation.cases if item.name == path.name), None
    )
    if case_report is None:
        analysis["validation"] = {
            "available": False,
            "reason": "The volume was not found in the validation pass.",
        }
        return analysis

    issues = [issue.as_dict() for issue in case_report.issues]
    # A missing mask is expected when segmenting a NEW scan, so it is reported but
    # does not block inference. Every other error does.
    blocking = [
        issue
        for issue in issues
        if issue["severity"] == "error" and issue["code"] != "missing_mask"
    ]

    analysis["validation"] = {
        "available": True,
        "valid": case_report.valid,
        "blocking": blocking,
        "issues": issues,
        "imageShape": list(case_report.image_shape or ()),
        "labelShape": list(case_report.label_shape or ()),
        "spacing": [round(float(value), 4) for value in (case_report.spacing or ())],
        "labelValues": case_report.label_values,
        "note": (
            "A segmentation mask was not supplied with this volume, which is normal "
            "for a new scan. Inference can still run."
            if any(issue["code"] == "missing_mask" for issue in issues)
            else "Image and mask were both supplied and were checked against each other."
        ),
    }

    if blocking:
        analysis["segmentation"] = {
            "available": False,
            "reason": (
                f"Inference was not run: the volume has {len(blocking)} blocking "
                "validation error(s) that must be fixed first."
            ),
        }
        return analysis

    label_path = None
    if case_report.label_path and Path(case_report.label_path).is_file():
        label_path = Path(case_report.label_path)
    analysis["segmentation"] = segment(path, label_path)
    return analysis


def _analyse_image_2d(path: Path) -> dict[str, Any]:
    """Describe a 2D image. It is NOT interpreted, and no model is applied to it."""
    try:
        from PIL import Image
    except ImportError:
        return {
            "available": False,
            "reason": "Image support is not installed in this deployment (Pillow is missing).",
        }

    try:
        with Image.open(path) as image:
            width, height = image.size
            mode = image.mode
            image_format = image.format
    except Exception as exc:  # noqa: BLE001
        return {
            "available": False,
            "reason": f"The image could not be opened: {type(exc).__name__}: {exc}",
        }

    return {
        "available": True,
        "width": width,
        "height": height,
        "mode": mode,
        "format": image_format,
        "pixelSpacing": None,
        "modality": "2D projection image (not stated in the file)",
        "interpretationSupported": False,
        "reason": (
            "This is a 2D image. Every model in this project consumes 3D volumes, so "
            "no analysis was performed. A 2D X-ray is not a 3D MRI volume and the two "
            "are not interchangeable inputs."
        ),
    }


def _analyse_report(path: Path) -> dict[str, Any]:
    extract = reports.extract(path)
    payload = extract.as_dict()
    payload["interpretationSupported"] = False
    payload["reason"] = (
        "Text is quoted verbatim. It is not summarised, classified or turned into an "
        "app-generated impression: the words shown are the document's own."
        if extract.status is reports.TextStatus.extracted
        else "Nothing could be quoted from this document."
    )
    return payload


def build(case_id: str, name: str, root: Path, staged: bool) -> CaseRecord:
    """Classify and analyse every file in a staged case directory."""
    case = CaseRecord(
        id=case_id, name=name, created_at=_utc_now(), root=root, staged=staged
    )

    files = sorted(entry for entry in root.iterdir() if entry.is_file())
    if not files:
        raise ApiError(
            "No files were supplied for this case.", code="case_empty", status_code=422
        )

    for path in files:
        classified = classify_module.classify(path)
        record = CaseFile(
            name=path.name,
            kind=classified.kind.value,
            size_bytes=path.stat().st_size,
            method=classified.method,
            note=classified.note,
        )

        if classified.kind is classify_module.FileKind.mri_volume:
            record.analysis = _analyse_mri(case, path)
            # "Analysed" means a model ran, not merely that the file was read.
            record.analysed = bool(
                (record.analysis.get("segmentation") or {}).get("available")
            )
        elif classified.kind is classify_module.FileKind.image_2d:
            record.analysis = _analyse_image_2d(path)
            record.analysed = False
        elif classified.kind in (
            classify_module.FileKind.report_pdf,
            classify_module.FileKind.report_text,
        ):
            record.analysis = _analyse_report(path)
            record.analysed = (
                record.analysis.get("status") == reports.TextStatus.extracted.value
            )
        else:
            record.analysis = {
                "interpretationSupported": False,
                "reason": classified.note,
            }
            record.analysed = False

        case.files.append(record)

    case.reasons = [
        "A case summary is not a diagnosis.",
        MODEL_PROVENANCE,
        "2D images and PDF reports are outside every model's input domain in this project.",
    ]
    return case


def summary(case: CaseRecord) -> dict[str, Any]:
    """Shape a case for the API. Explicit about counts and about what was skipped."""
    from ..imaging.paths import is_label_path

    analysed = [item for item in case.files if item.analysed]
    volumes = [
        item
        for item in case.files
        if item.kind == classify_module.FileKind.mri_volume.value
        and not is_label_path(Path(item.name))
    ]
    # A segmentation mask is a NIfTI volume too, but counting it as an image would
    # overstate how many scans were supplied.
    masks = [
        item
        for item in case.files
        if item.kind == classify_module.FileKind.mri_volume.value
        and is_label_path(Path(item.name))
    ]
    reports_found = [
        item
        for item in case.files
        if item.kind
        in (
            classify_module.FileKind.report_pdf.value,
            classify_module.FileKind.report_text.value,
        )
    ]
    two_d = [item for item in case.files if item.kind == classify_module.FileKind.image_2d.value]

    return {
        "caseId": case.id,
        "name": case.name,
        "createdAt": case.created_at,
        "fileCount": len(case.files),
        "analysedCount": len(analysed),
        "mriVolumeCount": len(volumes),
        "maskCount": len(masks),
        "reportCount": len(reports_found),
        "image2dCount": len(two_d),
        "files": [item.as_dict() for item in case.files],
        "interpretationSupported": False,
        "interpretationReason": (
            "No component of this project can read a 2D X-ray or a written report, and "
            "the 3D U-Net was trained only on synthetic phantoms. Producing a "
            "suggestion from these inputs would mean inventing it, so none is produced. "
            "What is offered is an account of what each file is and what could "
            "genuinely be done with it."
        ),
        "notice": NON_DIAGNOSTIC_NOTICE,
        "reasons": case.reasons,
        "capabilities": capabilities(),
        "staged": case.staged,
    }


def capabilities() -> dict[str, Any]:
    """The honest capability matrix for each input type."""
    checkpoint = _checkpoint_config()
    return {
        "mriVolume": {
            "accepted": [".nii.gz", ".nii"],
            "canValidate": True,
            "canSegment": checkpoint is not None,
            "segmentationMeaningfulOnRealData": False,
            "note": (
                "Geometry, shape and label values are checked. Segmentation runs only "
                "if a checkpoint exists, and its output is not clinically meaningful: "
                "the model has only ever seen synthetic phantoms."
            ),
        },
        "reportPdf": {
            "accepted": [".pdf"],
            "canExtractText": True,
            "canInterpret": False,
            "note": (
                "Text with a text layer is quoted verbatim, including the document's "
                "own headings. A scanned PDF has no text layer and cannot be read; OCR "
                "is not implemented."
            ),
        },
        "reportText": {
            "accepted": [".txt", ".md", ".text"],
            "canExtractText": True,
            "canInterpret": False,
            "note": "Quoted verbatim, including the document's own headings.",
        },
        "image2d": {
            "accepted": [".png", ".jpg", ".jpeg", ".bmp", ".tif"],
            "canDescribe": True,
            "canInterpret": False,
            "note": (
                "Dimensions, colour mode and format are reported. Every model in this "
                "project consumes 3D volumes, so a 2D X-ray is never analysed as if it "
                "were one, and no suggestion is produced from it."
            ),
        },
        "dicom": {
            "accepted": [],
            "canRead": False,
            "note": "DICOM series are not supported. Convert to NIfTI first.",
        },
    }


# --- registry ---------------------------------------------------------------------


class CaseRegistry:
    """In-process store of analysed cases. Staged files are swept by age."""

    def __init__(self) -> None:
        self._cases: dict[str, CaseRecord] = {}

    def add(self, case: CaseRecord) -> CaseRecord:
        self._cases[case.id] = case
        return case

    def get(self, case_id: str) -> CaseRecord:
        case = self._cases.get(case_id)
        if case is None:
            raise NotFoundError(f"No case with id {case_id!r}.", code="case_not_found")
        return case

    def list(self) -> list[CaseRecord]:
        return sorted(self._cases.values(), key=lambda item: item.created_at, reverse=True)

    def remove(self, case_id: str) -> bool:
        case = self.get(case_id)
        del self._cases[case_id]
        if case.staged:
            return staging.discard(case.root.name)
        return True

    def clear(self) -> None:
        self._cases.clear()


registry = CaseRegistry()


def _default_case_name() -> str:
    """A readable name for a case the user did not name.

    Falls back to the staging session id otherwise, which reads as
    "ds-d90977462562" and tells the user nothing.
    """
    return f"case {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')}"


def analyse_staged(session_id: str, name: str | None = None) -> CaseRecord:
    """Analyse a staged upload session as one case."""
    session = staging.session(session_id)
    if session is None:
        raise NotFoundError(
            f"No upload session {session_id!r}. It may have expired; staging is temporary.",
            code="upload_session_not_found",
        )
    case = build(
        case_id=f"case-{uuid.uuid4().hex[:12]}",
        name=name or _default_case_name(),
        root=session.path,
        staged=True,
    )
    case.extra["sessionId"] = session_id
    return registry.add(case)


def analyse_directory(path: str, name: str | None = None) -> CaseRecord:
    """Analyse a directory already on this machine as one case."""
    root = Path(path).expanduser()
    if not root.exists() or not root.is_dir():
        raise NotFoundError(
            f"Directory {root} does not exist.", code="case_path_missing"
        )
    case = build(
        case_id=f"case-{uuid.uuid4().hex[:12]}",
        name=name or root.name,
        root=root,
        staged=False,
    )
    return registry.add(case)
