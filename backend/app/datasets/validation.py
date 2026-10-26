"""Validation for NIfTI image/mask pairs.

Why validation runs on the RAW files, before preprocessing
----------------------------------------------------------
The MONAI pipeline resizes the image and the label *independently* to the network
input shape (``ResizeWithPadOrCropd`` is applied to both keys). A mask on a
completely different voxel grid therefore resizes successfully and trains happily,
producing a spatially meaningless target. The same is true of two volumes with the
same shape but different voxel spacing.

Measured before this module existed, with a 32x32x32 image and a 16x24x20 mask:

    preprocess_case(...) -> SUCCEEDED, image (1,32,32,32), label (1,32,32,32)

Silently training on that is worse than failing, so every geometry check here
happens on the original headers, where the mismatch is still visible.

Format support
--------------
NIfTI only: ``.nii`` and ``.nii.gz``. These are the formats the existing pipeline
can actually read (MONAI's ``LoadImaged`` with the nibabel reader). DICOM, NRRD,
NIfTI-2, PNG/JPEG slices, PDFs and reports are rejected by name rather than
passed through, because a 2D X-ray or a radiology report is not a 3D MRI volume
and must never be treated as one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..errors import ValidationProblem
from ..imaging.paths import find_case_images, is_label_path, label_path_for

# --- Accepted formats --------------------------------------------------------
ACCEPTED_SUFFIXES: tuple[str, ...] = (".nii.gz", ".nii")

# Formats that are deliberately refused, with the reason a user needs.
REJECTED_SUFFIXES: dict[str, str] = {
    ".dcm": "DICOM is not supported. Convert the series to NIfTI first.",
    ".dicom": "DICOM is not supported. Convert the series to NIfTI first.",
    ".nrrd": "NRRD is not supported. Convert to NIfTI first.",
    ".mha": "MetaImage is not supported. Convert to NIfTI first.",
    ".nii2": "NIfTI-2 is not supported. Convert to NIfTI-1 (.nii/.nii.gz) first.",
    ".png": "A 2D image is not a 3D volume. This model segments 3D volumes only.",
    ".jpg": "A 2D image is not a 3D volume. This model segments 3D volumes only.",
    ".jpeg": "A 2D image is not a 3D volume. This model segments 3D volumes only.",
    ".pdf": "A PDF is a document, not image data. It cannot be segmented.",
    ".txt": "A text file is not image data.",
    ".csv": "A CSV is not image data.",
    ".zip": "Archives are not image data. Extract and upload the NIfTI files.",
}

# --- Limits (requirement: size limits, safe handling) ------------------------
MAX_FILE_BYTES: int = 512 * 1024 * 1024  # 512 MB per volume
MAX_TOTAL_BYTES: int = 2 * 1024 * 1024 * 1024  # 2 GB per dataset
MAX_CASES: int = 500
# The network needs room to downsample 2**stages (8) times and still have voxels.
MIN_VOXELS: int = 8
MAX_VOXELS: int = 1024

# Spacing tolerance: voxels differing by more than this fraction are a mismatch.
SPACING_RTOL: float = 0.01


@dataclass
class Issue:
    """One validation finding, attached to a case when known."""

    code: str
    severity: str  # "error" | "warning"
    message: str
    case: str | None = None

    @property
    def is_error(self) -> bool:
        return self.severity == "error"

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "case": self.case,
        }


@dataclass
class CaseReport:
    """What was found for one image/mask pair."""

    name: str
    image_path: str
    label_path: str | None = None
    image_shape: tuple[int, ...] | None = None
    label_shape: tuple[int, ...] | None = None
    spacing: tuple[float, float, float] | None = None
    label_spacing: tuple[float, float, float] | None = None
    voxels: int | None = None
    foreground_voxels: int | None = None
    foreground_fraction: float | None = None
    label_values: list[float] = field(default_factory=list)
    issues: list[Issue] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not any(issue.is_error for issue in self.issues)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "imagePath": self.image_path,
            "labelPath": self.label_path,
            "imageShape": list(self.image_shape) if self.image_shape else None,
            "labelShape": list(self.label_shape) if self.label_shape else None,
            "spacing": list(self.spacing) if self.spacing else None,
            "voxels": self.voxels,
            "foregroundVoxels": self.foreground_voxels,
            "foregroundFraction": self.foreground_fraction,
            "labelValues": self.label_values,
            "valid": self.valid,
            "issues": [issue.as_dict() for issue in self.issues],
        }


@dataclass
class DatasetValidation:
    """The complete validation result for a directory of cases."""

    path: str
    modality: str
    modality_source: str
    case_count: int
    valid_case_count: int
    total_bytes: int
    cases: list[CaseReport] = field(default_factory=list)
    issues: list[Issue] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        """True only when every case parsed and no dataset-level error exists."""
        if any(issue.is_error for issue in self.issues):
            return False
        if not self.cases:
            return False
        return all(case.valid for case in self.cases)

    @property
    def error_count(self) -> int:
        return sum(1 for issue in self._all_issues() if issue.is_error)

    @property
    def warning_count(self) -> int:
        return sum(1 for issue in self._all_issues() if not issue.is_error)

    def _all_issues(self) -> list[Issue]:
        collected = list(self.issues)
        for case in self.cases:
            collected.extend(case.issues)
        return collected

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "valid": self.valid,
            "modality": self.modality,
            "modalitySource": self.modality_source,
            "caseCount": self.case_count,
            "validCaseCount": self.valid_case_count,
            "totalBytes": self.total_bytes,
            "errorCount": self.error_count,
            "warningCount": self.warning_count,
            "issues": [issue.as_dict() for issue in self.issues],
            "cases": [case.as_dict() for case in self.cases],
            "limits": {
                "maxFileBytes": MAX_FILE_BYTES,
                "maxTotalBytes": MAX_TOTAL_BYTES,
                "maxCases": MAX_CASES,
                "minVoxels": MIN_VOXELS,
                "maxVoxels": MAX_VOXELS,
                "acceptedSuffixes": list(ACCEPTED_SUFFIXES),
            },
        }


def _describe_modality(header: Any) -> tuple[str, str]:
    """Best-effort modality, and an honest note about where it came from.

    NIfTI has no dependable modality field. Rather than guess from intensity
    statistics and present that as fact, this reports what the header literally
    contains, or says plainly that the file does not state it.
    """
    try:
        description = str(header.get("descrip", "") or "").strip()
    except Exception:  # pragma: no cover - header oddities must not fail validation
        description = ""

    if description:
        return description[:200], "the NIfTI header `descrip` field"
    return "not stated in the file", (
        "NIfTI does not carry a reliable modality field. This is not an inference "
        "from the image content."
    )


def _spacing_from_affine(affine: np.ndarray) -> tuple[float, float, float]:
    """Voxel size per axis, from the column norms of the 3x3 direction block."""
    matrix = np.asarray(affine, dtype=float)
    return tuple(float(np.linalg.norm(matrix[:3, axis])) for axis in range(3))  # type: ignore[return-value]


def _close(left: float, right: float, tolerance: float = SPACING_RTOL) -> bool:
    if left == right:
        return True
    scale = max(abs(left), abs(right))
    if scale == 0:
        return True
    return abs(left - right) / scale <= tolerance


def _as_3d(shape: tuple[int, ...]) -> tuple[int, ...] | None:
    """Reduce a NIfTI shape to 3 spatial axes, or None when it is not a 3D volume.

    A trailing axis of size 1 is a common way to store a single channel, so
    (X, Y, Z, 1) is accepted as (X, Y, Z). Anything with more than one value on a
    fourth or fifth axis is a 4D series (time, multiple echoes, multi-channel),
    which this single-volume pipeline does not handle.
    """
    if len(shape) == 3:
        return tuple(int(value) for value in shape)
    if len(shape) == 4 and shape[3] == 1:
        return tuple(int(value) for value in shape[:3])
    if len(shape) == 4 and shape[0] == 1:
        return tuple(int(value) for value in shape[1:])
    return None


def _check_axes(name: str, shape: tuple[int, ...], issues: list[Issue]) -> None:
    small = [axis for axis, size in zip("DHW", shape) if size < MIN_VOXELS]
    if small:
        issues.append(
            Issue(
                code="unsupported_shape",
                severity="error",
                case=name,
                message=(
                    f"Axis {', '.join(small)} is smaller than the minimum "
                    f"{MIN_VOXELS} voxels (shape {shape}). The network downsamples "
                    f"3 times, so every axis needs at least {MIN_VOXELS} voxels."
                ),
            )
        )
    large = [axis for axis, size in zip("DHW", shape) if size > MAX_VOXELS]
    if large:
        issues.append(
            Issue(
                code="unsupported_shape",
                severity="error",
                case=name,
                message=(
                    f"Axis {', '.join(large)} exceeds the maximum {MAX_VOXELS} voxels "
                    f"(shape {shape}). Resample before uploading."
                ),
            )
        )


def _inspect_volume(
    name: str,
    path: Path,
    issues: list[Issue],
    *,
    check_axes: bool = True,
) -> dict[str, Any] | None:
    """Read one NIfTI volume's geometry and data. Returns None when unreadable.

    ``check_axes`` is False for masks: the image already carries the size check,
    and repeating it for the mask reports the same defect twice. A mask that is
    genuinely the wrong size is caught by the shape comparison instead.
    """
    try:
        import nibabel as nib

        image = nib.load(str(path))
    except Exception as exc:
        issues.append(
            Issue(
                code="unreadable",
                severity="error",
                case=name,
                message=f"{path.name} could not be read as NIfTI: {type(exc).__name__}: {exc}",
            )
        )
        return None

    raw_shape = tuple(int(size) for size in image.shape)
    spatial = _as_3d(raw_shape)
    if spatial is None:
        issues.append(
            Issue(
                code="not_a_3d_volume",
                severity="error",
                case=name,
                message=(
                    f"{path.name} has shape {raw_shape}, which is not a single 3D "
                    "volume. This model segments one 3D volume per case; 4D series "
                    "must be split into separate volumes first."
                ),
            )
        )
        return None

    if check_axes:
        _check_axes(name, spatial, issues)

    spacing = _spacing_from_affine(image.affine)
    if any(not math.isfinite(value) or value <= 0 for value in spacing):
        issues.append(
            Issue(
                code="invalid_geometry",
                severity="error",
                case=name,
                message=f"{path.name} has a non-physical voxel size {spacing}.",
            )
        )

    info: dict[str, Any] = {
        "shape": spatial,
        "raw_shape": raw_shape,
        "spacing": spacing,
        "affine": np.asarray(image.affine, dtype=float),
        "modality": _describe_modality(image.header),
    }

    # The label's values matter, so its data is read. Images are not read at all:
    # geometry is enough, and a 512 MB float volume is not worth loading twice.
    if is_label_path(path):
        try:
            data = np.asanyarray(image.dataobj)
        except Exception as exc:
            issues.append(
                Issue(
                    code="unreadable",
                    severity="error",
                    case=name,
                    message=f"{path.name} data could not be read: {type(exc).__name__}: {exc}",
                )
            )
            return None
        values = np.unique(data)
        info["label_values"] = [float(value) for value in values[:64]]
        info["foreground_voxels"] = int(np.count_nonzero(data))
        info["voxels"] = int(data.size)
        info.pop("affine", None)
        info.pop("modality", None)
    return info


def validate_directory(
    directory: Path | str,
    *,
    expected_label_suffix: str = "_seg",
) -> DatasetValidation:
    """Validate every image/mask pair in ``directory``.

    This is the single validator used by both entry points (a server-side directory
    and the staging directory an upload lands in), so both are held to the same
    rules.
    """
    root = Path(directory)
    issues: list[Issue] = []

    if not root.exists():
        raise ValidationProblem(
            f"Dataset directory {root} does not exist.",
            code="dataset_path_missing",
        )
    if not root.is_dir():
        raise ValidationProblem(
            f"Dataset path {root} is not a directory.",
            code="dataset_path_not_a_directory",
        )

    # Refuse the formats people actually try, by name, so the message is useful.
    rejected: list[Issue] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        lowered = path.name.lower()
        if any(lowered.endswith(suffix) for suffix in ACCEPTED_SUFFIXES):
            continue
        for suffix, reason in REJECTED_SUFFIXES.items():
            if lowered.endswith(suffix):
                rejected.append(
                    Issue(
                        code="unsupported_format",
                        severity="error",
                        message=f"{path.name}: {reason}",
                    )
                )
                break
    issues.extend(rejected[:40])

    images = find_case_images(root)
    if not images:
        issues.append(
            Issue(
                code="no_cases_found",
                severity="error",
                message=(
                    f"No NIfTI image files in {root}. Each case needs "
                    f"<name>.nii.gz plus <name>{expected_label_suffix}.nii.gz."
                ),
            )
        )

    # Orphan masks: a _seg file with no matching image is usually a naming mistake.
    # Compared against the labels the found images actually expect, so this stays
    # correct now that the case scan descends into subdirectories.
    expected_labels = {label_path_for(image).resolve() for image in images}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or not is_label_path(path):
            continue
        if path.resolve() in expected_labels:
            continue
        stem = path.name[: -len(".nii.gz")] if path.name.endswith(".nii.gz") else path.name
        if path.name.endswith(".nii") and not path.name.endswith(".nii.gz"):
            stem = path.name[: -len(".nii")]
        base = stem[: -len(expected_label_suffix)]
        issues.append(
            Issue(
                code="orphan_mask",
                severity="error",
                message=(
                    f"{path.name} has no matching image. Expected an image named "
                    f"{base}.nii.gz or {base}.nii alongside it."
                ),
            )
        )

    if len(images) > MAX_CASES:
        issues.append(
            Issue(
                code="too_many_cases",
                severity="error",
                message=(
                    f"{len(images)} cases found, above the {MAX_CASES} limit for one "
                    "dataset. Split the dataset before uploading."
                ),
            )
        )

    cases: list[CaseReport] = []
    total_bytes = 0

    for image_path in images[:MAX_CASES]:
        name = image_path.name
        case_issues: list[Issue] = []
        label_path = label_path_for(image_path)

        size = image_path.stat().st_size
        total_bytes += size
        if size > MAX_FILE_BYTES:
            case_issues.append(
                Issue(
                    code="file_too_large",
                    severity="error",
                    case=name,
                    message=(
                        f"{name} is {size / 1024 / 1024:.0f} MB, above the "
                        f"{MAX_FILE_BYTES // 1024 // 1024} MB per-file limit."
                    ),
                )
            )

        report = CaseReport(name=name, image_path=str(image_path))

        image_info = _inspect_volume(name, image_path, case_issues)
        if image_info is not None:
            report.image_shape = image_info["shape"]
            report.spacing = image_info["spacing"]
            report.voxels = int(math.prod(image_info["shape"]))

        if not label_path.exists():
            case_issues.append(
                Issue(
                    code="missing_mask",
                    severity="error",
                    case=name,
                    message=(
                        f"No segmentation mask for {name}. Expected "
                        f"{label_path.name} in the same folder."
                    ),
                )
            )
        else:
            report.label_path = str(label_path)
            label_size = label_path.stat().st_size
            total_bytes += label_size
            if label_size > MAX_FILE_BYTES:
                case_issues.append(
                    Issue(
                        code="file_too_large",
                        severity="error",
                        case=name,
                        message=(
                            f"{label_path.name} is {label_size / 1024 / 1024:.0f} MB, "
                            f"above the {MAX_FILE_BYTES // 1024 // 1024} MB limit."
                        ),
                    )
                )

            label_info = _inspect_volume(
                label_path.name, label_path, case_issues, check_axes=False
            )
            if label_info is not None:
                report.label_shape = label_info["shape"]
                report.label_spacing = label_info["spacing"]
                report.label_values = label_info.get("label_values", [])
                report.foreground_voxels = label_info.get("foreground_voxels")
                if label_info.get("voxels"):
                    report.foreground_fraction = (
                        report.foreground_voxels / label_info["voxels"]
                    )

                # --- the mismatch checks the old pipeline missed -----------
                shapes_differ = (
                    image_info is not None
                    and tuple(report.image_shape or ()) != tuple(report.label_shape or ())
                )
                if shapes_differ:
                    case_issues.append(
                        Issue(
                            code="shape_mismatch",
                            severity="error",
                            case=name,
                            message=(
                                f"Image is {tuple(report.image_shape)} but the mask is "
                                f"{tuple(report.label_shape)}. The mask must be on the "
                                "same voxel grid as the image."
                            ),
                        )
                    )
                elif (
                    image_info is not None
                    and report.spacing
                    and report.label_spacing
                    and any(
                        not _close(a, b)
                        for a, b in zip(report.spacing, report.label_spacing)
                    )
                ):
                    case_issues.append(
                        Issue(
                            code="spacing_mismatch",
                            severity="error",
                            case=name,
                            message=(
                                f"Image voxel size "
                                f"{tuple(round(s, 3) for s in report.spacing)} does not "
                                f"match the mask's "
                                f"{tuple(round(s, 3) for s in report.label_spacing)}. "
                                "The mask must be co-registered with the image."
                            ),
                        )
                    )

                # --- label values the binary model cannot represent --------
                allowed = [0.0, 1.0]
                unexpected = [
                    value
                    for value in report.label_values
                    if not any(abs(value - ok) < 1e-6 for ok in allowed)
                ]
                if unexpected:
                    case_issues.append(
                        Issue(
                            code="invalid_labels",
                            severity="error",
                            case=name,
                            message=(
                                f"Mask contains values {unexpected[:8]} besides 0 and 1. "
                                "This model performs binary segmentation, so the mask "
                                "must be binary. Binarise the mask (or extract one "
                                "label) before uploading rather than relying on a "
                                "threshold to hide it."
                            ),
                        )
                    )

                if report.foreground_voxels == 0:
                    case_issues.append(
                        Issue(
                            code="empty_mask",
                            severity="warning",
                            case=name,
                            message=(
                                f"The mask for {name} contains no foreground voxels. "
                                "It will contribute no positive examples."
                            ),
                        )
                    )
                elif (
                    report.foreground_fraction is not None
                    and report.foreground_fraction > 0.9
                ):
                    case_issues.append(
                        Issue(
                            code="mask_almost_everything",
                            severity="warning",
                            case=name,
                            message=(
                                f"{report.foreground_fraction:.0%} of the mask is "
                                "foreground. Check that the image and mask are not "
                                "swapped."
                            ),
                        )
                    )

        report.issues = case_issues
        cases.append(report)

    if total_bytes > MAX_TOTAL_BYTES:
        issues.append(
            Issue(
                code="dataset_too_large",
                severity="error",
                message=(
                    f"Dataset is {total_bytes / 1024 / 1024 / 1024:.2f} GB, above the "
                    f"{MAX_TOTAL_BYTES // 1024 // 1024 // 1024} GB limit."
                ),
            )
        )

    modality = "not stated in the file"
    modality_source = (
        "NIfTI does not carry a reliable modality field. This is not an inference "
        "from the image content."
    )
    for case in cases:
        if case.image_path:
            try:
                import nibabel as nib

                label, source = _describe_modality(nib.load(case.image_path).header)
                if label != "not stated in the file":
                    modality, modality_source = label, source
                    break
            except Exception:  # pragma: no cover - already reported as unreadable
                continue

    return DatasetValidation(
        path=str(root),
        modality=modality,
        modality_source=modality_source,
        case_count=len(cases),
        valid_case_count=sum(1 for case in cases if case.valid),
        total_bytes=total_bytes,
        cases=cases,
        issues=issues,
    )
