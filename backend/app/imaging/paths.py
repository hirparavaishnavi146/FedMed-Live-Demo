"""Case path naming: pure filesystem logic, no imaging stack.

WHY THIS MODULE EXISTS
    These helpers used to live in ``synthetic.py``, which imports nibabel and numpy.
    The dataset validator and the case-intake classifier need nothing more than the
    ``_seg`` naming convention, but importing them from ``synthetic`` executed
    ``app/imaging/__init__.py`` as well — and that eagerly imports the torch-backed
    modules. On the deployed web tier, where torch, nibabel and MONAI are deliberately
    not installed (see backend/requirements-web.txt), that chain died at startup with
    ``ModuleNotFoundError: No module named 'torch'`` and the service exited with
    status 1.

    Path naming is not an imaging concern, so it lives here, where importing it pulls
    in nothing. ``app/imaging/synthetic.py`` re-exports these names, so existing
    callers that import them from there keep working.

The convention: for ``case_000.nii.gz`` the label is ``case_000_seg.nii.gz``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

#: The suffixes a case file may use. Order matters: check .nii.gz before .nii.
NIFTI_SUFFIXES: tuple[str, ...] = (".nii.gz", ".nii")


def label_path_for(image_path: Path) -> Path:
    """`case_000.nii.gz` -> `case_000_seg.nii.gz`, preserving the .nii.gz suffix."""
    image_path = Path(image_path)
    name = image_path.name
    for suffix in NIFTI_SUFFIXES:
        if name.endswith(suffix):
            return image_path.with_name(name[: -len(suffix)] + "_seg" + suffix)
    raise ValueError(f"unexpected image suffix in {name!r}; expected .nii or .nii.gz.")


def is_label_path(path: Path) -> bool:
    """True when a path names a label rather than an image."""
    name = Path(path).name
    stem = name[: -len(".nii.gz")] if name.endswith(".nii.gz") else (
        name[: -len(".nii")] if name.endswith(".nii") else name
    )
    return stem.endswith("_seg")


def find_case_images(directory: Path) -> list[Path]:
    """Image paths under a directory, with label files excluded.

    Shared by every caller that scans for cases. It exists as one function because
    a glob returns labels as well, and a caller that forgets to filter them hands
    ``case_000_seg.nii.gz`` to `label_path_for`, which then looks for
    ``case_000_seg_seg.nii.gz``. That failure is confusing enough to be worth
    preventing in one place.

    Both ``.nii.gz`` and ``.nii`` are found, and subdirectories are searched, so
    this matches the formats the dataset validator advertises. A single-suffix,
    single-level scan would leave the guide promising something the scanner did
    not deliver.
    """
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"No such directory: {directory}.")

    found: list[Path] = []
    seen: set[Path] = set()
    for pattern in ("*.nii.gz", "*.nii"):
        for path in directory.rglob(pattern):
            if is_label_path(path) or path in seen:
                continue
            # A "_seg.nii" label is also matched by "*.nii"; is_label_path covers it.
            seen.add(path)
            found.append(path)
    return sorted(found)


def split_cases(
    paths: list[Path], validation_fraction: float, seed: int = 0
) -> tuple[list[Path], list[Path]]:
    """Split case paths into (train, validation), deterministically.

    Splitting is done by **case**, not by patch: patches from one volume must not
    appear in both splits, or validation would measure memory of the training
    volume rather than generalisation.
    """
    if not paths:
        return [], []
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError(
            f"validation_fraction must be in [0, 1), got {validation_fraction}."
        )

    ordered = sorted(paths)
    rng = np.random.default_rng(seed)
    shuffled = list(ordered)
    rng.shuffle(shuffled)

    validation_count = int(round(len(shuffled) * validation_fraction))
    # Always leave at least one training case when a split is requested.
    validation_count = min(validation_count, max(0, len(shuffled) - 1))
    return shuffled[validation_count:], shuffled[:validation_count]


__all__ = [
    "NIFTI_SUFFIXES",
    "find_case_images",
    "is_label_path",
    "label_path_for",
    "split_cases",
]
