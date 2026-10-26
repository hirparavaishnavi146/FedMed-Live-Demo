"""Medical-imaging model: PyTorch + MONAI 3D U-Net for volumetric MRI segmentation.

WHAT THIS PACKAGE IS
    A standalone, testable segmentation pipeline: MONAI preprocessing, a
    configurable 3D U-Net, a reproducible training loop with checkpoints, and
    metrics (Dice and others) computed from actual predictions.

WHAT IT IS NOT
    It is not a clinically validated diagnostic system, and nothing it produces is
    a clinical result. The default data is procedurally generated synthetic
    phantoms, not scans of anything. See docs/IMAGING.md for the limitations, the
    compute requirements, and the notes on obtaining real research data.

Module map:
    unet.py       configurable 3D U-Net, with the tensor-shape contract documented
    preprocess.py MONAI transform pipeline: orientation, spacing, intensity, resize
    dataset.py    case-level splits, batch iteration, dataset provenance
    train.py      training loop, validation, metrics integration, checkpoints
    metrics.py    Dice / IoU / sensitivity / specificity from real predictions
    paths.py      case path naming (``_seg`` convention) — NO imaging dependencies
    synthetic.py  procedural phantom volumes written as real NIfTI files
    pipeline.py   end-to-end standalone experiment, runnable from the command line

LAZY IMPORTS, AND WHY IT MATTERS
    This package used to import every submodule eagerly, so ``import app.imaging.paths``
    — a function of filenames — also dragged in torch, MONAI and nibabel, because the
    package ``__init__`` runs first. On the deployed web tier those are deliberately not
    installed (see backend/requirements-web.txt), so the whole API died at startup with
    ``ModuleNotFoundError: No module named 'torch'`` and the service exited with status
    1, all because a dataset validator wanted to know how labels are named.

    The names below are therefore resolved on first access (PEP 562) rather than at
    import. ``from app.imaging import build_unet`` still works exactly as before; what
    changes is that merely importing a lightweight submodule no longer pays for the ML
    stack, and cannot fail without it.
"""

from __future__ import annotations

import importlib

#: Public name -> the submodule that defines it.
_LAZY: dict[str, str] = {
    # model
    "UNet3D": ".unet",
    "build_unet": ".unet",
    "parameter_count": ".unet",
    # preprocessing
    "PreprocessConfig": ".preprocess",
    "preprocess_case": ".preprocess",
    # data
    "CaseDataset": ".dataset",
    "PreparedCase": ".dataset",
    "build_site_datasets": ".dataset",
    "dataset_summary": ".dataset",
    # phantoms
    "CLIENT_PROFILES": ".synthetic",
    "PhantomProfile": ".synthetic",
    "generate_site": ".synthetic",
    "generate_all_sites": ".synthetic",
    # path naming (no imaging dependencies at all)
    "find_case_images": ".paths",
    "is_label_path": ".paths",
    "label_path_for": ".paths",
    "split_cases": ".paths",
    # training
    "TrainConfig": ".train",
    "TrainingResult": ".train",
    "EpochRecord": ".train",
    "train_model": ".train",
    "evaluate": ".train",
    "save_checkpoint": ".train",
    "load_checkpoint": ".train",
    # metrics
    "dice_score": ".metrics",
    "iou_score": ".metrics",
    "sensitivity": ".metrics",
    "specificity": ".metrics",
    "confusion_counts": ".metrics",
    "segmentation_report": ".metrics",
}

__all__ = [
    # model
    "UNet3D",
    "build_unet",
    "parameter_count",
    # preprocessing
    "PreprocessConfig",
    "preprocess_case",
    # data
    "CaseDataset",
    "PreparedCase",
    "build_site_datasets",
    "dataset_summary",
    "CLIENT_PROFILES",
    "PhantomProfile",
    "generate_site",
    "generate_all_sites",
    # path naming
    "find_case_images",
    "is_label_path",
    "label_path_for",
    "split_cases",
    # training
    "TrainConfig",
    "TrainingResult",
    "EpochRecord",
    "train_model",
    "evaluate",
    "save_checkpoint",
    "load_checkpoint",
    # metrics
    "dice_score",
    "iou_score",
    "sensitivity",
    "specificity",
    "confusion_counts",
    "segmentation_report",
]


def __getattr__(name: str) -> object:
    """Resolve a public name on first access, importing only the module that has it."""
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = importlib.import_module(module_name, __name__)
    value = getattr(module, name)
    # Cache it, so later lookups are ordinary attribute reads rather than re-imports.
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY))
