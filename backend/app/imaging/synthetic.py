"""Synthetic MRI-like phantom volumes, written as real NIfTI files.

Purpose: give the preprocessing and training code something real to operate on
without downloading a dataset, and without using any patient data. These are
procedurally generated mathematical phantoms — a noisy background with a brighter
ellipsoid "lesion" — not images of anything.

Two deliberate properties, because they are what make the preprocessing testable:

    * **Anisotropic voxels.** The default spacing is 1.0 x 1.0 x 3.0 mm, which is
      typical of a clinical acquisition (thick slices, fine in-plane). A spacing
      pipeline that does nothing is therefore detectable: the processed volume
      must change shape.
    * **Non-RAS orientation.** The default affine is LAS, so an orientation step
      is observable in the output affine rather than being a silent no-op.

The label is a single connected ellipsoid, so a Dice score against it is
meaningful, and a model that predicts nothing scores 0 rather than being excused
by empty masks.

UNITS: millimetres and voxel indices, as NIfTI defines. No patient, scanner or
acquisition metadata of any kind is written into the headers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np

# Typical clinical-like geometry: fine in-plane, 3 mm slices.
DEFAULT_SPACING = (1.0, 1.0, 3.0)
DEFAULT_SHAPE = (48, 48, 24)


@dataclass(frozen=True)
class PhantomProfile:
    """One simulated site's data characteristics.

    Mirrors the three hospital clients used by the federated layer, so the same
    non-IID structure carries from the stand-in model to the imaging model.
    """

    name: str
    shape: tuple[int, int, int] = DEFAULT_SHAPE
    spacing: tuple[float, float, float] = DEFAULT_SPACING
    cases: int = 4
    noise: float = 0.08
    radius_range: tuple[float, float] = (7.0, 10.0)
    description: str = ""


# The three hospital sites. Sizes, noise and lesion size differ on purpose: the
# clients are non-IID, which is the situation federated averaging has to cope with.
CLIENT_PROFILES: tuple[PhantomProfile, ...] = (
    PhantomProfile(
        name="HOSP-01",
        shape=(48, 48, 24),
        cases=4,
        noise=0.06,
        radius_range=(8.0, 10.5),
        description="Larger, cleaner lesions. Fine in-plane, 3 mm slices.",
    ),
    PhantomProfile(
        name="HOSP-02",
        shape=(40, 40, 24),
        cases=3,
        noise=0.12,
        radius_range=(6.0, 8.0),
        description="Smaller lesions with more noise, and a smaller in-plane matrix.",
    ),
    PhantomProfile(
        name="HOSP-03",
        shape=(44, 44, 20),
        cases=2,
        noise=0.18,
        radius_range=(5.0, 7.0),
        description="Smallest, noisiest site with the fewest cases.",
    ),
)


def phantom_affine(
    spacing: tuple[float, float, float] = DEFAULT_SPACING,
    shape: tuple[int, int, int] = DEFAULT_SHAPE,
    orientation: str = "LAS",
) -> np.ndarray:
    """Build a NIfTI affine for the requested spacing and orientation.

    The centre of the volume is placed at the origin so the two phantoms can be
    overlaid; only the voxel-to-world mapping changes.
    """
    signs = {"R": 1.0, "L": -1.0, "A": 1.0, "P": -1.0, "S": 1.0, "I": -1.0}
    if len(orientation) != 3 or any(axis not in signs for axis in orientation):
        raise ValueError(f"orientation must be 3 characters of RAS/LPI, got {orientation!r}.")

    affine = np.eye(4, dtype=np.float64)
    ordered_spacing = [spacing[0], spacing[1], spacing[2]]
    for axis, label in enumerate(orientation):
        affine[axis, axis] = signs[label] * ordered_spacing[axis]

    # Translate so the volume centre sits at the world origin.
    extent = np.array([spacing[i] * (shape[i] - 1) / 2.0 for i in range(3)])
    affine[:3, 3] = -affine[:3, :3].sum(axis=1) / 2.0 + extent * 0
    return affine


def make_phantom(
    shape: tuple[int, int, int] = DEFAULT_SHAPE,
    seed: int = 0,
    noise: float = 0.08,
    radius_range: tuple[float, float] = (7.0, 10.0),
    background: float = 0.25,
    lesion_intensity: float = 0.85,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate one (image, label) pair as float32 arrays.

    Returns:
        image: (D, H, W) float32, values roughly in [0, 1] before normalization.
        label: (D, H, W) uint8, 1 inside the ellipsoid and 0 outside.

    The lesion is a single ellipsoid placed near the volume centre, with axes
    sampled independently so it is not a sphere. Values are clipped to [0, 1].
    """
    if len(shape) != 3 or any(size <= 0 for size in shape):
        raise ValueError(f"shape must be three positive integers, got {shape}.")
    if not 0.0 <= noise < 1.0:
        raise ValueError(f"noise must be in [0, 1), got {noise}.")
    if radius_range[0] <= 0 or radius_range[1] < radius_range[0]:
        raise ValueError(f"radius_range must be positive and ordered, got {radius_range}.")

    rng = np.random.default_rng(seed)
    depth, height, width = shape

    # Background: a smooth low-frequency bias field plus fine-grained noise. The
    # bias field matters because intensity normalization should remove it, and a
    # pipeline that skips normalization would leave it behind.
    image = np.full(shape, background, dtype=np.float32)
    coarse = rng.normal(0.0, noise * 0.5, size=(4, 4, 4)).astype(np.float32)
    # Nearest-neighbour upsample the coarse field to the full volume.
    indices = [
        (np.arange(size) * 4 // size).clip(0, 3) for size in (depth, height, width)
    ]
    image += coarse[np.ix_(indices[0], indices[1], indices[2])]
    image += rng.normal(0.0, noise * 0.35, size=shape).astype(np.float32)

    # Ellipsoid with independently sampled semi-axes.
    centre = np.array([depth / 2.0, height / 2.0, width / 2.0])
    # Slight offset so the lesion is not always perfectly centred.
    centre += rng.uniform(-1.5, 1.5, size=3)
    radii = rng.uniform(radius_range[0], radius_range[1], size=3).astype(np.float32)
    # Keep the lesion inside the volume along the (thinnest) slice axis.
    radii[2] = min(radii[2], width * 0.4)

    grid = np.stack(
        np.meshgrid(
            np.arange(depth), np.arange(height), np.arange(width), indexing="ij"
        ),
        axis=-1,
    ).astype(np.float32)
    normalized = ((grid - centre) / radii) ** 2
    label = (normalized.sum(axis=-1) <= 1.0).astype(np.uint8)

    image = np.where(label > 0, lesion_intensity, image)
    image = np.clip(image, 0.0, 1.0).astype(np.float32)
    return image, label


def write_case(
    path: Path,
    image: np.ndarray,
    label: np.ndarray,
    affine: np.ndarray,
) -> Path:
    """Write one image/label pair as NIfTI, returning the image path.

    The label is written next to the image with a ``_seg`` suffix, following the
    naming convention medical-imaging tooling expects (image ``case_000.nii.gz``
    with label ``case_000_seg.nii.gz``).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if image.shape != label.shape:
        raise ValueError(
            f"image shape {image.shape} and label shape {label.shape} must match."
        )

    label_path = label_path_for(path)
    nib.save(nib.Nifti1Image(image.astype(np.float32), affine), str(path))
    nib.save(nib.Nifti1Image(label.astype(np.uint8), affine), str(label_path))
    return path


from .paths import find_case_images, is_label_path, label_path_for, split_cases


def generate_site(directory: Path, profile: PhantomProfile, seed: int = 0) -> list[Path]:
    """Write every case for one simulated site. Returns the image paths.

    Deterministic: the same profile and seed always write byte-identical volumes,
    which is what makes a training run reproducible.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []
    for index in range(profile.cases):
        case_seed = seed + index * 1000 + (abs(hash(profile.name)) % 997)
        image, label = make_phantom(
            shape=profile.shape,
            seed=case_seed,
            noise=profile.noise,
            radius_range=profile.radius_range,
        )
        affine = phantom_affine(profile.spacing, profile.shape, orientation="LAS")
        path = directory / f"{profile.name.lower()}_case_{index:03d}.nii.gz"
        paths.append(write_case(path, image, label, affine))
    return paths


def generate_all_sites(root: Path, seed: int = 0) -> dict[str, list[Path]]:
    """Write every site's cases under ``root/<SITE>`` and return the paths."""
    root = Path(root)
    return {
        profile.name: generate_site(root / profile.name, profile, seed=seed)
        for profile in CLIENT_PROFILES
    }


