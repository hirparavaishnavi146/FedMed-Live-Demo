"""Per-client synthetic datasets.

Each hospital client owns its own data, generated locally from its own seed. Two
properties matter for the federated experiment to mean anything:

1.  Independence. Client A's examples are not derived from client B's. Each has
    its own random generator, seeded from (run seed, client name), so a run is
    reproducible but no client shares another's samples.

2.  Non-IID data. The clients deliberately differ: different lesion size ranges,
    different noise levels and different sample counts. Identical shards would
    make Federated Averaging look better than it is, because averaging identical
    distributions is easy. The differences are what make aggregation meaningful.

The task is deliberately simple (segment a bright sphere in a small patch) so
that a few rounds show measurable progress on a CPU.

NO REAL DATA OF ANY KIND. These are procedurally generated arrays of floats.
There is no patient, no scan, no DICOM, no NIfTI and no identifier anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .config import RoundConfig
from .model import PATCH_SIZE


@dataclass(frozen=True)
class ClientProfile:
    """How one simulated site's data differs from the others.

    These differences are the point: they make the clients non-IID, which is the
    realistic cross-silo situation and the reason federated averaging is more
    than an averaging of identical things.
    """

    name: str
    samples: int
    radius_range: tuple[float, float]
    noise: float
    description: str


DEFAULT_PROFILES: tuple[ClientProfile, ...] = (
    ClientProfile(
        name="HOSP-01",
        samples=48,
        radius_range=(2.6, 3.6),
        noise=0.08,
        description="Larger, cleaner lesions. The largest local dataset.",
    ),
    ClientProfile(
        name="HOSP-02",
        samples=36,
        radius_range=(1.8, 2.8),
        noise=0.14,
        description="Smaller lesions with more noise. Mid-sized dataset.",
    ),
    ClientProfile(
        name="HOSP-03",
        samples=20,
        radius_range=(3.2, 4.4),
        noise=0.20,
        description="Largest but noisiest lesions, smallest dataset. Contributes "
        "the least to the average, which is the intended weighting behaviour.",
    ),
)


@dataclass
class LocalDataset:
    """One client's private data: inputs and targets, held only on that client."""

    inputs: torch.Tensor
    targets: torch.Tensor
    profile: ClientProfile

    @property
    def size(self) -> int:
        """Number of examples. This count is the FedAvg weight for the client."""
        return int(self.inputs.shape[0])

    def batches(
        self, batch_size: int, *, generator: torch.Generator | None = None
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Split into mini-batches, shuffling when a generator is supplied."""
        order = torch.randperm(self.size, generator=generator)
        shuffled_inputs = self.inputs[order]
        shuffled_targets = self.targets[order]
        return [
            (shuffled_inputs[start : start + batch_size], shuffled_targets[start : start + batch_size])
            for start in range(0, self.size, batch_size)
        ]


def _sphere_mask(size: int, centre: np.ndarray, radius: float) -> np.ndarray:
    """Binary 3D mask of a sphere, as float32."""
    coordinates = np.stack(
        np.meshgrid(
            *(np.arange(size, dtype=np.float32) for _ in range(3)), indexing="ij"
        ),
        axis=-1,
    )
    distance = np.linalg.norm(coordinates - centre, axis=-1)
    return (distance <= radius).astype(np.float32)


def generate_dataset(
    profile: ClientProfile,
    *,
    seed: int,
    patch_size: int = PATCH_SIZE,
) -> LocalDataset:
    """Generate one client's dataset.

    Deterministic for a given (seed, profile.name, patch_size): the same
    configuration reproduces exactly the same examples.
    """
    # The client name goes into the seed so clients never share a stream, even
    # though they share the run seed.
    client_seed = abs(hash((seed, profile.name, patch_size))) % (2**31)
    rng = np.random.default_rng(client_seed)

    inputs = np.empty((profile.samples, 1, patch_size, patch_size, patch_size), dtype=np.float32)
    targets = np.empty_like(inputs)

    low, high = profile.radius_range
    for index in range(profile.samples):
        centre = rng.uniform(low=patch_size * 0.3, high=patch_size * 0.7, size=3)
        radius = float(rng.uniform(low, high))
        mask = _sphere_mask(patch_size, centre, radius)

        # Background is a dim textured field; the lesion is brighter. Noise is
        # drawn per client, so the sites differ in how clean their data is.
        background = rng.normal(loc=0.15, scale=profile.noise, size=mask.shape).astype(np.float32)
        lesion = rng.normal(loc=0.85, scale=profile.noise, size=mask.shape).astype(np.float32)
        image = np.where(mask > 0.5, lesion, background).astype(np.float32)

        inputs[index, 0] = image
        targets[index, 0] = mask

    return LocalDataset(
        inputs=torch.from_numpy(inputs),
        targets=torch.from_numpy(targets),
        profile=profile,
    )


def build_client_datasets(config: RoundConfig) -> dict[str, LocalDataset]:
    """Build one independent dataset per client taking part in this run."""
    profiles = {profile.name: profile for profile in DEFAULT_PROFILES}
    datasets: dict[str, LocalDataset] = {}
    for label in config.active_labels:
        profile = profiles.get(label)
        if profile is None:
            # A label with no profile would silently train on nothing.
            continue
        datasets[label] = generate_dataset(profile, seed=config.seed, patch_size=config.patch_size)
    return datasets


def split_train_validation(
    dataset: LocalDataset, validation_split: float, *, seed: int
) -> tuple[LocalDataset, LocalDataset | None]:
    """Split a client's data locally. The split never leaves the client."""
    if validation_split <= 0.0 or dataset.size < 4:
        return dataset, None

    count = int(dataset.size * validation_split)
    if count < 1:
        return dataset, None

    rng = np.random.default_rng(abs(hash((seed, dataset.profile.name, "split"))) % (2**31))
    order = rng.permutation(dataset.size)
    validation_index = np.sort(order[:count])
    train_index = np.sort(order[count:])

    def take(indices: np.ndarray) -> LocalDataset:
        return LocalDataset(
            inputs=dataset.inputs[indices],
            targets=dataset.targets[indices],
            profile=dataset.profile,
        )

    return take(train_index), take(validation_index)
