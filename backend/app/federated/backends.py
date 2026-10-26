"""Which model the federated clients train, selected explicitly.

Two backends exist, and the choice is recorded in every run's configuration so a
result can never be attributed to the wrong architecture:

    ``tiny``   The small stand-in from `federated.model` (~550 parameters). The
               default, because a full federated round completes in under a second
               on the CPU reference machine, which is what the aggregation code
               needs in order to be exercised quickly and repeatedly.

    ``unet3d`` The real 3D U-Net from `imaging.unet` (~350k parameters) trained on
               preprocessed synthetic NIfTI volumes. Genuine volumetric
               segmentation, and the model this project is ultimately about.

Both satisfy the same contract, which is all the federated layer requires:

    * parameters are a name -> NumPy array mapping via `state_to_numpy`
    * the names and shapes are identical on every client
    * `numpy_to_state` loads an aggregated update back in

Because that contract is model-agnostic, the existing cross-client validation in
`federated.fedavg.validate_updates` protects both. `tests/test_imaging_federated.py`
asserts it for the U-Net specifically, so swapping the backend cannot quietly
bypass the compatibility check.

HONESTY NOTE
    Whichever backend is selected, the clients are simulated on one machine, the
    data is synthetic, and `execution` remains `local-simulation`. Using the real
    U-Net here does not make the run a cross-silo deployment and does not make its
    metrics clinical results.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from torch import nn

#: The backend used when none is named. Changing this default would change what
#: every existing experiment measures, so it is deliberately the cheap one.
DEFAULT_BACKEND = "tiny"


@dataclass(frozen=True)
class ModelBackend:
    """A description of a trainable client model."""

    name: str
    build: Callable[..., nn.Module]
    description: str
    parameter_hint: str
    input_shape: str
    trained_on: str

    def describe(self, model: nn.Module | None = None) -> dict[str, Any]:
        from .model import parameter_count as tiny_parameter_count

        count = tiny_parameter_count(model if model is not None else self.build())
        return {
            "name": self.name,
            "description": self.description,
            "parameterHint": self.parameter_hint,
            "parameters": count,
            "inputShape": self.input_shape,
            "trainedOn": self.trained_on,
        }


def _build_tiny(seed: int | None = None) -> nn.Module:
    from .model import build_model

    return build_model(seed=seed)


def _build_unet(seed: int | None = None) -> nn.Module:
    # Imported lazily so that the federated layer does not depend on MONAI unless
    # the U-Net backend is actually selected.
    from ..imaging.unet import build_unet

    return build_unet(seed=seed)


BACKENDS: dict[str, ModelBackend] = {
    "tiny": ModelBackend(
        name="tiny",
        build=_build_tiny,
        description=(
            "A small 3D convolutional network used to exercise federated aggregation "
            "quickly. It is a real trainable model, not the MRI segmentation model."
        ),
        parameter_hint="~550",
        input_shape="(N, 1, P, P, P) with P the configured patch size",
        trained_on="synthetic sphere patches generated in memory",
    ),
    "unet3d": ModelBackend(
        name="unet3d",
        build=_build_unet,
        description=(
            "The 3D U-Net from app/imaging: a real volumetric segmentation model "
            "trained on preprocessed synthetic NIfTI phantoms."
        ),
        parameter_hint="~350k",
        input_shape="(N, 1, D, H, W), each spatial size divisible by 2 ** stages",
        trained_on="MONAI-preprocessed synthetic phantom volumes",
    ),
}


def resolve_backend(name: str | None) -> ModelBackend:
    """Look up a backend, with a message that names the valid options."""
    key = (name or DEFAULT_BACKEND).strip().lower()
    if key not in BACKENDS:
        raise ValueError(
            f"Unknown model backend {name!r}. Available: {sorted(BACKENDS)}."
        )
    return BACKENDS[key]


def backend_names() -> list[str]:
    return sorted(BACKENDS)


def describe_backends() -> list[dict[str, Any]]:
    """All backends, for the API's training-config response."""
    return [
        {
            "name": backend.name,
            "description": backend.description,
            "parameterHint": backend.parameter_hint,
            "inputShape": backend.input_shape,
            "trainedOn": backend.trained_on,
            "isDefault": backend.name == DEFAULT_BACKEND,
        }
        for backend in (BACKENDS[key] for key in backend_names())
    ]


def parameter_signature(model: nn.Module) -> list[tuple[str, tuple[int, ...]]]:
    """The (name, shape) list that must match across clients before aggregation.

    This is the concrete form of "compatible model structures": two clients can
    only average their updates if these lists are equal.
    """
    return [(name, tuple(tensor.shape)) for name, tensor in model.state_dict().items()]
