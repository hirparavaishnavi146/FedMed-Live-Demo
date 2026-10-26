"""The hospital client abstraction.

A `HospitalClient` holds one site's data and trains a model on it locally. When
asked for an update it returns **parameters and aggregate metrics only** — never
examples, labels, gradients per-sample, or anything from which a training example
could be reconstructed.

The isolation guarantee this module provides is structural, not a promise:

    * `LocalDataset` is created inside the client and never returned. There is no
      accessor that exposes it, so no caller of this class can read it by
      accident.
    * `ClientUpdate` is the only outbound type, and it is defined with exactly
      the fields the server is allowed to see. A test asserts its field set, so
      adding a raw-data field to it would fail the suite rather than quietly ship.
    * `fit()` trains in-process. Nothing here opens a socket, so "the data stayed
      local" is true by construction in the simulation.

On a real cross-silo deployment the same boundary has to be enforced by network
policy and by the client running inside the hospital's own infrastructure — the
in-process isolation here demonstrates the interface, not the deployment
guarantee. See docs/FEDERATED.md.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch import nn

from .config import RoundConfig
from .data import LocalDataset, split_train_validation
from .errors import ClientTrainingError
from .model import build_model, numpy_to_state, parameter_count, state_to_numpy


@dataclass(frozen=True)
class ClientUpdate:
    """Everything a client sends to the aggregation server.

    This is the complete and exhaustive list of what leaves a client. It holds
    model parameters and summary numbers. It holds no training examples, no
    labels and no per-sample information.

    `num_examples` is the number of local training examples used, and it is the
    FedAvg weight. Reporting the count is necessary for correct averaging and is
    not the same as sending the data.
    """

    client: str
    parameters: "dict[str, np.ndarray]"
    num_examples: int
    train_loss: float
    validation_loss: float | None
    validation_dice: float | None
    local_epochs: int
    duration_seconds: float
    seed: int

    # Field names that must never appear in an outbound update. Asserted by the
    # test suite so this contract cannot quietly rot.
    FORBIDDEN_FIELDS = ("inputs", "targets", "dataset", "samples", "images", "labels")

    def summary(self) -> dict[str, Any]:
        """JSON-safe description, for logs, history and the API."""
        return {
            "client": self.client,
            "numExamples": self.num_examples,
            "trainLoss": round(self.train_loss, 6),
            "validationLoss": None if self.validation_loss is None else round(self.validation_loss, 6),
            "validationDice": None if self.validation_dice is None else round(self.validation_dice, 6),
            "localEpochs": self.local_epochs,
            "durationSeconds": round(self.duration_seconds, 4),
            "seed": self.seed,
            "parameterNames": sorted(self.parameters.keys()),
        }


class HospitalClient:
    """One simulated hospital: owns its data, trains locally, sends parameters."""

    def __init__(
        self,
        name: str,
        dataset: LocalDataset,
        config: RoundConfig,
    ) -> None:
        self.name = name
        self._config = config
        # Private. No accessor exposes this, by design.
        self._train_data, self._validation_data = split_train_validation(
            dataset, config.validation_split, seed=config.seed
        )
        self._client_seed = abs(hash((config.seed, name))) % (2**31)

    # -- information the server is allowed to have -------------------------
    @property
    def training_examples(self) -> int:
        """Local training-set size. This is the FedAvg weight."""
        return self._train_data.size

    @property
    def validation_examples(self) -> int:
        return 0 if self._validation_data is None else self._validation_data.size

    @property
    def profile_description(self) -> str:
        return self._train_data.profile.description

    # -- local training ----------------------------------------------------
    def _build_local_model(self, global_parameters: dict[str, np.ndarray] | None) -> nn.Module:
        """A model initialised from the global parameters, or from the run seed."""
        if global_parameters is None:
            return build_model(seed=self._client_seed)
        model = build_model()
        try:
            numpy_to_state(model, global_parameters)
        except ValueError as exc:
            raise ClientTrainingError(
                f"Client {self.name} received parameters it cannot load: {exc}",
                code="client_incompatible_parameters",
                detail={"client": self.name, "reason": str(exc)},
            ) from exc
        return model

    def fit(self, global_parameters: dict[str, np.ndarray] | None) -> ClientUpdate:
        """Train locally and return the update that leaves this client.

        Local training runs entirely inside this process and uses only this
        client's own data. A failure raises ClientTrainingError: the caller
        excludes the client rather than inventing an update for it.
        """
        started = time.perf_counter()
        try:
            model = self._build_local_model(global_parameters)
            optimiser = torch.optim.SGD(model.parameters(), lr=self._config.learning_rate)
            loss_function = nn.BCEWithLogitsLoss()

            generator = torch.Generator().manual_seed(self._client_seed)
            running_loss = 0.0
            batches_seen = 0

            model.train()
            for _ in range(self._config.local_epochs):
                for inputs, targets in self._train_data.batches(
                    self._config.batch_size, generator=generator
                ):
                    optimiser.zero_grad(set_to_none=True)
                    outputs = model(inputs)
                    loss = loss_function(outputs, targets)
                    loss.backward()
                    optimiser.step()
                    running_loss += float(loss.item())
                    batches_seen += 1

            if batches_seen == 0:
                raise ClientTrainingError(
                    f"Client {self.name} produced no training batches.",
                    detail={"client": self.name},
                )

            train_loss = running_loss / batches_seen
            validation_loss, validation_dice = self._evaluate_model(model)

            return ClientUpdate(
                client=self.name,
                parameters=state_to_numpy(model),
                num_examples=self._train_data.size,
                train_loss=train_loss,
                validation_loss=validation_loss,
                validation_dice=validation_dice,
                local_epochs=self._config.local_epochs,
                duration_seconds=time.perf_counter() - started,
                seed=self._client_seed,
            )
        except ClientTrainingError:
            raise
        except Exception as exc:  # noqa: BLE001 - any failure means "client failed"
            raise ClientTrainingError(
                f"Client {self.name} failed during local training: {exc}",
                detail={"client": self.name, "reason": str(exc), "error": type(exc).__name__},
            ) from exc

    def _evaluate_model(self, model: nn.Module) -> tuple[float | None, float | None]:
        """Loss and Dice on this client's own validation split.

        Returns (None, None) when the split is empty, so a missing measurement is
        reported as missing rather than as a zero.
        """
        if self._validation_data is None or self._validation_data.size == 0:
            return None, None

        loss_function = nn.BCEWithLogitsLoss()
        model.eval()
        with torch.no_grad():
            outputs = model(self._validation_data.inputs)
            loss = float(loss_function(outputs, self._validation_data.targets).item())
            predicted = (torch.sigmoid(outputs) >= 0.5).float()
            dice = _dice_score(predicted, self._validation_data.targets)
        return loss, dice

    def evaluate(self, parameters: dict[str, np.ndarray]) -> dict[str, float] | None:
        """Score supplied parameters on this client's validation split.

        Used by the server to check the aggregated model on data it cannot see.
        Returns None when the client has no validation split.
        """
        model = self._build_local_model(parameters)
        loss, dice = self._evaluate_model(model)
        if loss is None:
            return None
        return {"validation_loss": loss, "validation_dice": dice or 0.0}


def _dice_score(predicted: torch.Tensor, target: torch.Tensor) -> float:
    """Mean Dice over the batch.

    Dice is the metric this project reports for segmentation, so it is measured
    here rather than estimated. A batch with no positive voxels in either mask is
    skipped instead of being scored 1.0, which would flatter the number.
    """
    predicted = predicted.reshape(predicted.shape[0], -1)
    target = target.reshape(target.shape[0], -1)

    scores: list[float] = []
    for index in range(predicted.shape[0]):
        prediction_row = predicted[index]
        target_row = target[index]
        denominator = float(prediction_row.sum() + target_row.sum())
        if denominator == 0.0:
            continue
        intersection = float((prediction_row * target_row).sum())
        scores.append(2.0 * intersection / denominator)
    return float(np.mean(scores)) if scores else 0.0


@dataclass
class ClientRegistry:
    """Builds and holds the clients for one run."""

    clients: "dict[str, HospitalClient]" = field(default_factory=dict)

    @classmethod
    def build(cls, config: RoundConfig, datasets: "dict[str, LocalDataset]") -> "ClientRegistry":
        registry = cls()
        for label in config.active_labels:
            dataset = datasets.get(label)
            if dataset is None:
                continue
            registry.clients[label] = HospitalClient(label, dataset, config)
        return registry

    def get(self, name: str) -> HospitalClient | None:
        return self.clients.get(name)

    def names(self) -> list[str]:
        return sorted(self.clients.keys())

    def total_examples(self) -> int:
        return sum(client.training_examples for client in self.clients.values())

    def describe(self) -> list[dict[str, Any]]:
        """Per-client facts that are safe to publish: sizes, not data."""
        return [
            {
                "name": client.name,
                "trainingExamples": client.training_examples,
                "validationExamples": client.validation_examples,
                "description": client.profile_description,
            }
            for client in self.clients.values()
        ]


def model_size(model: nn.Module | None = None) -> int:
    """Parameter count of the client model, for the API description."""
    return parameter_count(model if model is not None else build_model())
