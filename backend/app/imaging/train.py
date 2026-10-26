"""Reproducible training loop, validation, metrics and checkpointing.

Design points that matter for this project:

    * **Metrics come from predictions.** Every number in a `TrainingResult` is
      computed by `imaging.metrics` from the model's actual output on real
      validation volumes. Nothing is estimated or carried over from a previous
      step. A validation split that is empty yields ``None``, not zero, so a
      missing measurement is visibly missing.
    * **Reproducible.** One `seed` drives weight initialisation, shuffling and
      dropout. `TrainingResult.config` records everything that affected the run,
      so a result can be repeated rather than merely believed.
    * **Checkpoints record enough to resume.** Weights, optimizer state, epoch,
      the config and the metrics of the saved epoch. `load_checkpoint` refuses a
      checkpoint whose parameter names or shapes do not match the model it is
      being loaded into, because silently loading a mismatched state dict is how
      a "resumed" run quietly becomes a different model.
    * **CPU-first.** No pinning, no AMP, no DataLoader worker processes by
      default: the reference machine has 7.9 GB of RAM and no GPU.
"""

from __future__ import annotations

import dataclasses
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

import torch
from torch import nn

from .metrics import DEFAULT_THRESHOLD, segmentation_report

CHECKPOINT_FORMAT = "fedmed-imaging-checkpoint"
CHECKPOINT_VERSION = 1


@dataclass(frozen=True)
class TrainConfig:
    """Everything that affects the outcome of a training run."""

    epochs: int = 6
    batch_size: int = 1
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    seed: int = 42
    threshold: float = DEFAULT_THRESHOLD
    device: str = "cpu"
    max_batches_per_epoch: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def describe(self) -> str:
        return (
            f"{self.epochs} epochs, batch={self.batch_size}, lr={self.learning_rate}, "
            f"wd={self.weight_decay}, threshold={self.threshold}, seed={self.seed}, "
            f"device={self.device}"
        )


@dataclass
class EpochRecord:
    """One epoch's measured values."""

    epoch: int
    train_loss: float
    validation_loss: float | None
    validation_dice: float | None
    validation_iou: float | None
    validation_sensitivity: float | None
    validation_specificity: float | None
    validation_samples: int
    duration_seconds: float

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class TrainingResult:
    """The outcome of a run: measured history plus where the best weights are."""

    config: TrainConfig
    parameter_count: int
    epochs: list[EpochRecord] = field(default_factory=list)
    best_dice: float | None = None
    best_epoch: int | None = None
    best_checkpoint: str | None = None
    history_checkpoints: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    dataset: dict[str, Any] = field(default_factory=dict)

    @property
    def first_train_loss(self) -> float | None:
        return self.epochs[0].train_loss if self.epochs else None

    @property
    def last_train_loss(self) -> float | None:
        return self.epochs[-1].train_loss if self.epochs else None

    @property
    def loss_improved(self) -> bool:
        """True only when the loss actually fell over the run."""
        if len(self.epochs) < 2:
            return False
        return self.epochs[-1].train_loss < self.epochs[0].train_loss

    def as_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.as_dict(),
            "configDescription": self.config.describe(),
            "parameterCount": self.parameter_count,
            "epochs": [record.as_dict() for record in self.epochs],
            "epochCount": len(self.epochs),
            "firstTrainLoss": self.first_train_loss,
            "lastTrainLoss": self.last_train_loss,
            "lossImproved": self.loss_improved,
            "bestDice": self.best_dice,
            "bestEpoch": self.best_epoch,
            "bestCheckpoint": self.best_checkpoint,
            "historyCheckpoints": self.history_checkpoints,
            "durationSeconds": round(self.duration_seconds, 4),
            "dataset": self.dataset,
        }


def seed_everything(seed: int) -> None:
    """Seed torch and NumPy so a run is repeatable on this machine."""
    import numpy as np

    torch.manual_seed(seed)
    np.random.seed(seed % (2**32))
    # Deterministic algorithms where torch can guarantee them. Requested rather
    # than required: on CPU these kernels are deterministic already, and
    # warn_only avoids failing a run over a library choice.
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except (RuntimeError, AttributeError):
        pass


def build_optimizer(model: nn.Module, config: TrainConfig) -> torch.optim.Optimizer:
    """Adam, matching the reference configuration used in M5 experiments."""
    return torch.optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )


def count_batches(loader: Iterable) -> int:
    """Number of batches in a loader, without materialising it."""
    try:
        return len(loader)
    except TypeError:
        return 0


def train_epoch(
    model: nn.Module,
    loader: Iterable,
    optimizer: torch.optim.Optimizer,
    loss_function: nn.Module,
    config: TrainConfig,
) -> tuple[float, int]:
    """One pass over the training data. Returns (mean loss, batches seen).

    Raises RuntimeError when the loader yields nothing: a silent zero-batch epoch
    would report a loss of 0.0 and look like progress.
    """
    model.train()
    total_loss = 0.0
    batches = 0
    for inputs, targets in loader:
        if config.max_batches_per_epoch is not None and batches >= config.max_batches_per_epoch:
            break
        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = loss_function(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
        batches += 1

    if batches == 0:
        raise RuntimeError(
            "No training batches were produced. The dataset is empty or the loader "
            "is misconfigured."
        )
    return total_loss / batches, batches


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: Iterable,
    loss_function: nn.Module,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict[str, Any]:
    """Validation metrics, accumulated over the whole split.

    Loss is example-weighted so it is comparable between splits of different
    sizes. Dice and the other metrics are pooled voxel counts over the split
    (see `metrics.segmentation_report`), because averaging per-batch Dice would
    weight a nearly-empty patch the same as a dense one.
    """
    model.eval()
    total_loss = 0.0
    batches = 0
    pooled_predictions: list[torch.Tensor] = []
    pooled_targets: list[torch.Tensor] = []

    for inputs, targets in loader:
        outputs = model(inputs)
        total_loss += float(loss_function(outputs, targets).item())
        batches += 1
        pooled_predictions.append(outputs.detach())
        pooled_targets.append(targets.detach())

    if batches == 0:
        return {
            "loss": None,
            "dice": None,
            "iou": None,
            "sensitivity": None,
            "specificity": None,
            "samplesScored": 0,
            "samplesSkipped": 0,
            "batches": 0,
        }

    report = segmentation_report(
        torch.cat(pooled_predictions, dim=0),
        torch.cat(pooled_targets, dim=0),
        loss=total_loss / batches,
        threshold=threshold,
    )
    report["batches"] = batches
    return report


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    config: TrainConfig,
    metrics: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write a checkpoint that records how it was produced."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": CHECKPOINT_FORMAT,
        "version": CHECKPOINT_VERSION,
        "epoch": int(epoch),
        "modelState": {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()},
        "optimizerState": None if optimizer is None else optimizer.state_dict(),
        "config": config.as_dict(),
        "metrics": metrics or {},
        "extra": extra or {},
        "parameterCount": int(
            sum(p.numel() for p in model.parameters() if p.requires_grad)
        ),
    }
    # weights_only=False is required because the payload holds non-tensor objects
    # (the config dict and metric values). This is acceptable only because the file
    # is written by this project; do not load a checkpoint from an untrusted source.
    torch.save(payload, str(path))
    return path


def read_checkpoint(path: Path) -> dict[str, Any]:
    """Load a checkpoint payload, validating that it is one of ours."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No checkpoint at {path}.")
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(
            f"{path} is not a FedMed imaging checkpoint "
            f"(expected format {CHECKPOINT_FORMAT!r})."
        )
    if payload.get("version") != CHECKPOINT_VERSION:
        raise ValueError(
            f"Checkpoint version {payload.get('version')} is not supported "
            f"(this build reads version {CHECKPOINT_VERSION})."
        )
    return payload


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    strict: bool = True,
) -> dict[str, Any]:
    """Restore weights (and optionally optimizer state) into a model.

    Refuses a checkpoint whose parameter names or shapes differ from the model,
    rather than loading whatever happens to match. A partially-loaded state dict
    would produce a model that is neither the checkpoint's nor the model's.
    """
    payload = read_checkpoint(path)

    incoming = payload["modelState"]
    current = model.state_dict()

    missing = sorted(set(current.keys()) - set(incoming.keys()))
    unexpected = sorted(set(incoming.keys()) - set(current.keys()))
    if strict and (missing or unexpected):
        raise ValueError(
            "Checkpoint does not match this model: "
            f"missing {missing}, unexpected {unexpected}. Both were built from the "
            "same architecture with the same feature widths and channel counts."
        )

    shape_mismatches = [
        (name, tuple(tensor.shape), tuple(current[name].shape))
        for name, tensor in incoming.items()
        if name in current and tuple(tensor.shape) != tuple(current[name].shape)
    ]
    if shape_mismatches:
        detail = "; ".join(
            f"{name}: checkpoint {shape} vs model {expected}"
            for name, shape, expected in shape_mismatches
        )
        raise ValueError(f"Checkpoint parameter shapes do not match the model: {detail}.")

    model.load_state_dict(incoming, strict=strict)

    if optimizer is not None and payload.get("optimizerState") is not None:
        optimizer.load_state_dict(payload["optimizerState"])

    return payload


def train_model(
    model: nn.Module,
    train_loader: Iterable,
    validation_loader: Iterable | None,
    config: TrainConfig,
    checkpoint_dir: Path | None = None,
    dataset_description: dict[str, Any] | None = None,
    loss_function: nn.Module | None = None,
    on_epoch: Callable[[EpochRecord], None] | None = None,
) -> TrainingResult:
    """Run the full loop, checkpointing the best-Dice epoch and the last epoch.

    `validation_loader=None` is allowed: the run then reports training loss only
    and checkpoints the last epoch, rather than inventing validation numbers.

    `on_epoch` is called after each epoch with that epoch's record. The API's
    training job uses it to report live progress. It is optional so the CLI
    behaves exactly as before, and the loop stays the single implementation
    rather than being copied into a second, reporting variant.
    """
    seed_everything(config.seed)
    torch.set_num_threads(max(1, torch.get_num_threads()))

    loss_function = loss_function or nn.BCEWithLogitsLoss()
    optimizer = build_optimizer(model, config)

    result = TrainingResult(
        config=config,
        parameter_count=int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        dataset=dataset_description or {},
    )
    started = time.perf_counter()
    best_dice: float | None = None

    for epoch in range(1, config.epochs + 1):
        epoch_started = time.perf_counter()
        train_loss, _batches = train_epoch(model, train_loader, optimizer, loss_function, config)

        validation: dict[str, Any] = {
            "loss": None,
            "dice": None,
            "iou": None,
            "sensitivity": None,
            "specificity": None,
            "samplesScored": 0,
        }
        if validation_loader is not None:
            validation = evaluate(model, validation_loader, loss_function, config.threshold)

        record = EpochRecord(
            epoch=epoch,
            train_loss=train_loss,
            validation_loss=validation["loss"],
            validation_dice=validation["dice"],
            validation_iou=validation["iou"],
            validation_sensitivity=validation["sensitivity"],
            validation_specificity=validation["specificity"],
            validation_samples=int(validation.get("samplesScored") or 0),
            duration_seconds=time.perf_counter() - epoch_started,
        )
        result.epochs.append(record)

        # Reported here rather than after the loop, so a caller can show progress
        # and surface a mid-run failure at the epoch it happened.
        if on_epoch is not None:
            on_epoch(record)

        # Best-epoch tracking is a *measurement* and must not depend on whether a
        # checkpoint was requested. An earlier version set best_dice/best_epoch
        # only inside the checkpoint branch, so a run with no checkpoint directory
        # reported no best Dice at all despite having measured one.
        dice = record.validation_dice
        improved = dice is not None and (best_dice is None or dice > best_dice)
        if improved:
            best_dice = dice
            result.best_dice = dice
            result.best_epoch = epoch

        if checkpoint_dir is not None:
            metrics = record.as_dict()
            # Always keep the epoch just finished, so an interrupted run has a
            # usable artifact regardless of how the metrics moved.
            result.history_checkpoints.append(
                str(
                    save_checkpoint(
                        Path(checkpoint_dir) / "last.pt", model, optimizer, epoch, config, metrics
                    )
                )
            )
            if improved:
                result.best_checkpoint = str(
                    save_checkpoint(
                        Path(checkpoint_dir) / "best.pt", model, optimizer, epoch, config, metrics
                    )
                )

    if checkpoint_dir is not None and result.best_checkpoint is None and result.epochs:
        # No validation split (or no epoch improved), so the artifact a caller
        # should load is the final state. Named explicitly so nobody has to guess.
        result.best_checkpoint = result.history_checkpoints[-1]
        if result.best_epoch is None:
            result.best_epoch = result.epochs[-1].epoch

    result.duration_seconds = time.perf_counter() - started
    return result


def write_result(path: Path, result: TrainingResult) -> Path:
    """Persist a result as JSON, for the API and the frontend to read."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.as_dict(), indent=2), encoding="utf-8")
    return path
