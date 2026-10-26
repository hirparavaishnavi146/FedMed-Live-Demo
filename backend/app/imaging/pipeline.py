"""Standalone 3D U-Net experiment: preprocess, train, validate, report.

This is the entry point for requirement 10 — the model is trained and validated
here, on its own, before anything connects it to federated aggregation.

    python -m app.imaging.pipeline --site HOSP-01 --epochs 4

It writes `experiment.json` (measured values) and checkpoints under ``--out``.
The JSON is what the API and the frontend read, so a published number is always
one this script actually measured.

DATASET
    Runs on the synthetic phantoms from `imaging.synthetic` by default: nothing is
    downloaded, no patient data is involved, and the command works offline. To use
    real research data, pass ``--data-dir`` pointing at a directory of
    ``<case>.nii.gz`` files each paired with ``<case>_seg.nii.gz`` — see
    docs/IMAGING.md for the licence, download and storage notes for the candidate
    public datasets. This script never downloads anything itself.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .dataset import CaseDataset, build_site_datasets, dataset_summary
from .preprocess import PreprocessConfig
from .synthetic import (
    CLIENT_PROFILES,
    find_case_images,
    generate_site,
    label_path_for,
)
from .train import (
    TrainConfig,
    TrainingResult,
    build_optimizer,
    evaluate,
    load_checkpoint,
    train_model,
    write_result,
)
from .unet import build_unet, parameter_count

DEFAULT_OUTPUT_DIR = Path("experiments/imaging")


@dataclass(frozen=True)
class ExperimentConfig:
    """One standalone experiment, fully described."""

    site: str = "HOSP-01"
    epochs: int = 4
    batch_size: int = 1
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    seed: int = 42
    validation_fraction: float = 0.34
    features: tuple[int, ...] = (8, 16, 32)
    target_shape: tuple[int, int, int] = (32, 32, 32)
    target_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)
    dropout: float = 0.0
    data_dir: Path | None = None
    output_dir: Path = DEFAULT_OUTPUT_DIR

    def train_config(self) -> TrainConfig:
        return TrainConfig(
            epochs=self.epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            seed=self.seed,
        )

    def preprocess_config(self) -> PreprocessConfig:
        return PreprocessConfig(
            target_spacing=self.target_spacing, target_shape=self.target_shape
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "site": self.site,
            "epochs": self.epochs,
            "batchSize": self.batch_size,
            "learningRate": self.learning_rate,
            "weightDecay": self.weight_decay,
            "seed": self.seed,
            "validationFraction": self.validation_fraction,
            "features": list(self.features),
            "targetShape": list(self.target_shape),
            "targetSpacing": list(self.target_spacing),
            "dropout": self.dropout,
            "dataDir": None if self.data_dir is None else str(self.data_dir),
            "outputDir": str(self.output_dir),
        }


def _collect_cases(config: ExperimentConfig, data_root: Path) -> list[Path]:
    """Find the image files for a run, or generate them when none were supplied."""
    if config.data_dir is not None:
        data_dir = Path(config.data_dir)
        if not data_dir.exists():
            raise FileNotFoundError(
                f"--data-dir {data_dir} does not exist. This script never downloads a "
                "dataset; see docs/IMAGING.md for how to obtain one."
            )
        # Shared helper: excludes *_seg.nii.gz, which are labels, not cases.
        cases = find_case_images(data_dir)
        if not cases:
            raise FileNotFoundError(
                f"No .nii.gz image files in {data_dir}. Each case needs "
                "<name>.nii.gz plus <name>_seg.nii.gz."
            )
        return cases

    profile = next(
        (item for item in CLIENT_PROFILES if item.name == config.site), None
    )
    if profile is None:
        known = ", ".join(item.name for item in CLIENT_PROFILES)
        raise ValueError(f"Unknown site {config.site!r}; choose one of {known}.")
    return generate_site(data_root / profile.name, profile, seed=config.seed)


def run_experiment(
    config: ExperimentConfig,
    data_root: Path | None = None,
    on_epoch: Any | None = None,
) -> dict[str, Any]:
    """Run one standalone experiment and return its measured result.

    Order matters: build the model with the same seed the loop uses, so the whole
    run is reproducible from `config.seed` alone.

    `on_epoch` is forwarded to the training loop so a caller can report live
    progress. The CLI passes nothing and behaves exactly as before.
    """
    data_root = Path(data_root) if data_root is not None else config.output_dir / "data"
    config.output_dir.mkdir(parents=True, exist_ok=True)

    cases = _collect_cases(config, data_root)
    preprocess_config = config.preprocess_config()
    splits = build_site_datasets(
        cases,
        preprocess_config,
        validation_fraction=config.validation_fraction,
        seed=config.seed,
        name=config.site,
    )
    train_set: CaseDataset = splits["train"]
    validation_set: CaseDataset = splits["validation"]

    model = build_unet(
        seed=config.seed, features=config.features, dropout=config.dropout
    )
    train_loader = list(train_set.batches(config.batch_size))
    validation_loader = list(validation_set.batches(config.batch_size)) or None

    result: TrainingResult = train_model(
        model,
        train_loader,
        validation_loader,
        config.train_config(),
        checkpoint_dir=config.output_dir / "checkpoints",
        dataset_description=dataset_summary(train_set, validation_set, preprocess_config),
        on_epoch=on_epoch,
    )

    # Re-evaluate the saved best checkpoint from disk, so the reported Dice comes
    # from the artifact a consumer would load rather than from whatever happened
    # to be in memory at the end of the loop.
    holdout: dict[str, Any] = {"dice": None, "loss": None, "cases": 0}
    if result.best_checkpoint and validation_loader is not None:
        restored = build_unet(seed=0, features=config.features)
        load_checkpoint(result.best_checkpoint, restored)
        holdout = evaluate(restored, validation_loader, nn.BCEWithLogitsLoss())
        holdout["cases"] = validation_set.size
        holdout["source"] = "best checkpoint re-evaluated from disk"

    report = {
        "experiment": config.as_dict(),
        "result": result.as_dict(),
        "bestCheckpointEvaluation": holdout,
        "model": {
            "architecture": "UNet3D",
            "features": list(config.features),
            "parameterCount": parameter_count(model),
            "outputIsLogits": True,
        },
        "honesty": {
            "dataSource": "synthetic phantoms, procedurally generated",
            "containsPatientData": False,
            "clinicallyValidated": False,
            "isDemoValue": False,
            "note": (
                "These are measurements from real local training on synthetic data. "
                "They are not demo placeholders and they are not clinical results."
            ),
        },
    }

    # The full report is the primary artifact; the bare training result is also
    # written separately so a consumer can read either without unwrapping.
    write_result(config.output_dir / "training_result.json", result)
    (config.output_dir / "experiment.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app.imaging.pipeline",
        description=(
            "Train and validate the 3D U-Net standalone on synthetic phantoms. "
            "Nothing is downloaded and no patient data is used."
        ),
    )
    parser.add_argument(
        "--site",
        default="HOSP-01",
        choices=[profile.name for profile in CLIENT_PROFILES],
        help="Which simulated site's phantom characteristics to use.",
    )
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.34,
        help="Fraction of CASES held back for validation (split is by case).",
    )
    parser.add_argument(
        "--target-shape",
        type=int,
        nargs=3,
        default=list(ExperimentConfig().target_shape),
        metavar=("D", "H", "W"),
        help="Network input shape. Each value must be divisible by 2 ** stages (8).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help=(
            "Directory of <name>.nii.gz + <name>_seg.nii.gz pairs. Omit to use "
            "generated synthetic phantoms."
        ),
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = ExperimentConfig(
        site=args.site,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        validation_fraction=args.validation_fraction,
        target_shape=tuple(args.target_shape),
        data_dir=args.data_dir,
        output_dir=args.out,
    )

    print(f"FedMed imaging experiment: {config.site}")
    print(f"  data      : {config.data_dir or 'synthetic phantoms (generated)'}")
    print(f"  input     : {config.target_shape}, spacing {config.target_spacing}")
    print(f"  training  : {config.epochs} epochs, batch {config.batch_size}, "
          f"lr {config.learning_rate}, seed {config.seed}")
    print("  device    : cpu (no CUDA GPU on this machine)")
    print()

    report = run_experiment(config)

    print("epoch   train_loss   val_loss   val_dice   val_iou   secs")
    for record in report["result"]["epochs"]:
        print(
            f"{record['epoch']:>5}   {record['train_loss']:>10.4f}   "
            f"{(record['validation_loss'] if record['validation_loss'] is not None else float('nan')):>8.4f}   "
            f"{(record['validation_dice'] if record['validation_dice'] is not None else float('nan')):>8.4f}   "
            f"{(record['validation_iou'] if record['validation_iou'] is not None else float('nan')):>7.4f}   "
            f"{record['duration_seconds']:>4.1f}"
        )

    summary = report["result"]
    holdout = report["bestCheckpointEvaluation"]
    print()
    print(f"loss        : {summary['firstTrainLoss']:.4f} -> {summary['lastTrainLoss']:.4f} "
          f"(improved: {summary['lossImproved']})")
    print(f"best Dice   : {summary['bestDice']} at epoch {summary['bestEpoch']}")
    if holdout.get("dice") is not None:
        print(f"reloaded    : Dice {holdout['dice']:.4f} loss {holdout['loss']:.4f} "
              f"over {holdout['cases']} validation case(s)")
    print(f"parameters  : {report['model']['parameterCount']:,}")
    print(f"duration    : {summary['durationSeconds']:.1f}s")
    print()
    print(f"written     : {config.output_dir / 'experiment.json'}")
    print(f"checkpoints : {config.output_dir / 'checkpoints'}")
    print()
    print("These are measurements on synthetic phantoms from a local CPU run.")
    print("They are not clinical results and this model is not clinically validated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
