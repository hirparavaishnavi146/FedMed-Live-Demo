"""CLI for an in-process federated run.

    python -m app.federated.run_local --rounds 5 --clients 3 --seed 42

Prints each round as it completes. This is the same orchestration the API uses
(`FederatedServer`), with the same FedAvg, so the numbers here and the numbers in
the dashboard come from one implementation.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from .client import ClientRegistry
from .config import RoundConfig
from .data import build_client_datasets
from .errors import FederatedError
from .simulation import FederatedServer


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a FedMed federated experiment in one process."
    )
    parser.add_argument("--rounds", type=int, default=5, help="number of federated rounds")
    parser.add_argument("--local-epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--clients", type=int, default=3)
    parser.add_argument("--min-clients", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42, help="seeds data, init and batching")
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--json", action="store_true", help="print the full run as JSON")
    parser.add_argument(
        "--fail-client",
        action="append",
        default=[],
        help="client to fail (exercises the exclusion path); repeatable",
    )
    args = parser.parse_args(argv)

    try:
        config = RoundConfig(
            rounds=args.rounds,
            local_epochs=args.local_epochs,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            clients=args.clients,
            min_clients=args.min_clients,
            seed=args.seed,
            patch_size=args.patch_size,
            fail_clients=tuple(args.fail_client),
        )
    except FederatedError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2

    datasets = build_client_datasets(config)
    registry = ClientRegistry.build(config, datasets)

    print(f"FedMed federated run: {config.describe()}")
    print(f"execution: local-simulation (one process; not a cross-silo deployment)")
    print("clients (their data never leaves them):")
    for entry in registry.describe():
        print(
            f"  {entry['name']}: {entry['trainingExamples']} train / "
            f"{entry['validationExamples']} validation examples - {entry['description']}"
        )
    print()

    server = FederatedServer(config, registry)
    run = server.run()

    for result in run.rounds:
        participants = ",".join(result.participating)
        print(
            f"round {result.round}/{config.rounds}: "
            f"loss={result.aggregated_loss:.4f} "
            f"val_dice={result.aggregated_validation_dice if result.aggregated_validation_dice is None else format(result.aggregated_validation_dice, '.4f')} "
            f"clients=[{participants}] "
            f"{result.duration_seconds:.2f}s"
        )
        for outcome in result.excluded:
            print(f"    excluded {outcome.client}: {outcome.failure}")

    print()
    print(run.message)
    if run.error:
        print(f"error: {run.error}", file=sys.stderr)

    if args.json:
        print(json.dumps(run.as_dict(), indent=2))

    return 0 if run.status.value != "failed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
