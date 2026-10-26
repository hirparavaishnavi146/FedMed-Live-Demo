"""Flower client and server apps: the same federation across separate processes.

VERIFIED AGAINST THE INSTALLED PACKAGE
    Flower 1.39.0, inspected directly rather than assumed. The APIs used here are:

        flwr.client.NumPyClient          fit(parameters, config) -> (params, num_examples, metrics)
                                         evaluate(parameters, config) -> (loss, num_examples, metrics)
        flwr.client.start_client         start_client(*, server_address, client=..., insecure=...)
        flwr.server.start_server         start_server(*, server_address, strategy=..., config=...)
        flwr.server.strategy.FedAvg      aggregate_fit(server_round, results, failures)
        flwr.common                      ndarrays_to_parameters, parameters_to_ndarrays, FitRes

    Flower 1.39 also ships a newer ServerApp/ClientApp API (flwr.app, ArrayRecord,
    Message, Context). This module deliberately uses the classic API, because
    `FitRes.num_examples` supplies the sample count that FedAvg weights by, which
    is exactly the quantity this project must weight on.

ONE AGGREGATION IMPLEMENTATION
    `FedMedFedAvg` overrides Flower's aggregation and delegates to
    `fedavg.weighted_average`, the same function the in-process runner uses. So
    there is a single FedAvg implementation in the project, and the distributed
    path cannot drift from the tested one. Flower still does what it is for:
    client/server orchestration and transport.

WHAT THIS IS
    A real client/server split: server and clients are separate OS processes
    talking over gRPC. The client processes cannot read each other's data because
    they share no memory.

WHAT THIS IS NOT
    A cross-silo deployment. All processes are on one machine, the transport is
    plaintext gRPC on loopback (`insecure=True`), there is no authentication, and
    the same person starts every process. See docs/FEDERATED.md.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from flwr.client import NumPyClient, start_client
from flwr.common import (
    Code,
    FitRes,
    Status,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server import ServerConfig, start_server
from flwr.server.strategy import FedAvg

from .client import HospitalClient
from .config import RoundConfig
from .data import DEFAULT_PROFILES, generate_dataset
from .errors import ClientTrainingError
from .fedavg import weighted_average
from .model import build_model, state_to_numpy
from .results import ExecutionMode, utc_now

DEFAULT_SERVER_ADDRESS = "127.0.0.1:8080"


# ---------------------------------------------------------------------------
# Parameter ordering.
#
# Flower transports parameters as an anonymous list of arrays, so the mapping
# between array positions and parameter names has to be agreed by both sides or
# the model is silently scrambled. Ordering by sorted name and asserting the
# names match on arrival makes that agreement explicit rather than accidental.
# ---------------------------------------------------------------------------
def parameter_names() -> list[str]:
    """The canonical parameter order: sorted by name, stable across processes."""
    return sorted(state_to_numpy(build_model()).keys())


def state_to_ndarrays(state: dict[str, np.ndarray]) -> list[np.ndarray]:
    """Model state as an ordered list, in the canonical parameter order."""
    names = parameter_names()
    missing = set(names) - set(state.keys())
    if missing:
        raise ClientTrainingError(
            f"Model state is missing parameters: {sorted(missing)}",
            code="incomplete_model_state",
            detail={"missing": sorted(missing)},
        )
    return [np.ascontiguousarray(state[name]) for name in names]


def ndarrays_to_state(arrays: Sequence[np.ndarray]) -> dict[str, np.ndarray]:
    """Ordered list back to named model state, checking the count matches."""
    names = parameter_names()
    if len(arrays) != len(names):
        raise ClientTrainingError(
            f"Received {len(arrays)} parameter arrays but the model has {len(names)}.",
            code="parameter_count_mismatch",
            detail={"received": len(arrays), "expected": len(names)},
        )
    return {name: np.asarray(array) for name, array in zip(names, arrays, strict=True)}


def build_hospital_client(label: str, config: RoundConfig) -> HospitalClient:
    """Build the client for one hospital, from its own seeded data."""
    profile = next((p for p in DEFAULT_PROFILES if p.name == label), None)
    if profile is None:
        raise ClientTrainingError(
            f"Unknown hospital '{label}'. Known: {[p.name for p in DEFAULT_PROFILES]}",
            code="unknown_hospital",
            detail={"label": label, "known": [p.name for p in DEFAULT_PROFILES]},
        )
    dataset = generate_dataset(profile, seed=config.seed, patch_size=config.patch_size)
    return HospitalClient(label, dataset, config)


class HospitalNumPyClient(NumPyClient):
    """A Flower client that trains one simulated hospital locally.

    Each `fit` call returns parameters and the number of local training examples.
    Training data stays in this process: the return value carries no examples, and
    no method here exposes any.
    """

    def __init__(self, label: str, config: RoundConfig) -> None:
        self.label = label
        self.config = config
        # Held privately, as in the in-process client.
        self._client = build_hospital_client(label, config)

    def get_parameters(self, config: dict[str, Any]) -> list[np.ndarray]:
        """Starting parameters for this client.

        Only used if the server starts without initial parameters. The default
        path seeds the server, so every client begins from the same global model.
        """
        return state_to_ndarrays(state_to_numpy(build_model(seed=self.config.seed)))

    def fit(
        self, parameters: list[np.ndarray], config: dict[str, Any]
    ) -> tuple[list[np.ndarray], int, dict[str, Any]]:
        """Train locally and return (parameters, num_examples, metrics)."""
        global_state = ndarrays_to_state(parameters)
        update = self._client.fit(global_state)

        # `num_examples` is Flower's FedAvg weight and is deliberately the local
        # training-set size, not the dataset total.
        metrics: dict[str, Any] = {
            "hospital": self.label,
            "train_loss": round(update.train_loss, 6),
            "local_epochs": update.local_epochs,
            "duration_seconds": round(update.duration_seconds, 4),
            "seed": update.seed,
        }
        if update.validation_loss is not None:
            metrics["validation_loss"] = round(update.validation_loss, 6)
        if update.validation_dice is not None:
            metrics["validation_dice"] = round(update.validation_dice, 6)

        return state_to_ndarrays(update.parameters), update.num_examples, metrics

    def evaluate(
        self, parameters: list[np.ndarray], config: dict[str, Any]
    ) -> tuple[float, int, dict[str, Any]]:
        """Score the aggregated model on this client's own validation split."""
        scored = self._client.evaluate(ndarrays_to_state(parameters))
        if scored is None:
            return 0.0, 0, {"hospital": self.label, "has_validation_split": False}
        return (
            float(scored["validation_loss"]),
            self._client.validation_examples,
            {
                "hospital": self.label,
                "validation_dice": round(float(scored["validation_dice"]), 6),
                "has_validation_split": True,
            },
        )


class FedMedFedAvg(FedAvg):
    """Flower's FedAvg with the aggregation delegated to this project's own.

    Flower handles orchestration, transport and client selection. The averaging
    itself is `fedavg.weighted_average`, the same function the in-process runner
    calls and the one covered by the aggregation tests, so the distributed path
    cannot diverge from the tested one.
    """

    def __init__(self, *args: Any, recorder: "Callable[[dict[str, Any]], None] | None" = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._recorder = recorder

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[Any, FitRes]],
        failures: list[Any],
    ) -> tuple[Any, dict[str, Any]]:
        """Average client updates by sample count, using this project's FedAvg."""
        if not results:
            # Nothing to average. Reporting zero weights would fabricate a round.
            return None, {}

        started = utc_now()
        updates: list[dict[str, np.ndarray]] = []
        weights: list[int] = []
        labels: list[str] = []
        client_metrics: list[dict[str, Any]] = []

        for proxy, fit_res in results:
            arrays = parameters_to_ndarrays(fit_res.parameters)
            updates.append(ndarrays_to_state(arrays))
            weights.append(int(fit_res.num_examples))
            metrics = dict(getattr(fit_res, "metrics", None) or {})
            # Prefer the hospital's own name over Flower's assigned cid, which is
            # just a connection index and would be meaningless in a report.
            label = str(metrics.get("hospital") or getattr(proxy, "cid", "unknown"))
            labels.append(label)
            client_metrics.append(metrics)

        aggregated = weighted_average(updates, weights, labels=labels)
        total_examples = sum(weights)
        weighted_loss = sum(
            float(metrics.get("train_loss") or 0.0) * weight
            for metrics, weight in zip(client_metrics, weights, strict=True)
        ) / total_examples

        record = {
            "round": server_round,
            "strategy": "FedAvg",
            "measured": True,
            "execution": ExecutionMode.distributed_local.value,
            "participatingClients": labels,
            "weights": [weight / total_examples for weight in weights],
            "numExamples": weights,
            "aggregatedLoss": round(weighted_loss, 6),
            "clientMetrics": client_metrics,
            "failures": len(failures),
            "durationSeconds": round((utc_now() - started).total_seconds(), 4),
        }
        if self._recorder is not None:
            self._recorder(record)

        print(
            f"[round {server_round}] aggregated {len(results)} updates "
            f"({total_examples} examples) -> loss {weighted_loss:.4f}"
            + (f", {len(failures)} failure(s)" if failures else "")
        )

        return (
            ndarrays_to_parameters(state_to_ndarrays(aggregated)),
            {
                "aggregated_loss": round(weighted_loss, 6),
                "participating_clients": len(results),
                "total_examples": total_examples,
            },
        )

    def aggregate_evaluate(self, server_round: int, results: list[Any], failures: list[Any]) -> tuple[Any, dict[str, Any]]:
        """Example-weighted mean of the clients' validation scores."""
        if not results:
            return None, {}

        total = sum(int(res.num_examples) for _, res in results) or 1
        weighted_loss = sum(
            float(res.loss) * int(res.num_examples) for _, res in results
        ) / total
        dice_values = [
            float((res.metrics or {}).get("validation_dice"))
            for _, res in results
            if (res.metrics or {}).get("validation_dice") is not None
        ]
        metrics: dict[str, Any] = {"validation_loss": round(weighted_loss, 6)}
        if dice_values:
            metrics["validation_dice"] = round(sum(dice_values) / len(dice_values), 6)
        return weighted_loss, metrics


def build_strategy(
    config: RoundConfig,
    *,
    recorder: "Callable[[dict[str, Any]], None] | None" = None,
) -> FedAvg:
    """A Flower strategy wired to this project's FedAvg, starting from one model."""
    initial = state_to_ndarrays(state_to_numpy(build_model(seed=config.seed)))
    return FedMedFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=config.min_clients,
        min_evaluate_clients=config.min_clients,
        min_available_clients=config.clients,
        initial_parameters=ndarrays_to_parameters(initial),
        recorder=recorder,
    )


def summarise_run(records: list[dict[str, Any]], config: RoundConfig) -> dict[str, Any]:
    """Build a run summary from the rounds the strategy recorded.

    A pure function so it can be tested without starting a server. It is separate
    from Flower's `History` on purpose: an earlier version read
    `history.metrics_distributed`, which is empty unless the strategy is given a
    `fit_metrics_aggregation_fn`, and silently summarised a real run as having no
    losses at all.
    """
    aggregated_losses = [float(entry["aggregatedLoss"]) for entry in records]
    return {
        "rounds": len(records),
        "aggregatedLosses": aggregated_losses,
        "participatingClients": [len(entry["participatingClients"]) for entry in records],
        "totalExamples": [sum(entry["numExamples"]) for entry in records],
        "failedRounds": [entry["round"] for entry in records if entry["failures"]],
        "firstLoss": aggregated_losses[0] if aggregated_losses else None,
        "lastLoss": aggregated_losses[-1] if aggregated_losses else None,
        "lossImproved": (
            bool(aggregated_losses) and aggregated_losses[-1] < aggregated_losses[0]
        ),
        "clientsPerRound": [len(entry["participatingClients"]) for entry in records],
        "execution": ExecutionMode.distributed_local.value,
        "seed": config.seed,
        "config": config.as_dict(),
    }


def run_server(
    config: RoundConfig,
    *,
    server_address: str = DEFAULT_SERVER_ADDRESS,
    recorder: "Callable[[dict[str, Any]], None] | None" = None,
) -> dict[str, Any]:
    """Run a Flower server for `config.rounds` rounds. Blocks until finished.

    The strategy records each round as it aggregates, so the summary is built
    from records this code produced rather than from Flower's `History` object.
    That matters because `losses_distributed` holds only the metric vectors the
    strategy returns, and `metrics_distributed` stays empty unless the strategy is
    given a `fit_metrics_aggregation_fn`. Reading either silently yielded an empty
    summary. The strategy's own records cannot drift from what it aggregated.
    """
    records: list[dict[str, Any]] = []

    def record(entry: dict[str, Any]) -> None:
        records.append(entry)
        if recorder is not None:
            recorder(entry)

    strategy = build_strategy(config, recorder=record)
    print(
        f"FedMed Flower server on {server_address}: {config.describe()}\n"
        f"  execution: {ExecutionMode.distributed_local.value} "
        "(separate processes, plaintext gRPC on loopback, not a cross-silo deployment)\n"
        f"  waiting for {config.clients} clients..."
    )
    start_server(
        server_address=server_address,
        config=ServerConfig(num_rounds=config.rounds),
        strategy=strategy,
    )

    summary = summarise_run(records, config)
    print(
        f"server finished: rounds={summary['rounds']} "
        f"losses={[round(value, 4) for value in summary['aggregatedLosses']]}"
    )
    print(
        f"  execution {summary['execution']}: separate processes over plaintext gRPC "
        "on loopback. Not a cross-silo deployment."
    )
    return summary


def run_client(
    label: str,
    config: RoundConfig,
    *,
    server_address: str = DEFAULT_SERVER_ADDRESS,
) -> None:
    """Run one hospital client against a Flower server. Blocks until finished."""
    client = HospitalNumPyClient(label, config)
    print(
        f"FedMed client '{label}' connecting to {server_address} "
        f"(local training examples: {client._client.training_examples}, "
        f"validation: {client._client.validation_examples})"
    )
    start_client(
        server_address=server_address,
        client=client,
        # Plaintext gRPC. Deliberate and stated rather than left implicit: there is
        # no TLS here, so nothing on this transport is confidential.
        insecure=True,
    )
    print(f"client '{label}' finished")


def _client_status_ok(fit_res: FitRes) -> bool:
    """Helper kept for callers that inspect FitRes status codes."""
    status = getattr(fit_res, "status", None)
    if status is None:
        return True
    return status.code == Code.OK and status == Status(code=Code.OK, message="")


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point: `python -m app.federated.flower_app server|client ...`"""
    import argparse

    parser = argparse.ArgumentParser(description="FedMed Flower server or client")
    parser.add_argument("role", choices=["server", "client"])
    parser.add_argument("--address", default=DEFAULT_SERVER_ADDRESS)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--local-epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--clients", type=int, default=3)
    parser.add_argument("--min-clients", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--hospital", default=None, help="client role: HOSP-01/02/03")
    args = parser.parse_args(argv)

    config = RoundConfig(
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        learning_rate=args.learning_rate,
        clients=args.clients,
        min_clients=args.min_clients,
        seed=args.seed,
        patch_size=args.patch_size,
    )

    if args.role == "server":
        run_server(config, server_address=args.address)
        return 0

    if not args.hospital:
        print("client role needs --hospital HOSP-01|HOSP-02|HOSP-03", file=sys.stderr)
        return 2
    run_client(args.hospital, config, server_address=args.address)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
