"""Federated server orchestration.

This is the server side of the loop, written so it does not care whether the
clients are objects in this process or processes on the far side of a gRPC
connection:

    1. send the current global parameters to the selected clients
    2. each client trains locally and returns parameters plus its example count
    3. aggregate with sample-count-weighted FedAvg (fedavg.weighted_average)
    4. evaluate the aggregated model and record the round

Two concrete drivers use it:

    * `LocalFederatedRunner` (this module) runs the loop in one process, with
      clients built in memory. This is what the API uses, and it is labelled
      `local-simulation`.
    * `flower_app.py` runs it across separate Flower processes over gRPC, and is
      labelled `distributed-local`.

Neither proves deployment across separate hospital infrastructures; see
docs/FEDERATED.md. What is real in both cases is the arithmetic: parameters
computed by genuine local training, averaged by genuine sample counts.
"""

from __future__ import annotations

import importlib
import threading
import uuid
from typing import TYPE_CHECKING, Any

from .config import RoundConfig
from .errors import ClientTrainingError, NoEligibleClientsError
from .fedavg import weighted_average
from .results import (
    ClientOutcome,
    ExecutionMode,
    FederatedRun,
    RoundResult,
    RunStatus,
    utc_now,
)

if TYPE_CHECKING:  # pragma: no cover - annotations only, never imported at runtime
    from .client import ClientRegistry, ClientUpdate


def _torch_module(name: str):
    """Import one of the torch-backed modules (``client``, ``data``, ``model``).

    All three import torch. torch is deliberately **not** imported when the API
    starts, for two reasons:

    1. The dashboard must boot and stay usable in a deployment whose training
       component is absent or offline. That is the documented arrangement for the
       public demo, and it is what requirement 5 of the deployment brief asks for.
    2. torch costs roughly half a gigabyte on disk and ~200 MB resident. A
       read-only dashboard should not pay that to render numbers.

    Importing the real modules here keeps a *single* implementation. The
    alternative — a second, torch-free copy of the federated logic — would be a
    duplicate system that could drift from the tested one.
    """
    return importlib.import_module(f".{name}", __package__)


class FederatedServer:
    """Runs federated rounds over a registry of clients."""

    def __init__(
        self,
        config: RoundConfig,
        registry: ClientRegistry,
        *,
        execution: ExecutionMode = ExecutionMode.local_simulation,
    ) -> None:
        self.config = config
        self.registry = registry
        self.execution = execution

    def initial_parameters(self) -> "dict[str, Any]":
        """The model a run starts from, seeded so runs are reproducible."""
        model = _torch_module("model").build_model(seed=self.config.seed)
        return _torch_module("model").state_to_numpy(model)

    def run_round(
        self,
        round_number: int,
        global_parameters: dict[str, Any],
    ) -> tuple[RoundResult, dict[str, Any]]:
        """Execute one round and return the result plus the new global parameters.

        A client that fails is recorded as excluded and the round continues,
        provided enough clients remain to meet `min_clients`. Failing the whole
        run because one site had a bad day would be the wrong behaviour, and
        silently substituting that client's previous update would be worse.
        """
        started = utc_now()
        outcomes: list[ClientOutcome] = []
        updates: list[ClientUpdate] = []
        labels: list[str] = []

        for name in self.registry.names():
            client = self.registry.get(name)
            if client is None:
                continue

            if name in self.config.fail_clients:
                # Simulated failure, used to exercise the exclusion path.
                outcomes.append(
                    ClientOutcome(
                        client=name,
                        participated=False,
                        failure="simulated client failure (configured for this run)",
                    )
                )
                continue

            try:
                update = client.fit(global_parameters)
            except ClientTrainingError as exc:
                # A real failure. Recorded with its reason, client excluded.
                outcomes.append(
                    ClientOutcome(client=name, participated=False, failure=exc.message)
                )
                continue

            updates.append(update)
            labels.append(name)

        if len(updates) < self.config.min_clients:
            raise NoEligibleClientsError(
                f"Round {round_number} had {len(updates)} usable client updates but "
                f"requires at least {self.config.min_clients}.",
                detail={
                    "round": round_number,
                    "usable": len(updates),
                    "required": self.config.min_clients,
                    "excluded": [
                        {"client": outcome.client, "failure": outcome.failure}
                        for outcome in outcomes
                        if not outcome.participated
                    ],
                },
            )

        weights = [update.num_examples for update in updates]
        total_weight = sum(weights)

        aggregated = weighted_average(
            [update.parameters for update in updates],
            weights,
            expected_structure=global_parameters,
            labels=labels,
        )

        # Weighted mean of the client losses. This is the round's training loss.
        aggregated_loss = sum(
            update.train_loss * update.num_examples for update in updates
        ) / total_weight

        for update in updates:
            outcomes.append(
                ClientOutcome(
                    client=update.client,
                    participated=True,
                    num_examples=update.num_examples,
                    weight=update.num_examples / total_weight,
                    train_loss=update.train_loss,
                    validation_loss=update.validation_loss,
                    validation_dice=update.validation_dice,
                    duration_seconds=update.duration_seconds,
                )
            )

        # Score the aggregated model. Each client scores it on its own held-back
        # data, which the server never sees; the reported figure is the
        # example-weighted mean of those scores.
        validation_scores: list[tuple[int, float, float]] = []
        for name in labels:
            client = self.registry.get(name)
            if client is None:
                continue
            scored = client.evaluate(aggregated)
            if scored is None:
                continue
            validation_scores.append(
                (client.training_examples, scored["validation_loss"], scored["validation_dice"])
            )

        if validation_scores:
            weight_sum = sum(weight for weight, _, _ in validation_scores)
            aggregated_validation_loss = sum(
                weight * loss for weight, loss, _ in validation_scores
            ) / weight_sum
            aggregated_validation_dice = sum(
                weight * dice for weight, _, dice in validation_scores
            ) / weight_sum
        else:
            aggregated_validation_loss = None
            aggregated_validation_dice = None

        # Keep the outcomes in a stable, predictable order.
        outcomes.sort(key=lambda outcome: outcome.client)

        finished = utc_now()
        result = RoundResult(
            round=round_number,
            strategy=self.config.strategy,
            outcomes=outcomes,
            aggregated_loss=aggregated_loss,
            aggregated_validation_loss=aggregated_validation_loss,
            aggregated_validation_dice=aggregated_validation_dice,
            total_examples=total_weight,
            duration_seconds=(finished - started).total_seconds(),
            started_at=started,
            finished_at=finished,
            aggregated_parameters=sum(
                int(array.size) for array in aggregated.values()
            ),
        )
        return result, aggregated

    def run(
        self,
        *,
        on_round: "Any | None" = None,
        should_stop: "Any | None" = None,
    ) -> FederatedRun:
        """Run every configured round, recording results as they complete.

        `on_round` is called after each round and `should_stop` is polled before
        each one, which is what lets the API report progress and honour a stop
        request without this class knowing about HTTP.
        """
        run = FederatedRun(
            run_id=f"fl-{uuid.uuid4().hex[:12]}",
            config=self.config,
            status=RunStatus.running,
            execution=self.execution,
            started_at=utc_now(),
            seed=self.config.seed,
            clients=self.registry.describe(),
            model_parameters=_torch_module("model").parameter_count(
                _torch_module("model").build_model()
            ),
        )
        run.message = (
            f"{self.config.describe()}. Execution: {self.execution.value}. "
            "Local training is genuine; the data is synthetic."
        )

        parameters = self.initial_parameters()

        try:
            for round_number in range(1, self.config.rounds + 1):
                if should_stop is not None and should_stop():
                    run.status = RunStatus.stopped
                    run.message = (
                        f"Stopped after {run.completed_rounds} of "
                        f"{self.config.rounds} rounds on request."
                    )
                    break

                run.current_round = round_number
                if on_round is not None:
                    on_round(run)

                result, parameters = self.run_round(round_number, parameters)

                run.rounds.append(result)
                run.current_round = None
                if on_round is not None:
                    on_round(run)
            else:
                run.status = RunStatus.completed
                run.message = self._completion_message(run)
        except Exception as exc:  # noqa: BLE001 - recorded, not hidden
            run.status = RunStatus.failed
            run.error = str(exc)
            run.message = f"Run failed in round {run.current_round or run.completed_rounds}: {exc}"
        finally:
            run.current_round = None
            run.finished_at = utc_now()

        return run

    def _completion_message(self, run: FederatedRun) -> str:
        if run.first_loss is None or run.last_loss is None:
            return "Completed with no rounds recorded."
        direction = "fell" if run.last_loss < run.first_loss else "did not fall"
        return (
            f"Completed {run.completed_rounds} of {run.config.rounds} rounds. "
            f"Aggregated training loss {direction} from "
            f"{run.first_loss:.4f} to {run.last_loss:.4f}. These are measurements "
            f"from a {run.execution.value} over synthetic data, not a cross-silo "
            "deployment and not a result on real scans."
        )


class LocalFederatedRunner:
    """Runs one federated experiment in a background thread.

    The API needs to start a run and return immediately, then report progress.
    A thread is the simplest thing that does that honestly: the work really does
    happen, the status really is live, and `request_stop` really does stop it
    between rounds. A run that has finished cannot be restarted; start a new one.

    Not a job queue, not distributed, and not durable: the run lives in this
    process's memory and is gone when the process exits.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._run: FederatedRun | None = None
        self._stop_requested = False

    # -- state -------------------------------------------------------------
    @property
    def run(self) -> FederatedRun | None:
        with self._lock:
            return self._run

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._run is not None and self._run.status is RunStatus.running

    # -- control -----------------------------------------------------------
    def start(self, config: RoundConfig) -> FederatedRun:
        """Start a run. Raises RuntimeError if one is already in progress."""
        with self._lock:
            if self._run is not None and self._run.status is RunStatus.running:
                raise RuntimeError(
                    "A federated run is already in progress. Stop it before starting another."
                )
            datasets = _torch_module("data").build_client_datasets(config)
            registry = _torch_module("client").ClientRegistry.build(config, datasets)
            server = FederatedServer(config, registry)

            run = FederatedRun(
                run_id=f"fl-{uuid.uuid4().hex[:12]}",
                config=config,
                status=RunStatus.running,
                execution=ExecutionMode.local_simulation,
                started_at=utc_now(),
                seed=config.seed,
                clients=registry.describe(),
                model_parameters=_torch_module("model").parameter_count(
                    _torch_module("model").build_model()
                ),
            )
            run.message = f"Starting: {config.describe()}"
            self._run = run
            self._stop_requested = False

            self._thread = threading.Thread(
                target=self._execute, args=(server,), name="fedmed-federated-run", daemon=True
            )
            self._thread.start()
            return run

    def _execute(self, server: FederatedServer) -> None:
        """Run the experiment, replacing the placeholder run when it finishes."""
        try:
            finished = server.run(
                on_round=self._publish,
                should_stop=lambda: self._stop_requested,
            )
            with self._lock:
                self._run = finished
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                if self._run is not None:
                    self._run.status = RunStatus.failed
                    self._run.error = str(exc)
                    self._run.message = f"Run failed to start: {exc}"
                    self._run.finished_at = utc_now()

    def _publish(self, run: FederatedRun) -> None:
        with self._lock:
            self._run = run

    def request_stop(self) -> bool:
        """Ask the run to stop after the current round. True if one was running."""
        with self._lock:
            if self._run is None or self._run.status is not RunStatus.running:
                return False
            self._stop_requested = True
            return True

    def reset(self) -> None:
        """Clear any finished run. Refuses while one is still running."""
        with self._lock:
            if self._run is not None and self._run.status is RunStatus.running:
                raise RuntimeError("Cannot reset while a federated run is in progress.")
            self._run = None
            self._stop_requested = False

    def stop_and_wait(self, timeout: float = 60.0) -> bool:
        """Ask a running experiment to stop and wait for the thread to finish.

        Needed for orderly shutdown and for test teardown: clearing a run while
        its worker is still writing to it would be a race. Returns True if the
        runner is quiescent afterwards.
        """
        with self._lock:
            thread = self._thread
            running = self._run is not None and self._run.status is RunStatus.running
            if running:
                self._stop_requested = True

        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)

        with self._lock:
            still_running = self._run is not None and self._run.status is RunStatus.running
            if not still_running:
                self._thread = None
            return not still_running

    def force_reset(self, timeout: float = 60.0) -> None:
        """Stop any run, wait for it, then clear. Safe to call in teardown."""
        self.stop_and_wait(timeout=timeout)
        with self._lock:
            self._run = None
            self._stop_requested = False
            self._thread = None

    def history(self) -> list[dict[str, Any]]:
        """Round-by-round history of the current run, oldest first."""
        with self._lock:
            if self._run is None:
                return []
            return [result.as_dict() for result in self._run.rounds]


# One runner per process, matching the single-in-memory-run design.
local_runner = LocalFederatedRunner()
