"""FedAvg aggregation.

This is a real implementation of sample-count-weighted Federated Averaging over
model parameters. It is deliberately independent of the training framework: it
operates on plain mappings of parameter name -> NumPy array, so it can be tested
on its own and reused by both the in-process simulation and the Flower strategy.

What FedAvg does here:

    w_global = sum_k (n_k / sum_j n_j) * w_k

where n_k is the number of local training examples the k-th client used. That
weighting is the whole point: a hospital with 400 studies should influence the
global model more than one with 40, in proportion to how much data it actually
trained on.

Everything that could make an average meaningless or unsafe is rejected:

    * mismatched parameter names between updates (incompatible structures)
    * mismatched shapes for the same name
    * weights that are missing, negative, NaN or infinite
    * all weights zero, which would make the total zero and the average undefined
    * non-finite parameter values (NaN/Inf), which would poison the global model
    * an empty update list

Averaging happens in float64 and is cast back to the input dtype at the end, so
repeated rounds do not accumulate avoidable rounding error in float32.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .errors import InvalidWeightError, UpdateValidationError

# A parameter update: parameter name -> array of values.
ParameterUpdate = Mapping[str, np.ndarray]


def _as_array(name: str, value: Any, client_label: str) -> np.ndarray:
    """Coerce one parameter to a NumPy array, rejecting anything unusable."""
    if isinstance(value, np.ndarray):
        array = value
    else:
        try:
            array = np.asarray(value)
        except Exception as exc:  # noqa: BLE001 - any failure means unusable
            raise UpdateValidationError(
                f"Parameter '{name}' from {client_label} could not be read as an array.",
                detail={"parameter": name, "client": client_label, "reason": str(exc)},
            ) from exc

    if array.dtype == object or array.dtype.kind not in "fiub":
        raise UpdateValidationError(
            f"Parameter '{name}' from {client_label} is not numeric "
            f"(dtype {array.dtype}).",
            detail={"parameter": name, "client": client_label, "dtype": str(array.dtype)},
        )
    if array.size == 0:
        raise UpdateValidationError(
            f"Parameter '{name}' from {client_label} is empty.",
            detail={"parameter": name, "client": client_label, "shape": list(array.shape)},
        )
    if not np.all(np.isfinite(array)):
        raise UpdateValidationError(
            f"Parameter '{name}' from {client_label} contains NaN or infinite values; "
            "it would corrupt the global model if averaged.",
            detail={"parameter": name, "client": client_label},
        )
    return array


def validate_weights(weights: Sequence[float], *, expected: int | None = None) -> np.ndarray:
    """Validate sample counts and return them as a float64 array.

    Weights must be finite, non-negative, and not all zero.
    """
    if expected is not None and len(weights) != expected:
        raise InvalidWeightError(
            f"Expected {expected} weights, received {len(weights)}.",
            detail={"expected": expected, "received": len(weights)},
        )
    if len(weights) == 0:
        raise InvalidWeightError("No weights were supplied for aggregation.")

    try:
        array = np.asarray(weights, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise InvalidWeightError(
            "Weights must be numbers.", detail={"reason": str(exc)}
        ) from exc

    if not np.all(np.isfinite(array)):
        raise InvalidWeightError(
            "Weights must be finite; NaN or infinite weights make the average "
            "undefined.",
            detail={"weights": [float(w) for w in np.nan_to_num(array)]},
        )
    if np.any(array < 0):
        raise InvalidWeightError(
            "Weights must not be negative.",
            detail={"weights": [float(w) for w in array]},
        )
    if float(array.sum()) <= 0.0:
        raise InvalidWeightError(
            "At least one weight must be greater than zero; all-zero weights give "
            "an undefined average.",
            detail={"weights": [float(w) for w in array]},
        )
    return array


def validate_updates(
    updates: Sequence[ParameterUpdate],
    *,
    expected_structure: Mapping[str, Any] | None = None,
    labels: Sequence[str] | None = None,
) -> list[str]:
    """Check that every update has the same parameters and shapes.

    Returns the sorted parameter names, so callers know the agreed structure.

    Raises UpdateValidationError naming both the offending client and parameter,
    because "shape mismatch" without those two facts is not actionable.
    """
    if len(updates) == 0:
        raise UpdateValidationError("No client updates were supplied for aggregation.")

    names = list(labels) if labels is not None else [f"update[{i}]" for i in range(len(updates))]
    if len(names) != len(updates):
        raise UpdateValidationError(
            "Each update needs a matching label for error reporting.",
            detail={"updates": len(updates), "labels": len(names)},
        )

    reference_keys: set[str] | None = None
    reference_label = ""
    shapes: dict[str, tuple[int, ...]] = {}

    for update, label in zip(updates, names, strict=True):
        if not isinstance(update, Mapping) or len(update) == 0:
            raise UpdateValidationError(
                f"{label} supplied no parameters.",
                detail={"client": label},
            )
        keys = set(update.keys())

        if reference_keys is None:
            reference_keys = keys
            reference_label = label
            for name in keys:
                shapes[name] = _as_array(name, update[name], label).shape
            continue

        missing = reference_keys - keys
        extra = keys - reference_keys
        if missing or extra:
            raise UpdateValidationError(
                f"{label} does not match the model structure of {reference_label}: "
                f"missing {sorted(missing)}, unexpected {sorted(extra)}.",
                detail={
                    "client": label,
                    "referenceClient": reference_label,
                    "missing": sorted(missing),
                    "unexpected": sorted(extra),
                },
            )

        for name in keys:
            array = _as_array(name, update[name], label)
            if array.shape != shapes[name]:
                raise UpdateValidationError(
                    f"Parameter '{name}' from {label} has shape "
                    f"{list(array.shape)} but {reference_label} has "
                    f"{list(shapes[name])}.",
                    detail={
                        "client": label,
                        "parameter": name,
                        "shape": list(array.shape),
                        "expectedShape": list(shapes[name]),
                    },
                )

    if expected_structure is not None:
        expected_keys = set(expected_structure.keys())
        assert reference_keys is not None
        if reference_keys != expected_keys:
            raise UpdateValidationError(
                "Client updates do not match the global model structure: "
                f"missing {sorted(expected_keys - reference_keys)}, "
                f"unexpected {sorted(reference_keys - expected_keys)}.",
                detail={
                    "missing": sorted(expected_keys - reference_keys),
                    "unexpected": sorted(reference_keys - expected_keys),
                },
            )
        for name, value in expected_structure.items():
            expected_shape = np.asarray(value).shape
            if shapes[name] != expected_shape:
                raise UpdateValidationError(
                    f"Parameter '{name}' has shape {list(shapes[name])} but the "
                    f"global model expects {list(expected_shape)}.",
                    detail={
                        "parameter": name,
                        "shape": list(shapes[name]),
                        "expectedShape": list(expected_shape),
                    },
                )

    assert reference_keys is not None
    return sorted(reference_keys)


def weighted_average(
    updates: Sequence[ParameterUpdate],
    weights: Sequence[float],
    *,
    expected_structure: Mapping[str, Any] | None = None,
    labels: Sequence[str] | None = None,
) -> dict[str, np.ndarray]:
    """Sample-count-weighted Federated Averaging.

    Args:
        updates: one mapping of parameter name -> array per participating client.
        weights: the number of local training examples behind each update. These
            are the FedAvg sample counts, not arbitrary coefficients.
        expected_structure: optional global model parameters to check shape
            compatibility against.
        labels: optional client names, used only to make errors readable.

    Returns:
        The aggregated parameters, matching the input dtype.

    Raises:
        InvalidWeightError: weights are missing, negative, non-finite or all zero.
        UpdateValidationError: updates disagree on structure, shape or values.
    """
    if len(updates) == 0:
        # Checked before the weights so the message names the real problem: with
        # no updates at all, "no weights supplied" would be misleading.
        raise UpdateValidationError("No client updates were supplied for aggregation.")

    if len(updates) != len(weights):
        raise InvalidWeightError(
            f"Received {len(updates)} updates but {len(weights)} weights.",
            detail={"updates": len(updates), "weights": len(weights)},
        )

    weight_array = validate_weights(weights, expected=len(updates))
    names = validate_updates(updates, expected_structure=expected_structure, labels=labels)

    # Normalise once: n_k / sum(n_j). The sum is guaranteed > 0 by validation.
    coefficients = weight_array / weight_array.sum()

    aggregated: dict[str, np.ndarray] = {}
    for name in names:
        first = updates[0][name]
        # Accumulate in float64 so many rounds of float32 weights do not drift.
        total = np.zeros(np.asarray(first).shape, dtype=np.float64)
        for coefficient, update in zip(coefficients, updates, strict=True):
            total += coefficient * np.asarray(update[name], dtype=np.float64)
        aggregated[name] = total.astype(np.asarray(first).dtype, copy=False)

    return aggregated


def aggregate_losses(losses: Sequence[float], weights: Sequence[float]) -> float:
    """Weighted mean of the per-client training losses.

    Averaging the reported loss by sample count gives a number comparable to what
    a centralised run over the same pooled data would report, which is the point
    of tracking it.
    """
    weight_array = validate_weights(weights, expected=len(losses))
    loss_array = np.asarray(losses, dtype=np.float64)
    if not np.all(np.isfinite(loss_array)):
        raise UpdateValidationError(
            "Client losses must be finite; a NaN loss would hide a broken client.",
            detail={"losses": [float(v) for v in np.nan_to_num(loss_array)]},
        )
    return float(np.dot(loss_array, weight_array) / weight_array.sum())
