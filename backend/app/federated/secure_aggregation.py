"""Secure aggregation (SecAgg+ style masking) built on Flower's own primitive.

WHAT THIS ACTUALLY PROVIDES

    Each client masks its quantized update with pseudorandom masks derived from
    pairwise shared seeds, adding the mask for every partner ranked above it and
    subtracting the mask for every partner ranked below it. The server receives only
    masked vectors. When every participant is present the masks cancel exactly and
    the server recovers the weighted sum — but it never sees any individual update.

    The pseudorandom generator is `flwr.common.secure_aggregation.secaggplus_utils.
    pseudo_rand_gen`, i.e. Flower's own maintained implementation from the version
    this project pins, not a reimplementation.

VERIFIED PROTOCOL FACTS (see tests/test_secure_aggregation.py)

    * Masks cancel exactly: the recovered sum equals the quantized sum.
    * A client's masked vector is not its real vector.
    * ``pseudo_rand_gen`` requires a seed whose length is a multiple of 4 bytes.
      A shorter seed raises AssertionError, which is why seeds here are derived
      through SHA-256 and used whole (32 bytes).
    * Arithmetic is modulo ``modulus_range`` and therefore requires the true sum to
      be smaller than the modulus. That is an assumption, not a guarantee.

PROTOCOL ASSUMPTIONS AND LIMITS — do not skip these

    * **All participants must complete.** If a client drops after masking, its masks
      do not cancel and the sum is wrong. This implementation refuses to unmask when
      the participant set is incomplete rather than returning a corrupted aggregate.
      Real SecAgg+ handles this with secret sharing and a reconstruction threshold,
      which is not implemented here.
    * **Requires a secure channel for key agreement.** Seeds are derived from a
      shared secret supplied out of band. In a real deployment this comes from an
      authenticated Diffie-Hellman exchange; here it is configuration.
    * **No protection against collusion.** The server plus all-but-one client can
      recover the remaining client's update by subtraction. SecAgg+ protects against
      a honest-but-curious server, nothing stronger.
    * **Quantization error.** Updates are clipped and rounded to integers, so the
      aggregate differs slightly from plain FedAvg. The amount is reported, and it
      is a real accuracy cost paid for the privacy property.
    * **Client-level, not record-level.** The unit that is hidden is one client's
      entire update. This says nothing about individual records inside that client's
      dataset.

    This is a genuine masking implementation, and it is also not a production
    secure-aggregation deployment. See docs/SECURITY.md.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np

from .errors import FederatedError

#: SecAgg+ bounds, taken from Flower 1.39's own documented defaults
#: (SecAggPlusWorkflow: clipping_range 8.0, quantization_range 2**22,
#: modulus_range 2**32) so the two implementations agree. 2**32 also stays inside
#: what `pseudo_rand_gen` can generate: it calls np.random.RandomState.randint with
#: int64, so a modulus of 2**64 raises "high is out of bounds for int64".
DEFAULT_QUANTIZATION_RANGE = 2**22
DEFAULT_MODULUS_RANGE = 2**32
DEFAULT_CLIPPING_RANGE = 8.0

#: The generator asserts this; deriving 32-byte seeds satisfies it automatically.
SEED_LENGTH = 32


class SecureAggregationError(FederatedError):
    """The protocol could not complete correctly."""

    code = "secure_aggregation_failed"


@dataclass(frozen=True)
class SecureAggregationConfig:
    """Parameters of the masking protocol.

    `enabled` is explicit rather than inferred, so a caller cannot enable the
    protocol by forgetting to pass a flag.
    """

    enabled: bool = False
    clipping_range: float = DEFAULT_CLIPPING_RANGE
    quantization_range: int = DEFAULT_QUANTIZATION_RANGE
    modulus_range: int = DEFAULT_MODULUS_RANGE
    #: Documented for completeness and reported, not used: this implementation has
    #: no secret sharing, so it cannot tolerate a dropout.
    reconstruction_threshold: int | None = None
    num_shares: int | None = None

    def __post_init__(self) -> None:
        if self.clipping_range <= 0:
            raise ValueError(f"clipping_range must be positive, got {self.clipping_range}.")
        for name, value in (
            ("quantization_range", self.quantization_range),
            ("modulus_range", self.modulus_range),
        ):
            if value < 2 or (value & (value - 1)) != 0:
                raise ValueError(
                    f"{name} must be a power of two >= 2, got {value}. The protocol "
                    "needs exact modular arithmetic."
                )
        if self.quantization_range >= self.modulus_range:
            raise ValueError(
                f"quantization_range ({self.quantization_range}) must be smaller than "
                f"modulus_range ({self.modulus_range}), or a sum cannot be recovered."
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "protocol": "SecAgg+ style pairwise masking",
            "clippingRange": self.clipping_range,
            "quantizationRange": self.quantization_range,
            "modulusRange": self.modulus_range,
            "numShares": self.num_shares,
            "reconstructionThreshold": self.reconstruction_threshold,
            "primitive": "flwr.common.secure_aggregation.secaggplus_utils.pseudo_rand_gen",
            "toleratesDropout": False,
        }


def pair_seed(secret: bytes, label_a: str, label_b: str) -> bytes:
    """Derive a pairwise seed, order-independent and always 32 bytes.

    Ordering the labels means both parties compute the same seed without agreeing on
    who is first. SHA-256 gives the multiple-of-4 length that the generator requires.
    """
    if not secret:
        raise ValueError("A shared secret is required to derive pairwise seeds.")
    ordered = sorted((label_a, label_b))
    digest = hashlib.sha256()
    digest.update(secret)
    digest.update(b"\x00")
    digest.update(ordered[0].encode("utf-8"))
    digest.update(b"\x00")
    digest.update(ordered[1].encode("utf-8"))
    return digest.digest()[:SEED_LENGTH]


def quantize(values: np.ndarray, clipping_range: float, quantization_range: int) -> np.ndarray:
    """Clip to [-clipping_range, clipping_range] and map to integers.

    Clipping is what bounds an individual client's influence; without it a single
    large update would dominate the aggregate and amplify quantization error.
    """
    clipped = np.clip(np.asarray(values, dtype=np.float64), -clipping_range, clipping_range)
    # Scale [-c, c] onto [-q/2, q/2] and round to the nearest integer.
    scaled = clipped * (quantization_range / (2.0 * clipping_range))
    return np.rint(scaled).astype(np.int64)


def dequantize(
    integers: np.ndarray, clipping_range: float, quantization_range: int
) -> np.ndarray:
    """Inverse of `quantize`, without the clipping."""
    scale = (2.0 * clipping_range) / quantization_range
    return np.asarray(integers, dtype=np.float64) * scale


def _mask_for(seed: bytes, shape: tuple[int, ...], modulus_range: int) -> np.ndarray:
    """One pseudorandom mask of the requested shape, from Flower's generator."""
    from flwr.common.secure_aggregation.secaggplus_utils import pseudo_rand_gen

    generated = pseudo_rand_gen(seed, modulus_range, [shape])
    return np.asarray(generated[0], dtype=np.int64)


def _role_seed(base: bytes, role: str) -> bytes:
    """Derive a role-specific seed from an already order-independent pair seed.

    The role must be mixed in *after* the pair seed is formed. Appending it to the
    label before sorting was a real bug: `sorted(("A|params", "B"))` and
    `sorted(("B|params", "A"))` order differently, so the two sides of a pair derived
    different seeds and the masks never cancelled.
    """
    return hashlib.sha256(base + b"\x00" + role.encode("utf-8")).digest()[:SEED_LENGTH]


def client_masks(
    shape: tuple[int, ...],
    client_label: str,
    all_labels: list[str],
    secret: bytes,
    modulus_range: int,
) -> tuple[np.ndarray, int]:
    """The net masks this client applies to its contribution AND to its weight.

    Returns ``(parameter_mask, weight_mask)``.

    Both are needed. Masking only the parameters does not work once the aggregate is
    weighted: the server sums ``weight_i * masked_i``, so each client's mask is
    multiplied by a *different* weight and no longer cancels. The protocol therefore
    masks ``weight_i * params_i`` (the contribution) and ``weight_i`` separately, and
    recovers the average as sum(contribution) / sum(weight). That is the construction
    SecAgg+ uses, and it is why the weight is a masked value rather than a plain one.
    """
    parameter_mask = np.zeros(shape, dtype=np.int64)
    weight_mask = 0
    for other in all_labels:
        if other == client_label:
            continue
        # One order-independent pair seed, then role-specific derivation from it.
        base_seed = pair_seed(secret, client_label, other)
        parameter_contribution = _mask_for(
            _role_seed(base_seed, "params"), shape, modulus_range
        )
        weight_contribution = int(
            _mask_for(_role_seed(base_seed, "weight"), (1,), modulus_range)[0]
        )

        if client_label < other:
            parameter_mask = (parameter_mask + parameter_contribution) % modulus_range
            weight_mask = (weight_mask + weight_contribution) % modulus_range
        else:
            parameter_mask = (parameter_mask - parameter_contribution) % modulus_range
            weight_mask = (weight_mask - weight_contribution) % modulus_range

    return parameter_mask, int(weight_mask)


def mask_update(
    contribution: np.ndarray,
    weight: int,
    client_label: str,
    all_labels: list[str],
    secret: bytes,
    modulus_range: int,
) -> tuple[np.ndarray, int]:
    """Mask one client's weighted contribution and its weight.

    Returns ``(masked_contribution, masked_weight)``, the pair the server sums.
    """
    parameter_mask, weight_mask = client_masks(
        contribution.shape, client_label, all_labels, secret, modulus_range
    )
    return (contribution + parameter_mask) % modulus_range, int(
        (int(weight) + weight_mask) % modulus_range
    )


def _quantized_weighted_sum(
    quantized: dict[str, np.ndarray],
    weights: dict[str, int],
    modulus_range: int,
) -> np.ndarray:
    """Sum of (weight * update), reduced into [0, modulus).

    Retained for the reference (unmasked) sum used to measure the mask residual.
    """
    total: np.ndarray | None = None
    for label, array in quantized.items():
        term = (array * int(weights[label])) % modulus_range
        total = term if total is None else (total + term) % modulus_range
    if total is None:
        raise SecureAggregationError(
            "No client updates were supplied for secure aggregation.",
            detail={"clients": []},
        )
    return total


def secure_aggregate(
    updates: dict[str, np.ndarray],
    weights: dict[str, int],
    secret: bytes,
    config: SecureAggregationConfig,
    expected_clients: list[str] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Mask every update, sum them, and recover the weighted average.

    Returns ``(average, report)``, matching `aggregate_parameter_dicts` so callers
    handle one return shape rather than two. ``average`` is a flat float array;
    ``report`` records what happened, including the quantization error measured
    against plain FedAvg on the same inputs.

    Raises:
        SecureAggregationError: when the participant set is incomplete. A missing
            client's masks do not cancel, so an aggregate computed without it is
            silently wrong. Refusing is the only honest outcome.
    """
    if not updates:
        raise SecureAggregationError(
            "No client updates were supplied for secure aggregation.",
            detail={"clients": []},
        )

    labels = sorted(updates)
    if expected_clients is not None:
        missing = sorted(set(expected_clients) - set(labels))
        if missing:
            # Dropout breaks the masking invariant rather than degrading it.
            raise SecureAggregationError(
                "Secure aggregation cannot complete: "
                f"{len(missing)} expected client(s) did not report ({', '.join(missing)}). "
                "Their masks do not cancel, so an aggregate computed without them "
                "would be silently incorrect.",
                detail={"missing": missing, "present": labels},
            )

    total_weight = sum(int(weights[label]) for label in labels)
    if total_weight <= 0:
        raise SecureAggregationError(
            "Total weight must be positive to compute a weighted average.",
            detail={"weights": weights},
        )

    # The recovered sum is unwrapped by treating values above modulus/2 as negative,
    # so the true sum must stay within +/- modulus/2. A clipped value is at most
    # quantization_range/2, so the weighted sum is bounded by total_weight * q/2.
    # Exceeding that would wrap silently and produce a plausible-looking wrong
    # aggregate, so refuse instead.
    safe_bound = config.modulus_range // 2
    theoretical_max = total_weight * (config.quantization_range / 2.0)
    if theoretical_max >= safe_bound:
        raise SecureAggregationError(
            "Secure aggregation would overflow the modulus: the weighted sum could "
            f"reach {theoretical_max:.0f}, which is not below {safe_bound}. Reduce the "
            "total example count, raise modulus_range, or lower quantization_range.",
            detail={
                "totalWeight": total_weight,
                "theoreticalMax": theoretical_max,
                "safeBound": safe_bound,
                "quantizationRange": config.quantization_range,
                "modulusRange": config.modulus_range,
            },
        )

    quantized = {
        label: quantize(flat(updates[label]), config.clipping_range, config.quantization_range)
        for label in labels
    }

    # The client masks its contribution and its weight together. Masking the raw
    # update instead would not survive weighting: the server sums weight * masked, so
    # each client's mask would be scaled differently and fail to cancel.
    contributions: dict[str, np.ndarray] = {}
    masked_weights: dict[str, int] = {}
    for label in labels:
        contribution = (quantized[label] * int(weights[label])) % config.modulus_range
        masked_contribution, masked_weight = mask_update(
            contribution,
            int(weights[label]),
            label,
            labels,
            secret,
            config.modulus_range,
        )
        contributions[label] = masked_contribution
        masked_weights[label] = masked_weight

    # The server sums only these two masked quantities.
    masked_contribution_sum = np.zeros_like(quantized[labels[0]])
    for label in labels:
        masked_contribution_sum = (
            masked_contribution_sum + contributions[label]
        ) % config.modulus_range

    total_masked_weight = 0
    for label in labels:
        total_masked_weight = (total_masked_weight + masked_weights[label]) % config.modulus_range

    # Reference: the same sum WITHOUT masks, to prove the masks cancelled exactly.
    reference_sum = _quantized_weighted_sum(quantized, weights, config.modulus_range)

    recovered_sum = _unwrap(masked_contribution_sum, config.modulus_range)
    reference_unwrapped = _unwrap(reference_sum, config.modulus_range)
    recovered_weight = float(
        _unwrap(np.array([total_masked_weight]), config.modulus_range)[0]
    )

    if recovered_weight <= 0:
        raise SecureAggregationError(
            "The recovered total weight is not positive, so the aggregate cannot be "
            "normalised. The masking protocol did not cancel correctly.",
            detail={"recoveredWeight": recovered_weight, "expectedWeight": total_weight},
        )

    average = dequantize(
        recovered_sum / recovered_weight, config.clipping_range, config.quantization_range
    )
    plain = dequantize(
        reference_unwrapped / recovered_weight, config.clipping_range, config.quantization_range
    )

    # Quantization error measured against the *unquantized* client updates, which is
    # the honest "what did this cost us" number.
    unquantized_plain = sum(
        flat(updates[label]) * int(weights[label]) for label in labels
    ) / total_weight

    report = {
        "protocol": "SecAgg+ style pairwise masking",
        "enabled": True,
        "primitive": "secaggplus_utils.pseudo_rand_gen",
        "clients": labels,
        "weights": {label: int(weights[label]) for label in labels},
        "totalWeight": total_weight,
        "recoveredWeight": recovered_weight,
        "modulusRange": config.modulus_range,
        "quantizationRange": config.quantization_range,
        "clippingRange": config.clipping_range,
        # Zero means every mask cancelled exactly.
        "maskResidualMaxAbs": float(np.max(np.abs(recovered_sum - reference_unwrapped))),
        "weightResidual": float(abs(recovered_weight - total_weight)),
        "quantizationErrorMaxAbs": float(np.max(np.abs(average - unquantized_plain))),
        "quantizationErrorMeanAbs": float(np.mean(np.abs(average - unquantized_plain))),
        "individualUpdatesHidden": bool(
            all(
                not np.array_equal(contributions[label], (quantized[label] * int(weights[label])) % config.modulus_range)
                for label in labels
            )
        ),
        "assumptions": [
            "All participating clients reported; dropout would break mask cancellation.",
            "Pairwise seeds were shared over a channel assumed to be secure.",
            "The true weighted sum stays below the modulus.",
            "Protects a client's whole update from the server, not individual records.",
            "No secret sharing: a dropout cannot be recovered from.",
            "Not protection against collusion.",
        ],
    }
    return average, report


def _unwrap(values: np.ndarray, modulus_range: int) -> np.ndarray:
    """Interpret a modular sum as a signed integer.

    A summed value above half the modulus is read as its negative representative,
    which is how a true sum that went slightly negative is recovered.
    """
    half = modulus_range // 2
    signed = np.where(values >= half, values - modulus_range, values)
    return signed.astype(np.float64)


def flat(array: np.ndarray) -> np.ndarray:
    """One-dimensional, so masking works over a single vector per client."""
    return np.asarray(array, dtype=np.float64).reshape(-1)


def aggregate_parameter_dicts(
    updates: dict[str, dict[str, np.ndarray]],
    weights: dict[str, int],
    secret: bytes,
    config: SecureAggregationConfig,
    expected_clients: list[str] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Secure aggregation over named model parameters.

    Parameters are concatenated into one vector per client so a single masking pass
    covers the whole model, which is what makes the protocol's cost independent of
    the number of tensors.
    """
    labels = sorted(updates)
    if not labels:
        raise SecureAggregationError(
            "No client updates were supplied for secure aggregation.", detail={"clients": []}
        )

    reference_names = sorted(updates[labels[0]])
    for label in labels:
        if sorted(updates[label]) != reference_names:
            raise SecureAggregationError(
                f"Client {label} supplied parameters that do not match the other clients.",
                detail={"client": label, "expected": reference_names},
            )

    flattened: dict[str, np.ndarray] = {}
    shapes: dict[str, tuple[int, ...]] = {}
    for label in labels:
        pieces = [flat(updates[label][name]) for name in reference_names]
        flattened[label] = np.concatenate(pieces)
        shapes[label] = tuple(np.asarray(updates[label][reference_names[0]]).shape)

    result, report = secure_aggregate(flattened, weights, secret, config, expected_clients)

    # Rebuild the named parameter dict from the recovered vector.
    vector = np.asarray(result, dtype=np.float64)
    restored: dict[str, np.ndarray] = {}
    offset = 0
    for name in reference_names:
        shape = tuple(np.asarray(updates[labels[0]][name]).shape)
        size = int(np.prod(shape))
        restored[name] = vector[offset : offset + size].reshape(shape)
        offset += size

    report["parameters"] = reference_names
    report["parameterCount"] = int(vector.size)
    return restored, report
