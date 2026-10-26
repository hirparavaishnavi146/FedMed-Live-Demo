"""API-key authentication and scope-based authorization.

HOW IT WORKS

    A caller sends its key in the ``X-API-Key`` header. The server hashes the
    presented key with SHA-256 and compares it, in constant time, against the
    hashes in its configuration. The raw key is never stored, never logged and
    never returned.

    Configuration is a comma-separated list of ``label:role:sha256hex`` entries in
    the ``FEDMED_API_KEYS`` environment variable (or ``api_keys`` in settings):

        FEDMED_API_KEYS=ops:admin:<sha256>,reader:read:<sha256>

    Operators generate a key and store only its hash:

        python -m app.security.keys generate --label ops --role admin

    Storing the hash rather than the key means a leaked environment file or a copy
    of the repository does not hand over a working credential.

SCOPES

    read    GET endpoints.
    train   Starting, stopping and resetting a federated run.
    admin   Key and security configuration.

    ``admin`` implies ``train`` implies ``read``: a role grants its own scope and
    everything below it.

FAIL-CLOSED

    When no keys are configured, authentication is **unconfigured**, and the
    privileged endpoints refuse with 503 rather than allowing anonymous access. A
    missing configuration must not silently mean "no authentication": that is the
    single most common way a security control ends up doing nothing while looking
    present.

WHAT THIS IS NOT

    It is not user management, not a session or token system, and not multi-tenancy.
    A static key has no expiry, no revocation list and no per-user attribution
    beyond its label. It stops an unauthenticated caller from changing training
    configuration; it is not an authorization system for a hospital network. See
    docs/SECURITY.md.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
from dataclasses import dataclass
from enum import Enum

API_KEY_HEADER = "X-API-Key"

#: Characters a generated key is drawn from and its length, so the documented
#: strength is checkable rather than implied. 32 bytes of base64url is 256 bits.
KEY_BYTES = 32


class Scope(str, Enum):
    """What a caller is allowed to do."""

    read = "read"
    train = "train"
    admin = "admin"


#: A role grants its own scope and every scope below it.
_ROLE_RANK: dict[Scope, int] = {Scope.read: 1, Scope.train: 2, Scope.admin: 3}


@dataclass(frozen=True)
class ApiKey:
    """One configured key. Holds only the hash of the secret."""

    label: str
    role: Scope
    key_hash: str

    def __post_init__(self) -> None:
        if not self.label:
            raise ValueError("An API key entry needs a label so it can be attributed.")
        digest = self.key_hash.strip().lower()
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError(
                f"API key '{self.label}' has an invalid hash. Expected the 64-character "
                "hex SHA-256 of the key; store the hash, never the key itself."
            )

    def matches(self, presented_hash: str) -> bool:
        """Constant-time hash comparison, so timing cannot identify a prefix."""
        return hmac.compare_digest(self.key_hash, presented_hash)

    def __repr__(self) -> str:  # pragma: no cover - defensive
        # Never let a dataclass repr print hash material into a log.
        return f"ApiKey(label={self.label!r}, role={self.role.value!r})"


@dataclass(frozen=True)
class Principal:
    """The authenticated caller."""

    label: str
    role: Scope

    def has_scope(self, required: Scope) -> bool:
        return _ROLE_RANK[self.role] >= _ROLE_RANK[required]


def hash_key(raw_key: str) -> str:
    """SHA-256 hex of a key, which is all that is ever stored."""
    if not isinstance(raw_key, str) or not raw_key:
        raise ValueError("API key must be a non-empty string.")
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def generate_key(length: int = KEY_BYTES) -> str:
    """A cryptographically random key, URL-safe and 256 bits by default."""
    return secrets.token_urlsafe(length)


def parse_api_keys(specification: str) -> tuple[ApiKey, ...]:
    """Parse ``label:role:hash[,label:role:hash...]`` into keys.

    Raises ValueError on anything malformed, naming the offending entry, so a typo
    in configuration is a startup failure rather than a silently disabled control.
    """
    entries = [
        entry.strip()
        for entry in (specification or "").replace("\n", ",").split(",")
        if entry.strip()
    ]

    keys: list[ApiKey] = []
    seen_labels: set[str] = set()
    for entry in entries:
        parts = [part.strip() for part in entry.split(":")]
        if len(parts) < 3:
            raise ValueError(
                f"Malformed API key entry {entry!r}. Expected 'label:role:sha256hex'."
            )
        # Label first, role second, hash last: the hash is a single token, so a
        # label containing a colon is still parsed predictably.
        label = parts[0]
        role_text = parts[1].lower()
        key_hash = parts[-1]
        if len(parts) > 3:
            label = ":".join(parts[:-2])

        try:
            role = Scope(role_text)
        except ValueError as exc:
            valid = ", ".join(scope.value for scope in Scope)
            raise ValueError(
                f"API key '{label}' has unknown role {role_text!r}. Valid roles: {valid}."
            ) from exc

        if label in seen_labels:
            raise ValueError(f"Duplicate API key label {label!r}; labels must be unique.")
        seen_labels.add(label)

        keys.append(ApiKey(label=label, role=role, key_hash=key_hash))

    return tuple(keys)


def authenticate(
    presented_key: str | None,
    configured: tuple[ApiKey, ...],
) -> Principal | None:
    """Match a presented key against the configuration.

    Returns None when the key is absent or matches nothing. Checks every entry
    rather than stopping at the first mismatch so the comparison does not reveal
    how far a guess got.
    """
    if not presented_key or not configured:
        return None

    presented_hash = hash_key(presented_key)
    matched: Principal | None = None
    for key in configured:
        if key.matches(presented_hash):
            matched = Principal(label=key.label, role=key.role)
    return matched


def public_key_summary(keys: tuple[ApiKey, ...]) -> list[dict[str, str]]:
    """Safe description of the configured keys, for the security page.

    Returns labels and roles only. No key, no hash, no prefix of a hash: a partial
    hash is still material an attacker can test guesses against.
    """
    return [
        {"label": key.label, "role": key.role.value}
        for key in sorted(keys, key=lambda item: item.label)
    ]
