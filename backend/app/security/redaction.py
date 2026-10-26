"""Redaction of secrets before anything is logged, returned or audited.

One function, used everywhere, because a second implementation is a second place to
get it wrong. It is a belt-and-braces control: the primary defence is that secrets
are never passed to these functions in the first place.

Patterns covered: API keys in headers or URLs, bearer tokens, AWS-style keys,
private key blocks, and long high-entropy strings appearing next to a
secret-suggesting key name.
"""

from __future__ import annotations

import re
from typing import Any

PLACEHOLDER = "[REDACTED]"

#: Key names whose values are always redacted, matched case-insensitively.
SENSITIVE_KEY_NAMES = frozenset(
    {
        "api_key",
        "apikey",
        "api-key",
        "x-api-key",
        "authorization",
        "auth",
        "password",
        "passwd",
        "secret",
        "token",
        "access_token",
        "refresh_token",
        "private_key",
        "privatekey",
        "client_secret",
        "session",
        "cookie",
        "credential",
        "credentials",
    }
)

_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    # "X-API-Key: abc123" or "api_key=abc123" or "authorization: Bearer abc"
    (
        re.compile(
            r"(?i)\b(api[_-]?key|x[_-]api[_-]?key|authorization|bearer|token|password|"
            r"secret|client[_-]secret|private[_-]key)\b\s*[:=]\s*(?:bearer\s+)?([A-Za-z0-9._\-+/=]{6,})"
        ),
        rf"\1: {PLACEHOLDER}",
    ),
    # PEM private key blocks, however they were truncated.
    (
        re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----"),
        PLACEHOLDER,
    ),
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"), PLACEHOLDER),
    # Long base64url/hex runs that look like generated keys, 40+ characters.
    (re.compile(r"\b[A-Za-z0-9_\-]{40,}\b"), PLACEHOLDER),
    # SHA-256 digests, which are what we store instead of keys.
    (re.compile(r"\b[0-9a-fA-F]{64}\b"), PLACEHOLDER),
)


def redact_text(value: str) -> str:
    """Return `value` with anything secret-looking replaced."""
    if not value:
        return value
    result = value
    for pattern, replacement in _PATTERNS:
        result = pattern.sub(replacement, result)
    return result


def redact(value: Any) -> Any:
    """Recursively redact a value: dicts by key name, strings by pattern.

    Non-string scalars pass through unchanged, so log records stay readable.
    """
    if isinstance(value, dict):
        return {
            key: (
                PLACEHOLDER
                if str(key).strip().lower() in SENSITIVE_KEY_NAMES
                else redact(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        redacted = [redact(item) for item in value]
        return type(value)(redacted) if isinstance(value, tuple) else redacted
    if isinstance(value, str):
        return redact_text(value)
    return value


def is_likely_secret(value: str) -> bool:
    """Whether a string looks like it would be redacted. Used by the leak tests."""
    return redact_text(value) != value
