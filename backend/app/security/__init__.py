"""Privacy and security mechanisms for FedMed.

    auth.py          API-key authentication and scope-based authorization
    audit.py         a real, bounded, redacting audit log
    tls.py           TLS certificate generation, inspection and wiring
    redaction.py     one place that strips secrets before anything is logged
    dependencies.py  FastAPI authorization dependencies
    keys.py          CLI: generate API keys and development certificates

`app/federated/privacy.py` holds differential privacy, and
`app/federated/secure_aggregation.py` holds secure aggregation, because both are
properties of the federated aggregation path rather than of the API.

Nothing here makes the application production-secure. See docs/SECURITY.md for the
assumptions, the limitations, and the independent validation that would be required.
"""

from __future__ import annotations

from .auth import (
    API_KEY_HEADER,
    ApiKey,
    Principal,
    Scope,
    authenticate,
    generate_key,
    hash_key,
    parse_api_keys,
    public_key_summary,
)
from .audit import AuditLog, AuditRecord, audit_log
from .dependencies import (
    require_admin,
    require_read,
    require_scope,
    require_scope_or_public_demo,
    require_train,
)
from .redaction import PLACEHOLDER, redact, redact_text
from .tls import (
    CertificateInfo,
    generate_self_signed_certificate,
    inspect_certificate,
    uvicorn_ssl_kwargs,
)

__all__ = [
    # auth
    "API_KEY_HEADER",
    "ApiKey",
    "Principal",
    "Scope",
    "authenticate",
    "generate_key",
    "hash_key",
    "parse_api_keys",
    "public_key_summary",
    # audit
    "AuditLog",
    "AuditRecord",
    "audit_log",
    # authorization dependencies
    "require_scope",
    "require_scope_or_public_demo",
    "require_read",
    "require_train",
    "require_admin",
    # redaction
    "PLACEHOLDER",
    "redact",
    "redact_text",
    # tls
    "CertificateInfo",
    "generate_self_signed_certificate",
    "inspect_certificate",
    "uvicorn_ssl_kwargs",
]
