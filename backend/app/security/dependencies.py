"""Authorization dependencies for FastAPI.

`require_scope(scope)` returns a dependency that authenticates the caller and checks
the scope, refusing with a specific status and machine-readable code:

    401 invalid_api_key          no key, or one that matches nothing
    403 insufficient_scope       authenticated, but the role is too low
    503 auth_not_configured      no keys configured: fail closed, not open

Every outcome is written to the audit log with the caller identified by label only.
The presented key is never logged, and never appears in the response.
"""

from __future__ import annotations

from typing import Annotated, Callable

from fastapi import Header, Request

from ..errors import ApiError
from .auth import API_KEY_HEADER, Principal, Scope, authenticate


def require_scope(scope: Scope) -> Callable[..., Principal]:
    """Build a dependency that requires the given scope."""

    def dependency(
        request: Request,
        x_api_key: Annotated[str | None, Header(alias=API_KEY_HEADER)] = None,
    ) -> Principal:
        settings = request.app.state.settings
        audit = getattr(request.app.state, "audit_log", None)
        path = request.url.path

        # Imported here so this module does not import app.core at module scope.
        from ..core.config import Settings

        assert isinstance(settings, Settings)

        if not settings.auth_configured:
            if audit is not None:
                audit.record_auth(
                    "refused", None, path, reason="authentication is not configured"
                )
            raise ApiError(
                "This endpoint requires authentication, but no API keys are "
                "configured on the server. Configure FEDMED_API_KEYS to enable it; "
                "access is refused rather than allowed anonymously.",
                code="auth_not_configured",
                status_code=503,
                detail={"howToConfigure": "python -m app.security.keys generate --label ops --role admin"},
            )

        principal = authenticate(x_api_key, settings.parsed_api_keys())
        if principal is None:
            if audit is not None:
                audit.record_auth(
                    "denied",
                    None,
                    path,
                    reason="missing or invalid key" if x_api_key else "no key supplied",
                )
            raise ApiError(
                "A valid API key is required. Send it in the "
                f"{API_KEY_HEADER} header.",
                code="invalid_api_key",
                status_code=401,
                detail={"header": API_KEY_HEADER},
            )

        if not principal.has_scope(scope):
            if audit is not None:
                audit.record_auth(
                    "forbidden",
                    principal.label,
                    path,
                    reason=f"role '{principal.role.value}' lacks scope '{scope.value}'",
                )
            raise ApiError(
                f"This action requires the '{scope.value}' scope, but the key "
                f"'{principal.label}' has role '{principal.role.value}'.",
                code="insufficient_scope",
                status_code=403,
                detail={
                    "required": scope.value,
                    "actual": principal.role.value,
                    "principal": principal.label,
                },
            )

        if audit is not None:
            audit.record_auth("granted", principal.label, path)
        return principal

    return dependency


#: Convenience dependencies for the scopes this API uses.
require_read = require_scope(Scope.read)
require_train = require_scope(Scope.train)
require_admin = require_scope(Scope.admin)


def require_scope_or_public_demo(scope: Scope) -> Callable[..., Principal | None]:
    """Like :func:`require_scope`, but the caller may opt out of authentication.

    The medical-report routes are the only caller, and only because a *demonstration*
    deployment has no way to authenticate an anonymous visitor: a Vite build compiles
    every ``VITE_`` value into the JavaScript the browser downloads, so an API key
    cannot be shipped to the page. Without an escape hatch the feature is unusable on
    a public demo.

    The escape hatch is deliberately narrow:

    * off by default (``FEDMED_MEDICAL_REPORTS_PUBLIC_DEMO=false``);
    * it covers the medical-report routes only -- every other protected route keeps
      its requirement, which a test asserts;
    * the state is reported by ``GET /api/v1/medical-reports/health`` so the page can
      warn, and logged at startup so an operator sees it;
    * every anonymous access is written to the audit log.

    With the flag on, a *valid* key is still honoured, so an operator who presents one
    is identified by label. A key that is present but unusable does not become a hard
    failure -- the route is public, so refusing would only confuse a client that still
    carries a stale key -- but the attempt is recorded in the audit log, so the
    misconfiguration is visible rather than swallowed.
    """

    def dependency(
        request: Request,
        x_api_key: Annotated[str | None, Header(alias=API_KEY_HEADER)] = None,
    ) -> Principal | None:
        settings = request.app.state.settings
        audit = getattr(request.app.state, "audit_log", None)
        path = request.url.path

        from ..core.config import Settings

        assert isinstance(settings, Settings)

        if settings.medical_reports_public_demo:
            if x_api_key:
                principal = authenticate(x_api_key, settings.parsed_api_keys())
                if principal is None:
                    # The route is declared public, so an unusable key must not become
                    # a hard failure: a local .env.local that still carries a key would
                    # otherwise 401 against a demo server that has no keys configured.
                    # The attempt is recorded instead, so the misconfiguration is
                    # visible in the audit log rather than silently swallowed.
                    if audit is not None:
                        audit.record_auth(
                            "denied",
                            None,
                            path,
                            reason=(
                                "a key was supplied but matched nothing; continuing "
                                "anonymously because public demo mode is on"
                            ),
                        )
                    return None
                if audit is not None:
                    audit.record_auth("granted", principal.label, path)
                return principal

            if audit is not None:
                audit.record_auth(
                    "granted",
                    None,
                    path,
                    reason="anonymous access: medical reports public demo mode is enabled",
                )
            return None

        # Not in public demo mode: behave exactly as require_scope does, rather than
        # reimplementing the checks and letting the two drift apart.
        return require_scope(scope)(request, x_api_key)

    return dependency
