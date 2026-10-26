"""Application configuration.

All settings come from environment variables (optionally via a .env file in the
repository root) and every one of them has a safe default, so the API starts
without any local configuration at all.

Environment variable names are prefixed with FEDMED_, e.g.
    FEDMED_ENVIRONMENT=production
    FEDMED_CORS_ORIGINS=https://dash.example.org
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

# Resolve the .env location from this file rather than the current directory, so
# the settings load the same way whether uvicorn is started from the repository
# root or from backend/. backend/.env wins over the repository-root .env.
_BACKEND_DIR = Path(__file__).resolve().parents[2]
_REPO_ROOT = Path(__file__).resolve().parents[3]
_ENV_FILES = (str(_BACKEND_DIR / ".env"), str(_REPO_ROOT / ".env"))


class Settings(BaseSettings):
    """Runtime settings for the FedMed API."""

    model_config = SettingsConfigDict(
        env_prefix="FEDMED_",
        env_file=_ENV_FILES,
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Application ---------------------------------------------------------
    app_name: str = "FedMed API"
    version: str = "0.2.0"
    api_prefix: str = "/api/v1"
    description: str = (
        "Backend for the FedMed cross-silo federated learning dashboard. "
        "All data currently returned is synthetic and clearly labelled as such; "
        "no model is trained and no privacy mechanism is implemented."
    )

    # --- Server --------------------------------------------------------------
    environment: str = "development"
    host: str = "127.0.0.1"
    port: int = 8000
    log_level: str = "info"
    reload: bool = False

    # --- CORS ----------------------------------------------------------------
    # Explicit origins only. "*" is never the default and is refused in
    # production unless allow_all_origins_in_production is set on purpose.
    #
    # NoDecode is required: without it pydantic-settings tries json.loads() on
    # this list field before the validator below runs, so a perfectly ordinary
    # "http://a,http://b" value in .env would raise a JSONDecodeError at startup.
    # The validator below does the splitting instead.
    cors_origins: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: [
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:4173",
            "http://127.0.0.1:4173",
        ]
    )
    allow_all_origins: bool = False
    allow_all_origins_in_production: bool = False

    # --- Behaviour -----------------------------------------------------------
    # The training endpoints drive a simulated job only. This flag exists so the
    # simulated nature of the job is configuration, not something buried in code.
    simulated_training: bool = True
    # Seconds of wall-clock time that one simulated federated round takes.
    # Must be positive: zero would mean "every round is already finished", which
    # makes the running state unreachable rather than fast.
    simulated_round_seconds: float = Field(default=4.0, gt=0, le=3600)
    # Rounds that exist in the seeded baseline history before any job runs.
    baseline_rounds: int = 8

    # --- Security ------------------------------------------------------------
    # Comma-separated "label:role:sha256hex" entries. Store the HASH of a key, never
    # the key. Empty means authentication is UNCONFIGURED, in which case the
    # privileged endpoints refuse with 503 rather than allowing anonymous access —
    # a missing configuration must not silently mean "no authentication".
    api_keys: str = ""
    # Whether GET endpoints also require a key. Off by default so the read-only
    # dashboard stays browsable; the mutating and admin endpoints are always
    # protected. Turn this on to lock reads down too.
    auth_required_for_reads: bool = False

    #: Open the medical-report routes to anonymous callers. OFF by default, and it
    #: should stay off for anything handling real reports.
    #:
    #: It exists because a *demonstration* deployment cannot authenticate an anonymous
    #: visitor: every VITE_ value is compiled into the JavaScript the browser downloads,
    #: so an API key cannot be shipped to the page. Without this, the feature is simply
    #: unusable on a public demo.
    #:
    #: It is narrow on purpose -- it covers the medical-report routes only, every other
    #: protected route keeps its key requirement -- and it is conspicuous: the state is
    #: reported by GET /api/v1/medical-reports/health so the page can warn, logged at
    #: startup, and every anonymous access is written to the audit log.
    medical_reports_public_demo: bool = False

    #: Serve the built frontend from the API itself, so one origin hosts both the
    #: pages and the API. Off by default; a normal deployment keeps them on separate
    #: hosts (Vercel for the pages, Render for the API).
    #:
    #: It exists for the single-origin case, where the API is reached through a tunnel
    #: or a reverse proxy that exposes exactly one port. With the pages on the same
    #: origin as the API the browser makes same-origin requests, so there is no CORS
    #: configuration to get wrong.
    serve_frontend: bool = False

    #: Where the built frontend lives. Defaults to frontend/dist next to this package.
    #: Override it when the build is somewhere else.
    frontend_dir: str | None = None
    # TLS. When both paths are set the API can be served over HTTPS; see
    # app/security/tls.py and docs/SECURITY.md.
    tls_cert_file: str | None = None
    tls_key_file: str | None = None
    # Shared secret used to derive secure-aggregation pairwise seeds. Must come from
    # the environment. Empty means secure aggregation refuses to run, because seeds
    # derived from a default constant would be a false sense of protection.
    secure_aggregation_secret: str | None = None

    # --- Optional AI provider (medical report analysis) ----------------------
    # Any OpenAI-compatible chat-completions endpoint. EMPTY MEANS THE AI LAYER IS
    # UNAVAILABLE, which the API reports plainly: the generated sections
    # (possible complications, clinical suggestions, narrative summary) are then shown
    # as unavailable rather than filled with a plausible-looking template. The
    # deterministic extraction and comparison keep working either way.
    #
    # The key must never reach the frontend: it lives here, server-side only.
    ai_base_url: str | None = None
    ai_api_key: str | None = None
    ai_model: str | None = None
    # Long enough for a slow provider, short enough that a hung request does not pin
    # a worker indefinitely.
    ai_timeout_seconds: float = Field(default=60.0, gt=0, le=600)

    @property
    def auth_configured(self) -> bool:
        """Whether any API key is configured. False means auth is unconfigured."""
        return bool(self.api_keys.strip())

    @property
    def tls_configured(self) -> bool:
        """Whether certificate and key paths are both set."""
        return bool(self.tls_cert_file and self.tls_key_file)

    def parsed_api_keys(self):
        """Configured keys, or an empty tuple when authentication is unconfigured.

        Parsing raises ValueError on a malformed entry so a configuration typo is a
        visible failure rather than a silently disabled control.
        """
        # Imported here rather than at module scope: app.security imports nothing
        # from app.core, and keeping that direction avoids an import cycle.
        from ..security.auth import parse_api_keys

        return parse_api_keys(self.api_keys) if self.auth_configured else ()

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, value: object) -> object:
        """Accept a comma-separated string as well as a JSON list.

        This makes FEDMED_CORS_ORIGINS=http://a,http://b work in a .env file,
        which is far friendlier than requiring JSON syntax. Because the field is
        annotated NoDecode, the raw string arrives here and this method is
        responsible for producing the list in both forms.
        """
        if not isinstance(value, str):
            return value

        text = value.strip()
        if not text:
            return []

        if text.startswith("["):
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"FEDMED_CORS_ORIGINS looks like JSON but could not be parsed: {exc}. "
                    "Use a comma-separated list instead, e.g. "
                    "http://localhost:5173,http://127.0.0.1:5173"
                ) from exc
            if not isinstance(parsed, list):
                raise ValueError("FEDMED_CORS_ORIGINS must be a JSON array or a comma-separated list")
            return [str(item).strip() for item in parsed if str(item).strip()]

        return [item.strip() for item in text.split(",") if item.strip()]

    @property
    def is_production(self) -> bool:
        return self.environment.strip().lower() in {"production", "prod"}

    def cors_origin_list(self) -> list[str]:
        """Origins the API will accept, with production safety applied."""
        if not self.allow_all_origins:
            return self.cors_origins
        if self.is_production and not self.allow_all_origins_in_production:
            # Refuse the wildcard in production rather than silently allowing it.
            return self.cors_origins
        return ["*"]

    def effective_log_level(self) -> str:
        level = self.log_level.strip().lower()
        return level if level in {"critical", "error", "warning", "info", "debug"} else "info"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings instance, so the .env file is read only once."""
    return Settings()
