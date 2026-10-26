"""FastAPI application entry point.

Run it from the backend directory:

    ..\\.venv\\Scripts\\python.exe -m uvicorn app.main:app --reload

Interactive documentation is then at http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.datastructures import MutableHeaders

from .api.v1 import api_router
from .api.v1.health import router as health_router
from .core.config import Settings, get_settings
from .handlers import register_exception_handlers
from .schemas.common import SIMULATED_NOTICE
from .federated.simulation import local_runner as federated_runner
from .security import API_KEY_HEADER, AuditLog


class FrontendCachePolicy:
    """Cache policy for the served frontend.

    Vite writes a content hash into every asset filename, so an asset URL can never
    come to mean something different and may be cached indefinitely. The entry document
    must be the opposite: it is the file that NAMES the current hashes.

    Without this split the document gets cached, and after a rebuild it still points at
    an asset filename the build has deleted. The browser then requests a bundle that
    404s and the page renders blank -- a stale-HTML failure that looks like a broken
    application and is only cleared by a manual hard refresh.
    """

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")

        async def send_with_cache(message: Any) -> None:
            if message.get("type") == "http.response.start":
                headers = MutableHeaders(scope=message)
                if path.startswith("/assets/"):
                    headers["cache-control"] = "public, max-age=31536000, immutable"
                elif headers.get("content-type", "").startswith("text/html"):
                    headers["cache-control"] = "no-cache"
            await send(message)

        await self.app(scope, receive, send_with_cache)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the application. A factory so tests can pass their own settings."""
    config = settings or get_settings()

    logging.basicConfig(
        level=config.effective_log_level().upper(),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        logger = logging.getLogger("fedmed.startup")
        logger.info("%s %s starting", config.app_name, config.version)
        logger.info("environment=%s simulated_training=%s", config.environment, config.simulated_training)
        logger.info("CORS origins: %s", ", ".join(config.cors_origin_list()))
        if config.medical_reports_public_demo:
            logger.warning(
                "FEDMED_MEDICAL_REPORTS_PUBLIC_DEMO is ON: the medical-report routes "
                "accept anonymous uploads and analysis requests. This is a "
                "demonstration setting. Do not serve real patient data from this "
                "deployment."
            )
        if not config.auth_configured:
            logger.warning(
                "No API keys are configured: privileged endpoints will refuse with "
                "503 auth_not_configured rather than allowing anonymous access."
            )
        # A fresh process must not inherit a run from a previous one. Runs live in
        # memory only, so the process always starts clean.
        #
        # force_reset rather than reset: reset() raises while a run is in progress,
        # which would crash startup (or shutdown below) whenever the previous process
        # was killed with a run still going. Stopping the worker and joining it is the
        # only safe way to clear.
        federated_runner.force_reset()
        yield
        # Same reasoning on the way out: a shutdown during a run must stop the worker
        # and wait for it rather than raising.
        federated_runner.force_reset()
        logger.info("%s stopping", config.app_name)

    app = FastAPI(
        title=config.app_name,
        version=config.version,
        description=config.description + "\n\n" + SIMULATED_NOTICE,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    origins = config.cors_origin_list()
    # Added before CORS so the cache policy runs closest to the response it describes.
    app.add_middleware(FrontendCachePolicy)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        # Credentials stay off: there is no authentication yet, and the wildcard
        # combined with credentials is the configuration browsers reject.
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        # The API key header must be listed or a browser preflight will refuse it,
        # which would make every authenticated request fail in the UI rather than
        # at the authorization check the caller can read.
        #
        # ngrok-skip-browser-warning is here for the same reason: the client sends it
        # on every request to suppress ngrok's free-tier HTML interstitial, and a
        # custom header makes the request non-simple, so the preflight has to approve
        # it explicitly or the browser blocks the call before it is sent.
        allow_headers=["Accept", "Content-Type", API_KEY_HEADER, "ngrok-skip-browser-warning"],
        max_age=600,
    )

    register_exception_handlers(app)

    # Single source of truth for the running app's configuration; dependencies
    # read this instead of re-instantiating Settings.
    app.state.settings = config
    # Real audit log: recorded events from this process, not seeded rows.
    app.state.audit_log = AuditLog()

    app.include_router(health_router)
    app.include_router(api_router, prefix=config.api_prefix)

    dist = _frontend_dist(config) if config.serve_frontend else None

    if dist is not None:
        # Pages and API on one origin. Everything below is registered LAST, so the
        # API routes above always win and only unmatched paths reach the frontend.
        app.mount(
            "/assets",
            StaticFiles(directory=dist / "assets"),
            name="frontend-assets",
        )

        @app.get("/{full_path:path}", include_in_schema=False)
        async def frontend(full_path: str) -> FileResponse:
            """Serve the built single-page app, including client-side routes.

            A path the API owns but that does not exist must still be an API 404, not
            index.html: returning a page there would turn a mistyped endpoint into a
            confusing HTML response.
            """
            if full_path.startswith(("api/", "health", "docs", "redoc", "openapi.json")):
                raise HTTPException(status_code=404, detail="Not Found")

            candidate = dist / full_path
            if full_path and candidate.is_file():
                return FileResponse(candidate)
            # Any other path is a client-side route (the router handles it) or a
            # hard refresh on one, so the app's entry document is the right answer.
            return FileResponse(dist / "index.html")

        # A module-scope logger: `logger` inside lifespan is a local, not visible here.
        logging.getLogger("fedmed.startup").info("serving the built frontend from %s", dist)
    else:

        @app.get("/", include_in_schema=False)
        def root() -> dict[str, object]:
            """Small landing response so opening the host in a browser is not a 404."""
            return {
                "service": config.app_name,
                "version": config.version,
                "docs": "/docs",
                "health": "/health",
                "api": config.api_prefix,
                "notice": SIMULATED_NOTICE,
            }

    return app


def _frontend_dist(config: Settings) -> Path | None:
    """Locate the built frontend, or None when it is not there.

    When FEDMED_FRONTEND_DIR is set it wins outright -- an explicit path should not be
    silently overridden. Otherwise two layouts are tried:

    * ``<repo>/frontend/dist`` beside this package, which is how the checkout is laid
      out and is derived from this file rather than the working directory so the server
      can be started from either the repository root or ``backend/``;
    * ``<cwd>/frontend/dist``, which is how a deployment that uploads only what it
      needs is laid out -- there is no repository around the package, so the path
      relative to this file is meaningless.
    """
    logger = logging.getLogger("fedmed.startup")

    if config.frontend_dir:
        candidates = [Path(config.frontend_dir).expanduser()]
    else:
        candidates = [
            Path(__file__).resolve().parents[2] / "frontend" / "dist",
            Path.cwd() / "frontend" / "dist",
        ]

    for candidate in candidates:
        candidate = candidate.resolve()
        if (candidate / "index.html").is_file() and (candidate / "assets").is_dir():
            return candidate

    logger.warning(
        "FEDMED_SERVE_FRONTEND is on but no build was found at any of %s "
        "(run: npm run build in frontend/). Skipping the static mount.",
        ", ".join(str(c.resolve()) for c in candidates),
    )
    return None


app = create_app()
