# FedMed public demo — one container, one origin.
#
# Why Docker and not Render's native Python runtime: the medical-report workflow has two
# capabilities that are SYSTEM packages, which pip cannot install and a native runtime
# cannot apt-get:
#
#   tesseract-ocr   the OCR engine itself. Without it a scanned report — or a photograph
#                   of one, which is what most people actually have — is reported as
#                   unreadable instead of being read. pytesseract is only a wrapper.
#   poppler-utils   gives pdf2image the PDF rasteriser, so a scanned PDF can be rendered
#                   to images for Tesseract.
#
# The frontend is NOT built here: frontend/dist is committed, so no Node stage and no
# `npm ci` — that build is what exhausted a 512 MB instance in earlier attempts.
#
# Deliberately NOT installed: torch, MONAI, scipy, nibabel, opacus, Flower. torch alone is
# 511 MB against a 512 MB instance. Without them the training endpoints answer
# 503 training_unavailable — an honest refusal, not a failure that looks like a bug.

FROM python:3.14-slim

# OCR engine + PDF rasteriser (Tesseract reads images; poppler renders PDF pages to them).
RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies in their own layer, so editing application code does not reinstall them.
COPY backend/requirements-web.txt ./backend/requirements-web.txt
RUN pip install --no-cache-dir -r backend/requirements-web.txt \
    && pip install --no-cache-dir pytesseract==0.3.13 pdf2image==1.17.0 Pillow==12.3.0

# Application code and the pre-built pages. tests/ is absent on purpose: its fixtures
# embed key material and a deployed container has no use for them.
COPY backend/app ./backend/app
COPY frontend/dist ./frontend/dist

ENV FEDMED_ENVIRONMENT=production \
    FEDMED_LOG_LEVEL=info \
    FEDMED_SERVE_FRONTEND=true \
    FEDMED_FRONTEND_DIR=/app/frontend/dist \
    FEDMED_AUTH_REQUIRED_FOR_READS=false \
    FEDMED_MEDICAL_REPORTS_PUBLIC_DEMO=true \
    FEDMED_STAGING_DIR=/tmp/fedmed-staging \
    FEDMED_AI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai \
    FEDMED_AI_MODEL=gemini-3.5-flash-lite \
    FEDMED_AI_TIMEOUT_SECONDS=90

# $PORT is supplied by Render; 0.0.0.0 is required because the app's own default host is
# 127.0.0.1, which is reachable only from inside the container.
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --app-dir backend
