# FedMed — public demo

Privacy-preserving federated learning dashboard for medical MRI research, with a
medical-report analysis workflow.

This repository is the **deployable public copy**: the FastAPI application, the
pre-built dashboard, and the Render blueprint. It is published so the demo can be
hosted continuously; the full development repository (training pipeline, tests,
federated-learning core) stays private.

## What the deployed demo does

| Area | Status |
|---|---|
| Dashboard, hospitals, datasets, model results, privacy pages | Live, fed by the API |
| Medical-report analysis (upload a PDF / text / image report) | Live for text and digital PDFs |
| AI-written report sections | Only when `FEDMED_AI_API_KEY` is set |
| Federated training | **Not available** — answers `503 training_unavailable` (no torch on the web tier) |
| OCR of scanned pages or photos | **Not available** — needs system `tesseract`, not installed on this tier |

Nothing here is real patient data. Every figure is synthetic and the UI says so.

## Layout

```
backend/app              FastAPI application
backend/requirements-web.txt   the lean web dependency set (no torch / MONAI / Flower)
frontend/dist            pre-built dashboard (served by the API, same origin)
render.yaml              Render blueprint: one Python service, one origin
```

## Run it locally

```bash
python -m venv .venv
.venv/bin/pip install -r backend/requirements-web.txt      # Windows: .venv\Scripts\pip

FEDMED_SERVE_FRONTEND=true \
FEDMED_FRONTEND_DIR=../frontend/dist \
FEDMED_AUTH_REQUIRED_FOR_READS=false \
FEDMED_MEDICAL_REPORTS_PUBLIC_DEMO=true \
.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000 --app-dir backend
```

Then open <http://127.0.0.1:8000>. API docs at <http://127.0.0.1:8000/docs>.

## Rebuild the dashboard

Render's Python runtime has no Node, so the built pages are committed:

```bash
cd frontend
VITE_API_SAME_ORIGIN=true VITE_API_BASE_URL="" VITE_USE_DEMO_DATA=false \
VITE_HASH_ROUTER=false VITE_API_KEY="" npm run build
```

`VITE_API_SAME_ORIGIN=true` makes every request relative. Without it the bundle falls
back to `http://127.0.0.1:8000` — the *visitor's* own machine — and the page shows a
bare "Failed to fetch".
