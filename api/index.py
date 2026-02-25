"""api/index.py — Vercel ASGI entry-point for GodLocal.

Vercel looks for `app` (ASGI) or `handler` (WSGI) at module level.
We re-export the FastAPI `app` from main.py.
"""
import sys
import os

# Ensure project root is on path so `from main import app` works
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from main import app  # noqa: F401  — Vercel picks up `app` automatically
except Exception as e:  # pragma: no cover
    # Graceful fallback: minimal health endpoint so deploy doesn't hard-fail
    from fastapi import FastAPI
    app = FastAPI(title="GodLocal (import error fallback)")

    @app.get("/")
    def root():
        return {"status": "error", "detail": str(e)}
