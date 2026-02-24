"""extensions/web/static_router.py
Mount this in godlocal_v6.py to serve the browser UI at GET /

Add to godlocal_v6.py:
    from extensions.web.static_router import mount_ui
    mount_ui(app)   # after app = FastAPI(...)
"""
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


def mount_ui(app: FastAPI) -> None:
    """Serve static/index.html at GET /  and static assets at /static/*"""
    static_dir = Path(__file__).parent.parent.parent / "static"
    static_dir.mkdir(exist_ok=True)

    # Mount /static/* for future JS/CSS/image assets
    if (static_dir).exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/", include_in_schema=False)
    async def index():
        return FileResponse(str(static_dir / "index.html"))
