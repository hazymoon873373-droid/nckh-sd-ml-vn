#!/usr/bin/env python3
"""FastAPI server for local preview of VN-QUANT NCKH Web Workbench (mirrors ../docs)."""

from __future__ import annotations

import pathlib
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from research_pipeline import run_research_pipeline

BASE_DIR = pathlib.Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR.parent / "docs"

app = FastAPI(title="VN-QUANT NCKH: Distributional ML + Almost SD (HOSE)")
app.mount("/css", StaticFiles(directory=str(DOCS_DIR / "css")), name="css")
app.mount("/js", StaticFiles(directory=str(DOCS_DIR / "js")), name="js")
app.mount("/data", StaticFiles(directory=str(DOCS_DIR / "data")), name="data")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(DOCS_DIR / "index.html")


@app.post("/api/refresh")
def refresh_data() -> JSONResponse:
    bundle = run_research_pipeline(force_refresh=True)
    return JSONResponse({"status": "ok", "date_range": bundle["date_range"]})


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8505, log_level="warning")
