"""HTTP API and web UI: chat with any stage, inspect traces, watch the metrics dashboard."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .llm import LLMError
from .runtime import Runtime
from .stages import STAGES

STATIC = Path(__file__).resolve().parent / "static"


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    stage: str = "v6"
    session_id: str = "web"


class FeedbackRequest(BaseModel):
    trace_id: str
    value: int = Field(ge=-1, le=1)


class ResetRequest(BaseModel):
    session_id: str = "web"


def create_app(runtime: Runtime) -> FastAPI:
    app = FastAPI(title="Ari: observable assistant robot", version="1.0.0")
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    @app.get("/api/health")
    def health() -> dict:
        return {"status": "ok", "provider": runtime.settings.provider, "model": runtime.llm.model,
                "embedder": runtime.store.embedder.id, "chunks": len(runtime.store.chunks),
                "min_score": runtime.min_score}

    @app.get("/api/stages")
    def stages() -> list[dict]:
        return [asdict(s) for s in STAGES.values()]

    @app.post("/api/chat")
    def chat(req: ChatRequest) -> dict:
        if req.stage not in STAGES:
            raise HTTPException(400, f"unknown stage {req.stage!r}")
        try:
            reply = runtime.agent(req.stage).chat(req.message, session_id=req.session_id)
        except LLMError as exc:
            raise HTTPException(502, f"model server error: {exc}") from exc
        return asdict(reply)

    @app.post("/api/reset")
    def reset(req: ResetRequest) -> dict:
        for key in STAGES:
            runtime.memory.reset(f"{req.session_id}:{key}")
        return {"reset": req.session_id}

    @app.post("/api/feedback")
    def feedback(req: FeedbackRequest) -> dict:
        if not runtime.tracer.set_feedback(req.trace_id, req.value):
            raise HTTPException(404, "trace not found")
        return {"ok": True}

    @app.get("/api/traces")
    def traces(limit: int = 50, stage: str | None = None) -> list[dict]:
        return runtime.tracer.list_traces(min(limit, 500), stage)

    @app.get("/api/traces/{trace_id}")
    def trace(trace_id: str) -> dict:
        found = runtime.tracer.get_trace(trace_id)
        if found is None:
            raise HTTPException(404, "trace not found")
        return found

    @app.get("/api/metrics")
    def metrics() -> dict:
        return runtime.tracer.metrics()

    return app
