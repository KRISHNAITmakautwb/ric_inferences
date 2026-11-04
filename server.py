"""
RIC Pipeline — server.py
FastAPI server exposing the RIC pipeline as a REST API.

Endpoints:
  POST /infer                     → Run a query through the RIC pipeline
  GET  /stats                     → Get cache + routing statistics
  GET  /health                    → Health check (local + cloud connectivity)
  DELETE /cache/clear             → Clear all cached responses

  ── OpenAI-Compatible (for Hermes Agent / Claude CLI / any SDK) ──
  POST /v1/chat/completions       → OpenAI chat format → RIC pipeline
  GET  /v1/models                 → List available models
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Any
import time
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ric.pipeline import RICPipeline
from ric.utils import get_logger

logger = get_logger("ric.server")

# ── Global Pipeline Instance ──────────────────

pipeline: Optional[RICPipeline] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / Shutdown lifecycle."""
    global pipeline
    logger.info("Starting RIC server...")
    pipeline = RICPipeline()
    await pipeline.init()
    logger.info("RIC server ready.")
    yield
    logger.info("Shutting down RIC server...")
    await pipeline.close()


# ── FastAPI App ───────────────────────────────

app = FastAPI(
    title="RIC Inference Pipeline",
    description="Router → Inference → Cache | Mac Mini M3 Gemma + OmniRoute Cloud",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response Schemas ─────────────────

class InferRequest(BaseModel):
    query: str
    system: Optional[str] = None
    skip_cache: bool = False

    model_config = {"json_schema_extra": {
        "example": {
            "query": "What is the capital of France?",
            "system": "You are a helpful assistant.",
            "skip_cache": False
        }
    }}


class InferResponse(BaseModel):
    query: str
    response: str
    destination: str
    routing_reason: str
    routing_confidence: float
    latency_ms: float
    total_tokens: int
    model: str
    from_cache: bool
    success: bool
    error: Optional[str] = None


# ── Endpoints ─────────────────────────────────

@app.post("/infer", response_model=InferResponse)
async def infer(req: InferRequest):
    """
    Run a query through the full RIC pipeline.

    - Short/simple queries → Local Mac Mini (Gemma 3 12B, free)
    - Long/complex queries → Cloud OmniRoute
    - Repeated queries     → Cache (instant, 0 tokens)
    """
    result = await pipeline.run(
        query=req.query,
        system=req.system,
        skip_cache=req.skip_cache,
    )
    return InferResponse(**{k: v for k, v in result.to_dict().items() if k != "prompt_tokens" and k != "completion_tokens"})


@app.get("/stats")
async def get_stats():
    """Return pipeline and cache statistics."""
    return await pipeline.get_stats()


@app.get("/health")
async def health_check():
    """Check connectivity to local and cloud backends."""
    import httpx
    from ric.utils import load_config
    cfg = load_config()

    local_url = f"http://{cfg['local']['host']}:{cfg['local']['port']}/api/tags"
    cloud_url = f"{cfg['cloud']['base_url'].rstrip('/')}/models"

    results = {}

    # Check local Ollama
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(local_url)
        models = [m["name"] for m in resp.json().get("models", [])]
        results["local"] = {
            "status": "ok",
            "host": cfg["local"]["host"],
            "model": cfg["local"]["model"],
            "available_models": models,
            "target_model_loaded": cfg["local"]["model"] in models,
        }
    except Exception as e:
        results["local"] = {"status": "error", "error": str(e)}

    # Check cloud OmniRoute
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(
                cloud_url,
                headers={"Authorization": f"Bearer {cfg['cloud']['api_key']}"},
            )
        results["cloud"] = {"status": "ok", "base_url": cfg["cloud"]["base_url"]}
    except Exception as e:
        results["cloud"] = {"status": "error", "error": str(e)}

    results["overall"] = (
        "healthy"
        if results.get("local", {}).get("status") == "ok"
        or results.get("cloud", {}).get("status") == "ok"
        else "degraded"
    )
    return results


@app.delete("/cache/clear")
async def clear_cache():
    """Clear all cached inference responses."""
    await pipeline.clear_cache()
    return {"message": "Cache cleared successfully."}


# ── OpenAI-Compatible Endpoints (for Hermes / Claude CLI / any SDK) ──

class OAIMessage(BaseModel):
    role: str
    content: str


class OAIChatRequest(BaseModel):
    model: Optional[str] = None
    messages: List[OAIMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    # Extra RIC-specific (optional)
    skip_cache: Optional[bool] = False


@app.post("/v1/chat/completions")
async def openai_chat_completions(req: OAIChatRequest):
    """
    OpenAI-compatible chat completions endpoint.
    Supports both streaming (SSE) and non-streaming responses.
    Hermes Agent (provider: custom), Claude CLI, LiteLLM, etc. connect here.
    """
    import json
    from fastapi.responses import StreamingResponse

    # Extract system prompt and last user message
    system_prompt = None
    user_content = ""
    for msg in req.messages:
        if msg.role == "system":
            system_prompt = msg.content
        elif msg.role == "user":
            user_content = msg.content  # keep last user message

    if not user_content:
        raise HTTPException(status_code=400, detail="No user message found in messages.")

    # Run through RIC pipeline (always gets full response first)
    result = await pipeline.run(
        query=user_content,
        system=system_prompt,
        skip_cache=req.skip_cache or False,
    )

    completion_id = f"ric-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    # ── Streaming mode (Hermes Agent uses this) ──────────────────────────────
    if req.stream:
        async def event_stream():
            # Chunk 1: role delta
            role_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": result.model,
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(role_chunk)}\n\n"

            # Chunk 2: full content as single chunk
            # (RIC doesn't stream internally, so we send it all at once)
            content_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": result.model,
                "choices": [{"index": 0, "delta": {"content": result.response}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(content_chunk)}\n\n"

            # Chunk 3: stop signal
            stop_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": result.model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": result.prompt_tokens,
                    "completion_tokens": result.completion_tokens,
                    "total_tokens": result.total_tokens,
                },
            }
            yield f"data: {json.dumps(stop_chunk)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    # ── Non-streaming mode ───────────────────────────────────────────────────

    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": result.model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": result.response,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "total_tokens": result.total_tokens,
        },
        # RIC metadata (extra, ignored by standard clients)
        "ric": {
            "destination": result.destination,
            "from_cache": result.from_cache,
            "routing_reason": result.routing_reason,
            "latency_ms": round(result.latency_ms, 2),
        },
    }


@app.get("/v1/models")
async def list_models():
    """
    OpenAI-compatible model list endpoint.
    Returns both local (Gemma) and cloud models available through RIC.
    NOTE: context_length reported as 131072 (Gemma 3 12B real window = 128K).
    """
    from ric.utils import load_config
    cfg = load_config()
    # Gemma 3 12B true context = 128K. Report 131072 so Hermes 64K minimum passes.
    CTX = 131072
    models = [
        {
            "id": cfg["local"]["model"],
            "object": "model",
            "created": 1720000000,
            "owned_by": "ric-local",
            "description": "Local Mac Mini M3 — Gemma 3 12B via Ollama",
            "context_length": CTX,
            "context_window": CTX,
            "max_context_length": CTX,
        },
        {
            "id": cfg["cloud"]["model"],
            "object": "model",
            "created": 1720000000,
            "owned_by": "ric-cloud",
            "description": "Cloud model via OmniRoute",
            "context_length": CTX,
            "context_window": CTX,
            "max_context_length": CTX,
        },
        {
            "id": "ric-auto",
            "object": "model",
            "created": 1720000000,
            "owned_by": "ric-pipeline",
            "description": "Auto-routed: RIC selects local or cloud automatically",
            "context_length": CTX,
            "context_window": CTX,
            "max_context_length": CTX,
        },
    ]
    return {"object": "list", "data": models}


# ── Ollama-Compatible Stub Endpoints ─────────────────────────────────────────
# Open WebUI, LM Studio, and other clients probe these endpoints to detect
# if the server is Ollama. Return stub responses to stop 404 noise.

@app.get("/version")
async def version():
    return {"version": "0.3.0-ric"}

@app.get("/api/tags")
async def api_tags():
    """Ollama-compatible model list with correct context sizes."""
    from ric.utils import load_config
    cfg = load_config()
    return {
        "models": [
            {"name": cfg["local"]["model"], "modified_at": "2025-01-01T00:00:00Z",
             "size": 8000000000, "digest": "ric-local",
             "details": {"parameter_size": "12B", "context_length": 131072}},
            {"name": cfg["cloud"]["model"], "modified_at": "2025-01-01T00:00:00Z",
             "size": 0, "digest": "ric-cloud",
             "details": {"parameter_size": "unknown", "context_length": 131072}},
        ]
    }

@app.post("/api/show")
async def api_show():
    """Ollama-compatible model info stub."""
    return {"details": {"family": "gemma", "parameter_size": "12B"}, "modelfile": ""}

@app.get("/api/v1/models")
async def api_v1_models():
    """Alternate models endpoint some clients probe."""
    return await list_models()

@app.get("/v1/props")
@app.get("/props")
async def props():
    """Stub for Open WebUI props probe."""
    return {"status": "ok", "server": "ric-pipeline"}

@app.get("/v1/models/{model_id:path}")
async def get_model_by_id(model_id: str):
    """Individual model lookup with correct context window."""
    return {
        "id": model_id,
        "object": "model",
        "created": 1720000000,
        "owned_by": "ric-pipeline",
        "context_length": 131072,
        "context_window": 131072,
        "max_context_length": 131072,
    }


# ── Main Entry Point ──────────────────────────

if __name__ == "__main__":
    import uvicorn
    from ric.utils import load_config
    cfg = load_config()
    srv = cfg["server"]
    uvicorn.run(
        "server:app",
        host=srv.get("host", "0.0.0.0"),
        port=srv.get("port", 8000),
        workers=1,               # Use 1 for async (SQLite not thread-safe with multiple)
        reload=False,
        log_level="info",
    )
