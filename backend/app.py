"""
FastAPI Backend Service for Speculative Decoding Inference.
Accepts ANY arbitrary prompt and provides high-speed speculative inference and real-time streaming.
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import psutil
from contextlib import asynccontextmanager
from typing import Optional, AsyncGenerator

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from backend.schemas import (
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    MetricsResponse,
    StreamChunk
)
from backend.models.target_loader import TargetModelLoader
from backend.models.draft_model import DraftModelManager
from backend.engine import SpeculativeEngine
from backend.metrics import global_metrics

# Global state
engine: Optional[SpeculativeEngine] = None
target_loader: Optional[TargetModelLoader] = None
draft_loader: Optional[DraftModelManager] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine, target_loader, draft_loader
    print("[*] Starting Speculative Decoding Backend Server...")
    
    # 1. Initialize Target Model
    target_model_path = os.getenv("TARGET_MODEL_PATH", "checkpoints/Llama-3.2-1B-Instruct-Q4_K_M.gguf")
    if not os.path.exists(target_model_path):
        target_model_path = os.getenv("TARGET_MODEL_PATH", "checkpoints/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf")

    n_ctx = int(os.getenv("TARGET_N_CTX", "4096"))
    n_threads = int(os.getenv("TARGET_N_THREADS", "8"))
    
    target_loader = TargetModelLoader(
        model_path=target_model_path,
        n_ctx=n_ctx,
        n_threads=n_threads
    )
    
    # Check if target model file exists
    if os.path.exists(target_model_path):
        target_loader.load()
    else:
        print(f"[!] Warning: Target model not found at '{target_model_path}'. Server will start in mock/unloaded mode until model is placed.")

    # 2. Initialize Draft Model
    draft_checkpoint_path = os.getenv("DRAFT_MODEL_PATH", "checkpoints/draft_model.pt")
    draft_loader = DraftModelManager(checkpoint_path=draft_checkpoint_path)
    if os.path.exists(draft_checkpoint_path):
        draft_loader.load()
    else:
        print(f"[*] Initializing draft model architecture...")
        draft_loader.load()

    # 3. Initialize Speculative Engine
    engine = SpeculativeEngine(
        target_model=target_loader,
        draft_model=draft_loader
    )
    print("[OK] Speculative Engine initialized and ready to receive requests.")
    yield
    print("[*] Shutting down Speculative Decoding Backend Server...")

app = FastAPI(
    title="Speculative Decoding Backend (Llama 3.1 8B + Custom Draft Model)",
    description="High-performance speculative decoding inference engine optimized for Intel Iris Xe / CPU.",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Returns model loading status, hardware memory, and CPU utilization."""
    mem = psutil.virtual_memory()
    return HealthResponse(
        status="healthy" if (target_loader and target_loader.is_loaded) else "degraded",
        target_model_loaded=bool(target_loader and target_loader.is_loaded),
        draft_model_loaded=bool(draft_loader and draft_loader.is_loaded),
        target_model_path=target_loader.model_path if target_loader else None,
        draft_params=draft_loader.model.count_parameters() if (draft_loader and draft_loader.model) else None,
        context_window=target_loader.n_ctx if target_loader else 4096,
        memory_used_mb=round((mem.total - mem.available) / (1024 * 1024), 1),
        memory_total_mb=round(mem.total / (1024 * 1024), 1),
        cpu_percent=psutil.cpu_percent(interval=None)
    )

@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Returns cumulative inference performance and speedup statistics."""
    return global_metrics.get_summary()

@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    """
    Accepts ANY arbitrary prompt and returns generated text with speculative decoding speedup metrics.
    """
    if not engine or not target_loader or not target_loader.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Target model is not loaded yet. Please ensure the target GGUF checkpoint is available."
        )

    if not req.prompt or not req.prompt.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Prompt cannot be empty."
        )

    try:
        response = engine.generate(req)
        return response
    except MemoryError:
        raise HTTPException(
            status_code=status.HTTP_507_INSUFFICIENT_STORAGE,
            detail="Inference ran out of system memory. Try reducing max_tokens or context length."
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}"
        )

@app.post("/generate/stream")
async def generate_stream(req: GenerateRequest):
    """
    Streams generated tokens via Server-Sent Events (SSE) with token origin tags.
    """
    if not engine or not target_loader or not target_loader.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Target model is not loaded yet."
        )

    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            for chunk in engine.generate_stream(req):
                yield chunk.model_dump_json()
        except Exception as e:
            err_chunk = StreamChunk(token="", token_id=-1, source="error", is_final=True, stats={"error": str(e)})
            yield err_chunk.model_dump_json()

    return EventSourceResponse(event_generator())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app:app", host="0.0.0.0", port=8000, reload=False)
