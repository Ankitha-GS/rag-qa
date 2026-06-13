"""
FastAPI backend — REST + streaming SSE endpoints.

Endpoints:
  POST /index          Upload + index documents
  POST /query          Single query (JSON response)
  POST /query/stream   Streaming query (Server-Sent Events)
  GET  /stats          Collection info
  GET  /health         Health check
"""

import json
import shutil
from pathlib import Path
from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.rag_engine import RAGEngine, RAGConfig

app = FastAPI(title="RAG Document Q&A API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

engine = RAGEngine()


class QueryRequest(BaseModel):
    question: str
    chunk_size: int = 500
    top_k: int = 4


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/stats")
def stats():
    return engine.get_collection_stats()


@app.post("/index")
async def index_documents(
    files: List[UploadFile] = File(...),
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    reset: bool = False,
):
    saved_paths = []
    for f in files:
        dest = UPLOAD_DIR / f.filename
        with dest.open("wb") as out:
            shutil.copyfileobj(f.file, out)
        saved_paths.append(str(dest))

    engine.config.chunk_size = chunk_size
    engine.config.chunk_overlap = chunk_overlap
    result = engine.index_documents(saved_paths, reset=reset)
    return {"message": "Indexed successfully", **result}


@app.post("/query")
def query(req: QueryRequest):
    engine.config.top_k = req.top_k
    result = engine.query(req.question)
    return {
        "answer": result.answer,
        "sources": result.sources,
        "latency_ms": result.latency_ms,
        "chunks_retrieved": result.chunks_retrieved,
    }


@app.post("/query/stream")
def query_stream(req: QueryRequest):
    """Returns a Server-Sent Events stream — each token is a separate event."""
    engine.config.top_k = req.top_k

    def event_stream():
        for token in engine.query_stream(req.question):
            yield f"data: {json.dumps({'token': token})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
