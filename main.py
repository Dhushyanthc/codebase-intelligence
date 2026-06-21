import logging
import os
import time

from fastapi import FastAPI, HTTPException, Request, Security, Depends, APIRouter
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from google import genai
from google.genai import types
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from app.config import (
    GEMINI_API_KEY, API_SECRET_KEY,
    BM25_PATH, MAX_N_RESULTS, MAX_QUESTION_LENGTH,
    INDEX_RATE_LIMIT, QUERY_RATE_LIMIT,
)
from app.repo_handler import clone_repository, filter_repo_files, is_valid_git_url
from app.chunker import chunk_files
from app.embedder import embed_chunks, get_query_embedding
from app.vector_store import store_embeddings, query_collection, delete_collection, get_collection, get_client as get_chroma_client
from app.bm25_store import save_bm25_index, invalidate_bm25_cache
from app.retrieval import hybrid_search
from app.generator import generate_answer
from app.agent import agent, AgentState
from app.eviction import record_index_time, cleanup_clone_dir, cleanup_stale_repos

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("codebase-intelligence")

_gemini_health_cache = {"status": None, "checked_at": 0}
GEMINI_HEALTH_TTL = 300

genai_client = genai.Client(api_key=GEMINI_API_KEY)

limiter = Limiter(key_func=get_remote_address)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str = Security(api_key_header)):
    if not API_SECRET_KEY:
        return
    if api_key != API_SECRET_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


app = FastAPI(title="Codebase Intelligence Engine")
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded. Try again later."})


@app.on_event("startup")
def startup_eviction():
    stale = cleanup_stale_repos()
    if stale:
        logger.info(f"Evicted {len(stale)} stale repos on startup: {stale}")


@app.get("/health")
def health():
    checks = {}

    try:
        chroma_client = get_chroma_client()
        chroma_client.heartbeat()
        checks["chromadb"] = "ok"
    except Exception as e:
        checks["chromadb"] = f"error: {str(e)}"

    checks["bm25_storage"] = "ok" if os.path.isdir(BM25_PATH) else "error: directory missing"

    now = time.time()
    if now - _gemini_health_cache["checked_at"] > GEMINI_HEALTH_TTL:
        try:
            genai_client.models.embed_content(
                model="gemini-embedding-001",
                contents="health check",
                config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
            )
            _gemini_health_cache["status"] = "ok"
        except Exception as e:
            _gemini_health_cache["status"] = f"error: {str(e)}"
        _gemini_health_cache["checked_at"] = now
    checks["gemini_api"] = _gemini_health_cache["status"] or "not checked yet"

    status = "healthy" if all(v == "ok" for v in checks.values()) else "degraded"
    return {"status": status, "checks": checks}


class IndexRequest(BaseModel):
    repo_url: str

    @field_validator('repo_url')
    @classmethod
    def validate_repo_url(cls, v):
        if not is_valid_git_url(v):
            raise ValueError("Invalid GitHub repository URL")
        return v


class QueryRequest(BaseModel):
    repo_url: str
    question: str = Field(..., max_length=MAX_QUESTION_LENGTH)
    n_results: int = Field(default=5, ge=1, le=MAX_N_RESULTS)

    @field_validator('repo_url')
    @classmethod
    def validate_repo_url(cls, v):
        if not is_valid_git_url(v):
            raise ValueError("Invalid GitHub repository URL")
        return v


router = APIRouter(dependencies=[Depends(verify_api_key)])


def repo_url_to_name(repo_url: str) -> str:
    name = repo_url.strip('/').split('github.com/')[-1].replace('/', '_')
    if name.endswith('.git'):
        name = name[:-4]
    return name


def is_repo_indexed(repo_name: str) -> bool:
    try:
        collection = get_collection(repo_name)
        if collection.count() == 0:
            return False
    except Exception:
        return False
    bm25_path = os.path.join(BM25_PATH, f"{repo_name}.json")
    return os.path.exists(bm25_path)


@router.post("/index")
@limiter.limit(INDEX_RATE_LIMIT)
def index_repo(request: Request, body: IndexRequest):
    try:
        repo_name = repo_url_to_name(body.repo_url)
        delete_collection(repo_name)
        invalidate_bm25_cache(repo_name)
        path = clone_repository(body.repo_url)
        files = filter_repo_files(path)
        chunks = chunk_files(files)
        embedded = embed_chunks(chunks)
        store_embeddings(repo_name, embedded)
        save_bm25_index(repo_name, embedded)
        record_index_time(repo_name)
        cleanup_clone_dir(repo_name)
        return {
            "status": "success",
            "repo": repo_name,
            "chunks_indexed": len(embedded)
        }
    except Exception as e:
        logger.exception("Error in /index")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/query")
@limiter.limit(QUERY_RATE_LIMIT)
def query_repo(request: Request, body: QueryRequest):
    try:
        repo_name = repo_url_to_name(body.repo_url)
        if not is_repo_indexed(repo_name):
            raise HTTPException(status_code=404, detail=f"Repository '{repo_name}' has not been indexed yet. Call POST /index first.")
        query_vector = get_query_embedding(body.question)
        results = query_collection(repo_name, query_vector, body.n_results)

        matches = []
        for doc, meta, distance in zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        ):
            matches.append({
                "file_path": meta['file_path'],
                "start_line": meta['start_line'],
                "end_line": meta['end_line'],
                "node_type": meta['node_type'],
                "content": doc,
                "relevance_score": round(1 - distance, 4)
            })
        return {"question": body.question, "results": matches}
    except Exception as e:
        logger.exception("Error in /query")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/query/hybrid")
@limiter.limit(QUERY_RATE_LIMIT)
def query_hybrid(request: Request, body: QueryRequest):
    try:
        repo_name = repo_url_to_name(body.repo_url)
        if not is_repo_indexed(repo_name):
            raise HTTPException(status_code=404, detail=f"Repository '{repo_name}' has not been indexed yet. Call POST /index first.")
        ranked = hybrid_search(repo_name, body.question, body.n_results)
        return {"question": body.question, "results": ranked}
    except Exception as e:
        logger.exception("Error in /query/hybrid")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/ask")
@limiter.limit(QUERY_RATE_LIMIT)
def ask(request: Request, body: QueryRequest):
    try:
        repo_name = repo_url_to_name(body.repo_url)
        if not is_repo_indexed(repo_name):
            raise HTTPException(status_code=404, detail=f"Repository '{repo_name}' has not been indexed yet. Call POST /index first.")
        top_chunks = hybrid_search(repo_name, body.question, body.n_results)
        result = generate_answer(body.question, top_chunks)
        return {
            "question": body.question,
            "answer": result['answer'],
            "chunks_used": result['chunks_used'],
            "sources": [
                {
                    "file_path": c['file_path'],
                    "start_line": c['start_line'],
                    "end_line": c['end_line'],
                }
                for c in top_chunks
            ]
        }
    except Exception as e:
        logger.exception("Error in /ask")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/agent/ask")
@limiter.limit(QUERY_RATE_LIMIT)
def agent_ask(request: Request, body: QueryRequest):
    try:
        repo_name = repo_url_to_name(body.repo_url)
        if not is_repo_indexed(repo_name):
            raise HTTPException(status_code=404, detail=f"Repository '{repo_name}' has not been indexed yet. Call POST /index first.")
        initial_state: AgentState = {
            "question": body.question,
            "repo_url": body.repo_url,
            "repo_name": repo_name,
            "tool_decision": "",
            "pre_route_decision": "",
            "clarification_question": "",
            "retrieved_chunks": [],
            "final_answer": "",
            "sources": []
        }
        final_state = agent.invoke(initial_state)
        return {
            "question": body.question,
            "tool_used": final_state['tool_decision'],
            "answer": final_state['final_answer'],
            "sources": final_state['sources'],
            "needs_clarification": final_state['tool_decision'] == 'clarify'
        }
    except Exception as e:
        logger.exception("Error in /agent/ask")
        raise HTTPException(status_code=500, detail="Internal server error")


app.include_router(router)
