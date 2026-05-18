from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google import genai
from google.genai import types
import os
from dotenv import load_dotenv

from app.repo_handler import clone_repository, filter_repo_files
from app.chunker import chunk_files
from app.embedder import embed_chunks
from app.vector_store import store_embeddings, query_collection
from app.bm25_store import save_bm25_index, query_bm25

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

app = FastAPI(title="Codebase Intelligence Engine")


class IndexRequest(BaseModel):
    repo_url: str


class QueryRequest(BaseModel):
    repo_url: str
    question: str
    n_results: int = 5


def repo_url_to_name(repo_url: str) -> str:
    return repo_url.strip('/').split('github.com/')[-1].replace('/', '_')


@app.post("/index")
def index_repo(request: IndexRequest):
    try:
        repo_name = repo_url_to_name(request.repo_url)
        path = clone_repository(request.repo_url)
        files = filter_repo_files(path)
        chunks = chunk_files(files)
        embedded = embed_chunks(chunks)
        store_embeddings(repo_name, embedded)
        save_bm25_index(repo_name, embedded)
        return {
            "status": "success",
            "repo": repo_name,
            "chunks_indexed": len(embedded)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query")
def query_repo(request: QueryRequest):
    try:
        repo_name = repo_url_to_name(request.repo_url)

        result = client.models.embed_content(
            model="gemini-embedding-001",
            contents=request.question,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
        )
        query_vector = result.embeddings[0].values

        results = query_collection(repo_name, query_vector, request.n_results)

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

        return {
            "question": request.question,
            "results": matches
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query/hybrid")
def query_hybrid(request: QueryRequest):
    try:
        repo_name = repo_url_to_name(request.repo_url)

        result = client.models.embed_content(
            model="gemini-embedding-001",
            contents=request.question,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
        )
        query_vector = result.embeddings[0].values
        dense_results = query_collection(repo_name, query_vector, request.n_results * 2)

        sparse_results = query_bm25(repo_name, request.question, request.n_results * 2)

        K = 60
        rrf_scores = {}

        for rank, (doc, meta, distance) in enumerate(zip(
            dense_results['documents'][0],
            dense_results['metadatas'][0],
            dense_results['distances'][0]
        ), start=1):
            key = f"{meta['file_path']}:{meta['start_line']}"
            rrf_scores[key] = {
                **meta,
                'content': doc,
                'dense_score': round(1 - distance, 4),
                'bm25_score': 0.0,
                'rrf_score': 1 / (K + rank)
            }

        for rank, r in enumerate(sparse_results, start=1):
            key = f"{r['file_path']}:{r['start_line']}"
            bm25_rrf = 1 / (K + rank)
            if key in rrf_scores:
                rrf_scores[key]['rrf_score'] += bm25_rrf
                rrf_scores[key]['bm25_score'] = r['bm25_score']
            else:
                rrf_scores[key] = {
                    **r,
                    'dense_score': 0.0,
                    'rrf_score': bm25_rrf
                }

        
        for key in rrf_scores:
            rrf_scores[key]['rrf_score'] = round(rrf_scores[key]['rrf_score'], 6)

        ranked = sorted(
            rrf_scores.values(),
            key=lambda x: x['rrf_score'],
            reverse=True
        )

        return {
            "question": request.question,
            "results": ranked[:request.n_results]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok"}