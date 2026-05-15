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


@app.get("/health")
def health():
    return {"status": "ok"}