import os
import re
import json
import hashlib
import logging
from rank_bm25 import BM25Okapi
from app.config import BM25_PATH

logger = logging.getLogger("codebase-intelligence")

_bm25_cache = {}  # repo_name -> (bm25_obj, metadata, file_hash)


def tokenize(text: str) -> list[str]:
    tokens = re.findall(r'[a-zA-Z][a-z]*|[A-Z]+(?=[A-Z][a-z]|\b)', text)
    tokens += text.lower().split()
    return [t.lower() for t in tokens if len(t) > 1]


def _get_file_hash(path: str) -> str:
    with open(path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()


def save_bm25_index(repo_name: str, chunks: list[dict]):
    os.makedirs(BM25_PATH, exist_ok=True)
    corpus = [chunk['content'] for chunk in chunks]
    metadata = [{
        'file_path': chunk['file_path'],
        'start_line': chunk['start_line'],
        'end_line': chunk['end_line'],
        'node_type': chunk['node_type'],
        'content': chunk['content'],
    } for chunk in chunks]
    path = os.path.join(BM25_PATH, f"{repo_name}.json")
    with open(path, 'w') as f:
        json.dump({'corpus': corpus, 'metadata': metadata}, f)
    logger.info(f"Saved BM25 index for '{repo_name}' with {len(corpus)} documents")


def load_bm25_index(repo_name: str):
    path = os.path.join(BM25_PATH, f"{repo_name}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No BM25 index found for repo: {repo_name}")

    file_hash = _get_file_hash(path)
    cached = _bm25_cache.get(repo_name)
    if cached and cached[2] == file_hash:
        return cached[0], cached[1]

    with open(path, 'r') as f:
        index_data = json.load(f)
    tokenized_corpus = [tokenize(doc) for doc in index_data['corpus']]
    bm25 = BM25Okapi(tokenized_corpus)
    _bm25_cache[repo_name] = (bm25, index_data['metadata'], file_hash)
    return bm25, index_data['metadata']


def query_bm25(repo_name: str, question: str, n_results: int = 5) -> list[dict]:
    bm25, metadata = load_bm25_index(repo_name)
    tokenized_query = tokenize(question)
    scores = bm25.get_scores(tokenized_query)
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:n_results]
    results = []
    for idx, score in ranked:
        if score > 0:
            results.append({**metadata[idx], 'bm25_score': round(float(score), 4)})
    return results


def invalidate_bm25_cache(repo_name: str):
    _bm25_cache.pop(repo_name, None)
