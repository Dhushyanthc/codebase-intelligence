import logging
import concurrent.futures
from app.config import RRF_K
from app.vector_store import query_collection
from app.bm25_store import query_bm25
from app.embedder import get_query_embedding

logger = logging.getLogger("codebase-intelligence")


def hybrid_search(repo_name: str, question: str, n_results: int = 5) -> list[dict]:
    fetch_count = n_results * 2

    # Run embedding and BM25 in parallel — they don't depend on each other
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        embedding_future = executor.submit(get_query_embedding, question)
        bm25_future = executor.submit(query_bm25, repo_name, question, fetch_count)

        query_vector = embedding_future.result()
        try:
            sparse_results = bm25_future.result()
        except FileNotFoundError:
            logger.warning(f"[retrieval] BM25 index not found for {repo_name}, falling back to dense-only search")
            sparse_results = []

    # ChromaDB query depends on the embedding, so it runs after
    dense_results = query_collection(repo_name, query_vector, fetch_count)

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
            'rrf_score': 1 / (RRF_K + rank)
        }

    for rank, r in enumerate(sparse_results, start=1):
        key = f"{r['file_path']}:{r['start_line']}"
        bm25_rrf = 1 / (RRF_K + rank)
        if key in rrf_scores:
            rrf_scores[key]['rrf_score'] += bm25_rrf
            rrf_scores[key]['bm25_score'] = r.get('bm25_score', 0.0)
        else:
            rrf_scores[key] = {**r, 'dense_score': 0.0, 'rrf_score': bm25_rrf}

    for key in rrf_scores:
        rrf_scores[key]['rrf_score'] = round(rrf_scores[key]['rrf_score'], 6)

    ranked = sorted(rrf_scores.values(), key=lambda x: x['rrf_score'], reverse=True)
    return ranked[:n_results]
