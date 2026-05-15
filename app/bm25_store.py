import os
import json
from rank_bm25 import BM25Okapi

BM25_PATH = os.path.expanduser("~/codebase-intelligence/bm25_db")


def tokenize(text: str) -> list[str]:
    return text.lower().split()


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

    index_data = {
        'corpus': corpus,
        'metadata': metadata
    }

    path = os.path.join(BM25_PATH, f"{repo_name}.json")
    with open(path, 'w') as f:
        json.dump(index_data, f)

    print(f"[bm25] Saved index for '{repo_name}' with {len(corpus)} documents")


def load_bm25_index(repo_name: str):
    path = os.path.join(BM25_PATH, f"{repo_name}.json")

    if not os.path.exists(path):
        raise FileNotFoundError(f"No BM25 index found for repo: {repo_name}")

    with open(path, 'r') as f:
        index_data = json.load(f)

    tokenized_corpus = [tokenize(doc) for doc in index_data['corpus']]
    bm25 = BM25Okapi(tokenized_corpus)

    return bm25, index_data['metadata']


def query_bm25(repo_name: str, question: str, n_results: int = 5) -> list[dict]:
    bm25, metadata = load_bm25_index(repo_name)

    tokenized_query = tokenize(question)
    scores = bm25.get_scores(tokenized_query)

    ranked = sorted(
        enumerate(scores),
        key=lambda x: x[1],
        reverse=True
    )[:n_results]

    results = []
    for idx, score in ranked:
        if score > 0:
            results.append({
                **metadata[idx],
                'bm25_score': round(float(score), 4)
            })

    return results