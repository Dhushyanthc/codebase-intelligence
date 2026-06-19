import logging
import chromadb
from app.config import CHROMA_PATH

logger = logging.getLogger("codebase-intelligence")

_client = None


def get_client():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=CHROMA_PATH)
    return _client


def get_collection(repo_name: str):
    client = get_client()
    return client.get_or_create_collection(
        name=repo_name,
        metadata={"hnsw:space": "cosine"}
    )


def store_embeddings(repo_name: str, embedded_chunks: list[dict]):
    collection = get_collection(repo_name)
    ids, embeddings, documents, metadatas = [], [], [], []
    for i, chunk in enumerate(embedded_chunks):
        ids.append(f"{repo_name}_chunk_{i}")
        embeddings.append(chunk['embedding'])
        documents.append(chunk['content'])
        metadatas.append({
            'file_path': chunk['file_path'],
            'start_line': chunk['start_line'],
            'end_line': chunk['end_line'],
            'node_type': chunk['node_type'],
        })
    collection.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
    logger.info(f"Stored {len(ids)} chunks in collection '{repo_name}'")


def query_collection(repo_name: str, query_embedding: list[float], n_results: int = 5):
    collection = get_collection(repo_name)
    return collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=['documents', 'metadatas', 'distances']
    )


def delete_collection(repo_name: str):
    client = get_client()
    try:
        client.delete_collection(name=repo_name)
    except Exception:
        pass
