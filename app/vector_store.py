import os
import chromadb

CHROMA_PATH = os.path.expanduser("~/codebase-intelligence/chroma_db")

def get_collection(repo_name: str):
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(
        name=repo_name,
        metadata={"hnsw:space": "cosine"}
    )
    return collection


def store_embeddings(repo_name: str, embedded_chunks: list[dict]):
    collection = get_collection(repo_name)

    ids = []
    embeddings = []
    documents = []
    metadatas = []

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

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas
    )
    print(f"[stored] {len(ids)} chunks in collection '{repo_name}'")


def query_collection(repo_name: str, query_embedding: list[float], n_results: int = 5):
    collection = get_collection(repo_name)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=['documents', 'metadatas', 'distances']
    )
    return results