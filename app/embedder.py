import time
import hashlib
import logging
from google import genai
from google.genai import types
from app.config import GEMINI_API_KEY, EMBEDDING_MODEL, EMBEDDING_BATCH_SIZE
from app.cache import query_embedding_cache

logger = logging.getLogger("codebase-intelligence")
client = genai.Client(api_key=GEMINI_API_KEY)


def embed_chunks(chunks: list[dict]) -> list[dict]:
    embedded = []
    for i in range(0, len(chunks), EMBEDDING_BATCH_SIZE):
        batch = chunks[i:i + EMBEDDING_BATCH_SIZE]
        contents = [chunk['content'] for chunk in batch]
        try:
            result = client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=contents,
                config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
            )
            for chunk, embedding in zip(batch, result.embeddings):
                chunk['embedding'] = embedding.values
                embedded.append(chunk)
        except Exception as e:
            logger.error(f"Batch embedding error (batch {i // EMBEDDING_BATCH_SIZE}): {e}")
            for chunk in batch:
                try:
                    result = client.models.embed_content(
                        model=EMBEDDING_MODEL,
                        contents=chunk['content'],
                        config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
                    )
                    chunk['embedding'] = result.embeddings[0].values
                    embedded.append(chunk)
                except Exception as inner_e:
                    logger.error(
                        f"Embedding error {chunk['file_path']}:"
                        f"{chunk['start_line']}-{chunk['end_line']}: {inner_e}"
                    )
        if i + EMBEDDING_BATCH_SIZE < len(chunks):
            time.sleep(1)
    return embedded


def get_query_embedding(question: str) -> list[float]:
    cache_key = hashlib.md5(question.encode()).hexdigest()
    cached = query_embedding_cache.get(cache_key)
    if cached is not None:
        return cached
    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=question,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
    )
    embedding = result.embeddings[0].values
    query_embedding_cache.set(cache_key, embedding)
    return embedding
