import os
import time
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

EMBEDDING_MODEL = "gemini-embedding-001"
BATCH_SIZE = 5


def embed_chunks(chunks: list[dict]) -> list[dict]:
    embedded = []

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]

        for chunk in batch:
            try:
                result = client.models.embed_content(
                    model=EMBEDDING_MODEL,
                    contents=chunk['content'],
                    config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
                )
                chunk['embedding'] = result.embeddings[0].values
                embedded.append(chunk)
                print(f"[embedded] {chunk['file_path']} lines {chunk['start_line']}-{chunk['end_line']}")
            except Exception as e:
                print(f"[embedding error] {chunk['file_path']} lines {chunk['start_line']}-{chunk['end_line']}: {e}")

        if i + BATCH_SIZE < len(chunks):
            time.sleep(1)

    return embedded