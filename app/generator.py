import logging
from google import genai
from google.genai import types
from app.config import GEMINI_API_KEY, GENERATION_MODEL, GENERATION_MAX_TOKENS, GENERATION_TEMPERATURE

logger = logging.getLogger("codebase-intelligence")
client = genai.Client(api_key=GEMINI_API_KEY)

SYSTEM_PROMPT = """You are a code assistant that answers questions ONLY using the provided code chunks. Follow these rules strictly:

1. ONLY use information present in the code chunks below. Do not use your own knowledge about libraries, frameworks, or coding patterns.
2. If the answer is not in the provided chunks, say "Based on the retrieved code, I cannot find information about this." Do not guess or infer beyond what the code shows.
3. Every claim you make must reference a specific file and line number from the chunks.
4. If a chunk partially answers the question, say what you can confirm from the code and explicitly state what is not covered in the retrieved context.
5. Do not explain how things "typically work" or "usually are implemented" — only describe what the actual code does."""


def format_context(chunks: list[dict]) -> str:
    context_parts = []
    for i, chunk in enumerate(chunks, start=1):
        context_parts.append(
            f"--- Chunk {i} ---\n"
            f"File: {chunk['file_path']}\n"
            f"Lines: {chunk['start_line']}-{chunk['end_line']}\n"
            f"Type: {chunk['node_type']}\n"
            f"Code:\n{chunk['content']}\n"
        )
    return '\n'.join(context_parts)


def generate_answer(question: str, chunks: list[dict]) -> dict:
    if not chunks:
        return {"answer": "No relevant code chunks found for this question.", "chunks_used": 0}

    context = format_context(chunks)
    prompt = f"Question: {question}\n\nRelevant code from the codebase:\n\n{context}\n\nAnswer the question using ONLY the code chunks above. Every claim must cite a specific file and line number. If the code chunks don't contain the answer, say so explicitly."

    response = client.models.generate_content(
        model=GENERATION_MODEL,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=GENERATION_TEMPERATURE,
            max_output_tokens=GENERATION_MAX_TOKENS,
        ),
        contents=prompt
    )
    text = response.text or ""
    if not text and response.candidates:
        for part in reversed(response.candidates[0].content.parts):
            if getattr(part, 'text', None):
                text = part.text
                break
    return {"answer": text, "chunks_used": len(chunks)}
