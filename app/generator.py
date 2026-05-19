from google import genai
from google.genai import types
import os
from dotenv import load_dotenv

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

SYSTEM_PROMPT = """You are an expert code assistant that answers questions about software codebases.

You will be given:
1. A question about a codebase
2. Relevant code chunks retrieved from that codebase, each with file path and line numbers

Your job:
- Answer the question based ONLY on the provided code chunks
- Always cite the specific file and line numbers when referencing code
- If the answer cannot be found in the provided chunks, say so explicitly
- Be precise and technical — the user is a developer
- Do not hallucinate code or behavior that isn't in the chunks

Format citations as: `filename.py` (lines X-Y)"""


def format_context(chunks: list[dict]) -> str:
    context_parts = []

    for i, chunk in enumerate(chunks, start=1):
        file_name = chunk['file_path'].split('/')[-1]
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
        return {
            "answer": "No relevant code chunks found for this question.",
            "chunks_used": 0
        }

    context = format_context(chunks)

    prompt = f"""Question: {question}

Relevant code from the codebase:

{context}

Answer the question based on the code above. Cite specific files and line numbers."""

    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.2,
        ),
        contents=prompt
    )

    return {
        "answer": response.text,
        "chunks_used": len(chunks)
    }