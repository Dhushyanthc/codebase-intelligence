import os
import json
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
from google import genai
from google.genai import types
from langsmith import traceable

from app.vector_store import query_collection
from app.bm25_store import query_bm25
from app.generator import generate_answer

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

class AgentState(TypedDict):
    question: str
    repo_url: str
    repo_name: str
    tool_decision: str
    clarification_question: str
    retrieved_chunks: list
    final_answer: str
    sources: list


ROUTER_PROMPT = """You are a routing agent for a codebase intelligence system.

Given a user's question and the GitHub repository URL they are asking about, decide which tool to use:

1. "codebase_search" - Use when the question is about:
   - How specific code works (functions, classes, logic)
   - Implementation details in the repository
   - Code structure, architecture, or patterns used
   - Specific files, methods, or variables

2. "web_search" - Use when the question is about:
   - External libraries or dependencies used in the repo
   - Security vulnerabilities or CVEs affecting the repo
   - Best practices or comparisons with other approaches
   - Anything that requires knowledge beyond the repository code

3. "clarify" - Use when the question is:
   - Too vague to search effectively (e.g., "how does it work?")
   - Ambiguous about which part of the codebase it refers to
   - Missing context needed to give a useful answer

Respond with valid JSON only. No explanation, no markdown, just JSON:
{
  "tool": "codebase_search" | "web_search" | "clarify",
  "reason": "one sentence explaining why",
  "clarification_question": "only if tool is clarify, the question to ask the user, otherwise null"
}"""


@traceable(name="router_node")
def router_node(state: AgentState) -> AgentState:
    prompt = f"Question: {state['question']}\nRepository: {state['repo_url']}"

    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        config=types.GenerateContentConfig(
            system_instruction=ROUTER_PROMPT,
            temperature=0.1,
            response_mime_type="application/json",
        ),
        contents=prompt
    )

    decision = json.loads(response.text)

    state['tool_decision'] = decision['tool']
    state['clarification_question'] = decision.get('clarification_question') or ''

    print(f"[router] Decision: {decision['tool']} | Reason: {decision['reason']}")
    return state

@traceable(name="codebase_search_node")
def codebase_search_node(state: AgentState) -> AgentState:
    repo_name = state['repo_name']
    question = state['question']

    # Embed the question
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=question,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
    )
    query_vector = result.embeddings[0].values

    # Dense retrieval
    dense_results = query_collection(repo_name, query_vector, n_results=10)

    # BM25 retrieval
    sparse_results = query_bm25(repo_name, question, n_results=10)

    # RRF fusion
    K = 60
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
            'rrf_score': 1 / (K + rank)
        }

    for rank, r in enumerate(sparse_results, start=1):
        key = f"{r['file_path']}:{r['start_line']}"
        bm25_rrf = 1 / (K + rank)
        if key in rrf_scores:
            rrf_scores[key]['rrf_score'] += bm25_rrf
        else:
            rrf_scores[key] = {**r, 'rrf_score': bm25_rrf}

    ranked = sorted(rrf_scores.values(), key=lambda x: x['rrf_score'], reverse=True)
    state['retrieved_chunks'] = ranked[:5]

    print(f"[codebase_search] Retrieved {len(state['retrieved_chunks'])} chunks")
    return state


@traceable(name="generate_node")
def generate_node(state: AgentState) -> AgentState:
    result = generate_answer(state['question'], state['retrieved_chunks'])
    state['final_answer'] = result['answer']
    state['sources'] = [
        {
            'file_path': c['file_path'],
            'start_line': c['start_line'],
            'end_line': c['end_line'],
        }
        for c in state['retrieved_chunks']
    ]
    print(f"[generate] Answer generated")
    return state


@traceable(name="clarify_node")
def clarify_node(state: AgentState) -> AgentState:
    state['final_answer'] = state['clarification_question']
    state['sources'] = []
    print(f"[clarify] Asking: {state['clarification_question']}")
    return state


def route_decision(state: AgentState) -> Literal["codebase_search", "clarify"]:
    if state['tool_decision'] == 'clarify':
        return 'clarify'
    return 'codebase_search'


def build_agent():
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("router", router_node)
    graph.add_node("codebase_search", codebase_search_node)
    graph.add_node("generate", generate_node)
    graph.add_node("clarify", clarify_node)

    # Add edges
    graph.add_edge(START, "router")
    graph.add_conditional_edges("router", route_decision)
    graph.add_edge("codebase_search", "generate")
    graph.add_edge("generate", END)
    graph.add_edge("clarify", END)

    return graph.compile()


agent = build_agent()