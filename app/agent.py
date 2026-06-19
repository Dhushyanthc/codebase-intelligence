import json
import logging
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from google import genai
from google.genai import types
from langsmith import traceable
from app.config import GEMINI_API_KEY, ROUTER_MODEL, ROUTER_TEMPERATURE
from app.retrieval import hybrid_search
from app.generator import generate_answer

logger = logging.getLogger("codebase-intelligence")
client = genai.Client(api_key=GEMINI_API_KEY)


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

2. "clarify" - Use when:
   - The question is too vague (e.g., "how does it work?") — ask the user to be more specific
   - The question requires external knowledge not in the codebase (CVEs, library docs, best practices comparisons) — set clarification_question to a message explaining this limitation and suggesting they consult external sources

Respond with valid JSON only. No explanation, no markdown, just JSON:
{
  "tool": "codebase_search" | "clarify",
  "reason": "one sentence explaining why",
  "clarification_question": "required when tool is clarify — either ask for more specifics or explain the limitation"
}"""


def _extract_response_text(response) -> str:
    text = response.text or ""
    if not text and response.candidates:
        for part in reversed(response.candidates[0].content.parts):
            if getattr(part, 'text', None):
                text = part.text
                break
    return text.strip()


@traceable(name="router_node")
def router_node(state: AgentState) -> AgentState:
    prompt = f"Question: {state['question']}\nRepository: {state['repo_url']}"
    response = client.models.generate_content(
        model=ROUTER_MODEL,
        config=types.GenerateContentConfig(
            system_instruction=ROUTER_PROMPT,
            temperature=ROUTER_TEMPERATURE,
            response_mime_type="application/json",
            max_output_tokens=256,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
        contents=prompt
    )
    text = _extract_response_text(response)
    if not text:
        raise ValueError("Router returned empty response")
    decision = json.loads(text)
    state['tool_decision'] = decision['tool']
    state['clarification_question'] = decision.get('clarification_question') or ''
    logger.info(f"Router decision: {decision['tool']} | Reason: {decision['reason']}")
    return state


@traceable(name="codebase_search_node")
def codebase_search_node(state: AgentState) -> AgentState:
    ranked = hybrid_search(state['repo_name'], state['question'], n_results=5)
    state['retrieved_chunks'] = ranked
    logger.info(f"Retrieved {len(ranked)} chunks for query")
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
    logger.info("Answer generated")
    return state


@traceable(name="clarify_node")
def clarify_node(state: AgentState) -> AgentState:
    message = state['clarification_question'] or (
        "This question requires information beyond what's available in the codebase. "
        "Please consult external sources, or ask a question about the code itself."
    )
    state['final_answer'] = message
    state['sources'] = []
    logger.info(f"Clarify response: {message}")
    return state


def route_decision(state: AgentState) -> Literal["codebase_search", "clarify"]:
    if state['tool_decision'] == 'clarify':
        return 'clarify'
    return 'codebase_search'


def build_agent():
    graph = StateGraph(AgentState)
    graph.add_node("router", router_node)
    graph.add_node("codebase_search", codebase_search_node)
    graph.add_node("generate", generate_node)
    graph.add_node("clarify", clarify_node)
    graph.add_edge(START, "router")
    graph.add_conditional_edges("router", route_decision)
    graph.add_edge("codebase_search", "generate")
    graph.add_edge("generate", END)
    graph.add_edge("clarify", END)
    return graph.compile()


agent = build_agent()
