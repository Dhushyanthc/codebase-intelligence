import re
import logging

logger = logging.getLogger("codebase-intelligence")

# Questions with these patterns are almost always codebase queries
CODE_SIGNAL_PATTERNS = [
    # Direct code references
    r'\b(function|method|class|module|endpoint|route|handler|middleware)\b',
    r'\b(implement|implementation|architecture|logic|algorithm|pipeline)\b',
    r'\b(api|database|schema|model|config|controller|service|repository)\b',
    r'\b(error handling|authentication|validation|parsing|routing)\b',
    # File references
    r'\.\w{1,4}\b',  # .py, .js, .go, .ts etc.
    r'\b\w+\.(py|js|ts|go|java|rs|cpp)\b',  # specific_file.py
    # Explicit code questions
    r'\bhow\s+(does|do|is|are)\s+\w+.*(work|implemented|structured|organized|handled|processed)\b',
    r'\bwhat\s+(does|do|is|are)\s+\w+.*(do|return|accept|handle|use|for)\b',
    r'\b(explain|describe|show|trace|walk\s*through)\b',
    r'\b(where|which)\s+(is|are|does)\b',
    # Code concepts
    r'\b(import|require|dependency|dependencies|package|library)\b',
    r'\b(variable|parameter|argument|return\s*type|type|interface|struct)\b',
    r'\b(loop|condition|branch|switch|if.else|try.catch|exception)\b',
    r'\b(async|await|promise|callback|goroutine|thread|worker)\b',
    r'\b(test|spec|mock|fixture)\b',
    r'\b(webhook|event|trigger|listener|subscriber|queue)\b',
]

# Questions matching these are out of scope for codebase search
OUT_OF_SCOPE_PATTERNS = [
    r'\bCVEs?\b|\bvulnerabilit(y|ies)\b|\bsecurity\s*(audit|scan|advisory|advisories)\b|\bpenetration\s*test(ing)?\b',
    r'\b(pricing|cost|billing|subscription|licen[sc](e|ing))\b',
    r'\b(install(ation)?|setup|deploy(ment)?)\s+(on|to|in|guide|instructions|steps)\b',
    r'\b(compare|vs\.?|versus|alternative|competitor)s?\b',
    r'\b(roadmap|feature\s*request|when\s*will|planned|upcoming)\b',
    r'\b(hire|hiring|job|career|interview|salary|compensation)\b',
    r'\b(bug\s*bounty|exploit(s|ation)?|attack\s*(vector|surface))\b',
]

# Too vague to route without LLM help
MIN_MEANINGFUL_WORDS = 3

# Pronoun-only subjects with no specific referent.
# "this" is intentionally excluded: "is this project X?" has a real subject and
# should fall through to uncertain rather than being treated as vague.
VAGUE_SUBJECT_PATTERN = (
    r'^(how|what|why|can|could|does|is|are|will|would)\s+'
    r'(does|do|is|are|can|could|would|will|about)?\s*'
    r'(it|that|these|those|the\s+project|the\s+repo)\s'
)


def pre_route(question: str, repo_name: str) -> dict:
    """
    Classify a question without an LLM call.

    Returns:
        {
            "decision": "codebase_search" | "clarify" | "uncertain",
            "reason": str,  # for logging/tracing
            "clarification_question": str  # only if decision is "clarify"
        }
    """
    q_lower = question.lower().strip()
    words = q_lower.split()

    # 1. Too short / vague
    if len(words) < MIN_MEANINGFUL_WORDS:
        logger.info(f"[pre-router] CLARIFY — too short ({len(words)} words): '{question}'")
        return {
            "decision": "clarify",
            "reason": "question_too_short",
            "clarification_question": (
                f"Could you be more specific about what you'd like to know about the "
                f"'{repo_name}' repository? For example, ask about a specific function, "
                f"module, or how a particular feature is implemented."
            )
        }

    # 1.5. Vague subject check — pronouns with no specific referent
    if re.search(VAGUE_SUBJECT_PATTERN, q_lower) and len(words) <= 6:
        logger.info(f"[pre-router] CLARIFY — vague subject with no specific referent: '{question}'")
        return {
            "decision": "clarify",
            "reason": "vague_subject_pronoun",
            "clarification_question": (
                f"Could you be more specific? For example, ask about a specific function, "
                f"endpoint, or feature in the '{repo_name}' repository."
            )
        }

    # 2 & 3. Evaluate out-of-scope and code signals independently so that
    # conflicting cases (e.g. "vulnerabilities in the auth module") can be
    # detected and deferred to the LLM router rather than making a wrong call.

    scope_match = None
    for pattern in OUT_OF_SCOPE_PATTERNS:
        # IGNORECASE required: q_lower is lowercased but patterns like \bCVEs?\b
        # use uppercase, so without this flag "cves" would never match.
        if re.search(pattern, q_lower, re.IGNORECASE):
            scope_match = pattern
            break

    code_signal_count = 0
    matched_patterns = []
    for pattern in CODE_SIGNAL_PATTERNS:
        if re.search(pattern, q_lower):
            code_signal_count += 1
            matched_patterns.append(pattern)
        if code_signal_count >= 2:
            break

    strong_code = code_signal_count >= 2

    # Conflict: strong code signals AND out-of-scope term present → let LLM decide.
    # A single weak code signal (e.g. "dependencies") is not enough to override a
    # clear out-of-scope match.
    if scope_match and strong_code:
        logger.info(f"[pre-router] UNCERTAIN — conflicting signals (code + out-of-scope): '{question}'")
        return {
            "decision": "uncertain",
            "reason": "conflicting_code_and_scope_signals",
            "clarification_question": ""
        }

    if scope_match:
        logger.info(f"[pre-router] CLARIFY — out of scope (matched: {scope_match}): '{question}'")
        return {
            "decision": "clarify",
            "reason": f"out_of_scope:{scope_match}",
            "clarification_question": (
                "This question is outside the scope of codebase search. "
                "I can only answer questions about the code, architecture, "
                "and implementation details in the repository."
            )
        }

    if strong_code:
        logger.info(f"[pre-router] CODEBASE_SEARCH — strong signal ({code_signal_count} matches): '{question}'")
        return {
            "decision": "codebase_search",
            "reason": f"strong_code_signals:{code_signal_count}",
            "clarification_question": ""
        }

    if code_signal_count == 1 and len(words) >= 5:
        # One signal + reasonably specific question = likely code
        logger.info(f"[pre-router] CODEBASE_SEARCH — single signal + specific: '{question}'")
        return {
            "decision": "codebase_search",
            "reason": "single_code_signal_with_specificity",
            "clarification_question": ""
        }

    # 4. Can't decide — let the LLM router handle it
    logger.info(f"[pre-router] UNCERTAIN — falling through to LLM router: '{question}'")
    return {
        "decision": "uncertain",
        "reason": "no_clear_signals",
        "clarification_question": ""
    }
