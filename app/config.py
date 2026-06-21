import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
API_SECRET_KEY = os.getenv("API_SECRET_KEY")

# Paths
CLONE_BASE_DIR = os.path.abspath(os.getenv("CLONE_BASE_DIR", "./cloned_repos"))
CHROMA_PATH = os.path.abspath(os.getenv("CHROMA_PATH", "./chroma_db"))
BM25_PATH = os.path.abspath(os.getenv("BM25_PATH", "./bm25_db"))

# Embedding
EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_BATCH_SIZE = 5

# Chunking
MAX_LINES = 50
OVERLAP_LINES = 10

# Generation
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "gemini-2.5-flash")
GENERATION_MAX_TOKENS = 4096
GENERATION_TEMPERATURE = 0.2

# Router
ROUTER_MODEL = os.getenv("ROUTER_MODEL", "gemini-2.5-flash")
ROUTER_TEMPERATURE = 0.1

# Retrieval
RRF_K = 60
DEFAULT_N_RESULTS = 5
MAX_N_RESULTS = 20
MAX_QUESTION_LENGTH = 2000

# Rate Limits
INDEX_RATE_LIMIT = "5/hour"
QUERY_RATE_LIMIT = "30/minute"

# Eviction
REPO_TTL_DAYS = int(os.getenv("REPO_TTL_DAYS", "7"))
