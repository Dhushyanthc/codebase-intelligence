import os
import logging
from git import Repo, GitCommandError
from urllib.parse import urlparse
from app.config import CLONE_BASE_DIR

logger = logging.getLogger("codebase-intelligence")

SUPPORTED_EXTENSIONS = {
    '.py', '.js', '.ts', '.go', '.java', '.cpp', '.c', '.h', '.cs', '.rb',
    '.rs', '.php', '.swift', '.kt', '.scala', '.jsx', '.tsx', '.vue', '.mod', '.sum'
}

IGNORED_DIRS = {
    'node_modules', 'vendor', '__pycache__', '.git',
    'dist', 'build', '.next', 'venv', 'env', 'target'
}


def is_valid_git_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
        return (
            parsed.scheme in ('http', 'https', 'git') and
            parsed.netloc == 'github.com' and
            len(parsed.path.strip('/').split('/')) >= 2
        )
    except Exception:
        return False


def clone_repository(repo_url: str) -> str:
    if not is_valid_git_url(repo_url):
        raise ValueError("Invalid GitHub repository URL")

    parsed = urlparse(repo_url)
    repo_name = parsed.path.strip('/').replace('/', '_')
    clone_path = os.path.join(CLONE_BASE_DIR, repo_name)

    os.makedirs(CLONE_BASE_DIR, exist_ok=True)

    if os.path.exists(clone_path):
        logger.info(f"Repository already cloned at {clone_path}")
        return clone_path

    logger.info(f"Cloning {repo_url} to {clone_path}")
    try:
        Repo.clone_from(repo_url, clone_path, depth=1)
    except GitCommandError as e:
        raise RuntimeError(f"Failed to clone repository: {e}")

    return clone_path


def filter_repo_files(repo_path: str) -> list:
    code_files = []
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in IGNORED_DIRS]
        for file in files:
            _, ext = os.path.splitext(file)
            if ext in SUPPORTED_EXTENSIONS:
                code_files.append(os.path.join(root, file))
    return code_files
