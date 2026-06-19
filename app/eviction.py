import os
import stat
import json
import shutil
import logging
from datetime import datetime, timedelta
from app.config import BM25_PATH, CLONE_BASE_DIR, REPO_TTL_DAYS
from app.vector_store import delete_collection
from app.bm25_store import invalidate_bm25_cache

logger = logging.getLogger("codebase-intelligence")

METADATA_FILE = os.path.join(BM25_PATH, "_repo_metadata.json")


def _load_metadata() -> dict:
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, 'r') as f:
            return json.load(f)
    return {}


def _save_metadata(metadata: dict):
    os.makedirs(BM25_PATH, exist_ok=True)
    with open(METADATA_FILE, 'w') as f:
        json.dump(metadata, f)


def record_index_time(repo_name: str):
    metadata = _load_metadata()
    metadata[repo_name] = {"indexed_at": datetime.utcnow().isoformat()}
    _save_metadata(metadata)


def cleanup_stale_repos():
    metadata = _load_metadata()
    cutoff = datetime.utcnow() - timedelta(days=REPO_TTL_DAYS)
    stale_repos = []

    for repo_name, info in metadata.items():
        indexed_at = datetime.fromisoformat(info["indexed_at"])
        if indexed_at < cutoff:
            stale_repos.append(repo_name)

    for repo_name in stale_repos:
        logger.info(f"Evicting stale repo: {repo_name}")
        delete_collection(repo_name)
        bm25_file = os.path.join(BM25_PATH, f"{repo_name}.json")
        if os.path.exists(bm25_file):
            os.remove(bm25_file)
        invalidate_bm25_cache(repo_name)
        clone_dir = os.path.join(CLONE_BASE_DIR, repo_name)
        if os.path.exists(clone_dir):
            shutil.rmtree(clone_dir, onexc=_force_remove_readonly)
        del metadata[repo_name]

    _save_metadata(metadata)
    return stale_repos


def _force_remove_readonly(func, path, exc):
    os.chmod(path, stat.S_IWRITE)
    func(path)


def cleanup_clone_dir(repo_name: str):
    clone_dir = os.path.join(CLONE_BASE_DIR, repo_name)
    if os.path.exists(clone_dir):
        shutil.rmtree(clone_dir, onexc=_force_remove_readonly)
        logger.info(f"Cleaned up clone dir: {clone_dir}")
