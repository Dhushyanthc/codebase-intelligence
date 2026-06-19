import time


class TTLCache:
    def __init__(self, ttl_seconds=3600):
        self._cache = {}
        self._ttl = ttl_seconds

    def get(self, key: str):
        entry = self._cache.get(key)
        if entry and (time.time() - entry[1]) < self._ttl:
            return entry[0]
        if entry:
            del self._cache[key]
        return None

    def set(self, key: str, value):
        self._cache[key] = (value, time.time())

    def clear(self):
        self._cache.clear()


query_embedding_cache = TTLCache(ttl_seconds=3600)
