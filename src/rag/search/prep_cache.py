import hashlib
import threading
import time
from collections import OrderedDict
from typing import Optional, Tuple

from src.constants.app_constant import PREP_CACHE_SIZE, PREP_CACHE_TTL_SECONDS

PreparedTuple = Tuple[Optional[str], Optional[str], Optional[str]]

_cache: "OrderedDict[str, Tuple[float, PreparedTuple]]" = OrderedDict()
_lock = threading.Lock()


def _key(user_input: str) -> str:
    return hashlib.sha1(user_input.strip().encode("utf-8")).hexdigest()


def get(user_input: str) -> Optional[PreparedTuple]:
    key = _key(user_input)
    now = time.monotonic()
    with _lock:
        entry = _cache.get(key)
        if entry is None:
            return None
        ts, value = entry
        if now - ts > PREP_CACHE_TTL_SECONDS:
            _cache.pop(key, None)
            return None
        _cache.move_to_end(key)
        return value


def put(user_input: str, value: PreparedTuple) -> None:
    key = _key(user_input)
    now = time.monotonic()
    with _lock:
        _cache[key] = (now, value)
        _cache.move_to_end(key)
        while len(_cache) > PREP_CACHE_SIZE:
            _cache.popitem(last=False)


def clear() -> None:
    with _lock:
        _cache.clear()
