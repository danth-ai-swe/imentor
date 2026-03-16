import pickle
from functools import lru_cache
from typing import Any, Optional

import redis

from src.config.app_config import get_app_config, _SingletonMeta

app_config = get_app_config()


class RedisCache(metaclass=_SingletonMeta):
    def __init__(self):
        self.pool = redis.ConnectionPool(
            host=app_config.REDIS_HOST,
            port=app_config.REDIS_PORT,
            db=app_config.REDIS_DB,
            decode_responses=False,
            max_connections=app_config.REDIS_MAX_CONNECTIONS,
            password=app_config.REDIS_PASSWORD
        )
        self.client = redis.Redis(connection_pool=self.pool)

    def set(self, key: str, value: Any, ttl: int = app_config.REDIS_DEFAULT_TTL) -> bool:
        binary_data = pickle.dumps(value)
        return self.client.setex(key, ttl, binary_data)

    def get(self, key: str) -> Optional[Any]:
        data = self.client.get(key)
        if data:
            return pickle.loads(data)
        return None

    def delete(self, key: str):
        return self.client.delete(key)


@lru_cache()
def get_redis_cache() -> RedisCache:
    return RedisCache()
