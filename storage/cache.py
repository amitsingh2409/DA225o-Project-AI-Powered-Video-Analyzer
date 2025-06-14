import time
from typing import Dict, Any, Optional, Callable
import logging
import json
import hashlib
from functools import wraps

from ..config import CACHE_EXPIRY

logger = logging.getLogger(__name__)


class Cache:
    """Simple in-memory cache with expiration."""

    def __init__(self, expiry_time: int = CACHE_EXPIRY):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.expiry_time = expiry_time

    def _generate_key(self, *args, **kwargs) -> str:
        """Generate a cache key from args and kwargs."""
        key_parts = [str(arg) for arg in args]
        key_parts.extend([f"{k}:{v}" for k, v in sorted(kwargs.items())])
        key_string = json.dumps(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache if it exists and hasn't expired."""
        if key in self.cache:
            item = self.cache[key]
            if item["expiry"] > time.time():
                logger.debug(f"Cache hit for key: {key}")
                return item["value"]
            else:
                # Remove expired item
                logger.debug(f"Cache expired for key: {key}")
                del self.cache[key]
        return None

    def set(self, key: str, value: Any) -> None:
        """Set value in cache with expiration."""
        self.cache[key] = {"value": value, "expiry": time.time() + self.expiry_time}
        logger.debug(f"Added to cache: {key}")

    def clear(self) -> None:
        """Clear all items from cache."""
        self.cache.clear()
        logger.info("Cache cleared")

    def remove_expired(self) -> int:
        """Remove all expired items from cache. Returns count of removed items."""
        now = time.time()
        expired_keys = [k for k, v in self.cache.items() if v["expiry"] <= now]
        for key in expired_keys:
            del self.cache[key]

        if expired_keys:
            logger.debug(f"Removed {len(expired_keys)} expired items from cache")

        return len(expired_keys)


# Create a global cache instance
global_cache = Cache()


def cached(func: Callable) -> Callable:
    """Decorator to cache function results."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Skip cache if explicitly requested
        skip_cache = kwargs.pop("skip_cache", False)

        if skip_cache:
            return func(*args, **kwargs)

        # Generate cache key
        cache_key = global_cache._generate_key(func.__name__, *args, **kwargs)

        # Try to get from cache
        result = global_cache.get(cache_key)
        if result is not None:
            return result

        # Execute function and store result
        result = func(*args, **kwargs)
        global_cache.set(cache_key, result)
        return result

    return wrapper
