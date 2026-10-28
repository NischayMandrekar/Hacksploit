import logging, threading, time
from platform_app.config import settings
logger = logging.getLogger(__name__)

class RateLimiter:
    """Atomic Redis fixed-window limiter when configured; local per-process fallback otherwise."""
    def __init__(self):
        self._lock = threading.Lock(); self._local: dict[str, tuple[int, int]] = {}; self._redis = None
        if settings.redis_url:
            try:
                import redis
                self._redis = redis.Redis.from_url(settings.redis_url, decode_responses=True, socket_timeout=1)
                self._redis.ping()
            except Exception:
                logger.warning("redis_unavailable; using local per-process rate limits", exc_info=True)
                self._redis = None

    def _ensure_redis(self):
        if self._redis is None and settings.redis_url:
            try:
                import redis
                client = redis.Redis.from_url(settings.redis_url, decode_responses=True, socket_timeout=1)
                client.ping()
                self._redis = client
            except Exception:
                self._redis = None
        return self._redis

    def allowed(self, key_id: str) -> bool:
        now = int(time.time()); window = settings.rate_limit_window_seconds
        if settings.redis_url and not self._ensure_redis():
            logger.warning("redis_unavailable; rate limiting fails open until Redis recovers")
            return True
        if self._redis:
            try:
                redis_key = f"platform:rate:{key_id}:{now // window}"
                pipe = self._redis.pipeline(); pipe.incr(redis_key); pipe.expire(redis_key, window + 1); count, _ = pipe.execute()
                return int(count) <= settings.rate_limit_requests
            except Exception:
                logger.warning("redis_rate_limit_failed; fail-open for availability", exc_info=True)
                return True
        bucket = now // window
        with self._lock:
            previous_bucket, count = self._local.get(key_id, (bucket, 0))
            count = count + 1 if previous_bucket == bucket else 1
            self._local[key_id] = (bucket, count)
            return count <= settings.rate_limit_requests

rate_limiter = RateLimiter()
