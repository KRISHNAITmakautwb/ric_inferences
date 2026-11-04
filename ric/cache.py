"""
RIC Pipeline — ric/cache.py
SQLite-backed cache with Redis fallback support.

Key features:
  - Semantic-safe key hashing (MD5 of normalized query)
  - TTL-based expiration
  - Hit/miss statistics
  - Async interface (aiosqlite)
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any

import aiosqlite

from .utils import load_config, get_logger

logger = get_logger("ric.cache")


# ─────────────────────────────────────────────
# Data Classes
# ─────────────────────────────────────────────

@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    total_saved_tokens: int = 0
    total_saved_ms: float = 0.0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate_pct": round(self.hit_rate * 100, 2),
            "total_saved_tokens": self.total_saved_tokens,
            "total_saved_ms": round(self.total_saved_ms, 2),
        }


# ─────────────────────────────────────────────
# Cache Layer
# ─────────────────────────────────────────────

class CacheLayer:
    """
    Async SQLite cache for RIC inference responses.

    Usage:
        cache = CacheLayer()
        await cache.init()

        hit = await cache.get("What is AI?")
        if hit:
            print("CACHE HIT:", hit)
        else:
            result = run_inference(...)
            await cache.set("What is AI?", result, tokens=42, latency_ms=320.0)
    """

    _CREATE_TABLE = """
    CREATE TABLE IF NOT EXISTS ric_cache (
        cache_key   TEXT PRIMARY KEY,
        query       TEXT NOT NULL,
        response    TEXT NOT NULL,
        destination TEXT NOT NULL,
        tokens      INTEGER DEFAULT 0,
        latency_ms  REAL DEFAULT 0.0,
        created_at  REAL NOT NULL,
        expires_at  REAL NOT NULL,
        hit_count   INTEGER DEFAULT 0
    );
    """

    _CREATE_INDEX = """
    CREATE INDEX IF NOT EXISTS idx_expires ON ric_cache (expires_at);
    """

    def __init__(self, config: dict = None):
        cfg = config or load_config()
        cache_cfg = cfg["cache"]
        self.db_path = cache_cfg.get("db_path", "./cache_db/ric_cache.db")
        self.ttl_seconds = cache_cfg.get("ttl_hours", 24) * 3600
        self.max_entries = cache_cfg.get("max_entries", 10000)
        self.stats = CacheStats()
        self._db: Optional[aiosqlite.Connection] = None
        logger.info(f"CacheLayer configured | db={self.db_path} | ttl={self.ttl_seconds//3600}h")

    # ── Lifecycle ─────────────────────────────

    async def init(self):
        """Initialize the SQLite database."""
        import os
        os.makedirs(os.path.dirname(self.db_path) if os.path.dirname(self.db_path) else ".", exist_ok=True)
        self._db = await aiosqlite.connect(self.db_path)
        await self._db.execute(self._CREATE_TABLE)
        await self._db.execute(self._CREATE_INDEX)
        await self._db.commit()
        # Clean expired on startup
        await self._evict_expired()
        logger.info("CacheLayer initialized and ready.")

    async def close(self):
        """Close the database connection."""
        if self._db:
            await self._db.close()

    # ── Public API ────────────────────────────

    async def get(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a cached response. Returns dict with 'response', 'destination',
        'tokens', 'latency_ms', or None on miss.
        """
        key = self._hash(query)
        now = time.time()

        async with self._db.execute(
            "SELECT response, destination, tokens, latency_ms, expires_at "
            "FROM ric_cache WHERE cache_key = ?",
            (key,),
        ) as cursor:
            row = await cursor.fetchone()

        if row is None:
            self.stats.misses += 1
            return None

        response, destination, tokens, latency_ms, expires_at = row

        if now > expires_at:
            await self._delete(key)
            self.stats.misses += 1
            logger.debug(f"Cache EXPIRED for key={key[:8]}")
            return None

        # Update hit count
        await self._db.execute(
            "UPDATE ric_cache SET hit_count = hit_count + 1 WHERE cache_key = ?",
            (key,),
        )
        await self._db.commit()

        self.stats.hits += 1
        self.stats.total_saved_tokens += tokens
        self.stats.total_saved_ms += latency_ms
        logger.info(f"Cache HIT | key={key[:8]} | saved_tokens={tokens} | saved_ms={latency_ms:.1f}")

        return {
            "response": response,
            "destination": destination,
            "tokens": tokens,
            "latency_ms": latency_ms,
            "from_cache": True,
        }

    async def set(
        self,
        query: str,
        response: str,
        destination: str = "unknown",
        tokens: int = 0,
        latency_ms: float = 0.0,
    ):
        """Store a response in cache."""
        key = self._hash(query)
        now = time.time()
        expires_at = now + self.ttl_seconds

        await self._db.execute(
            """
            INSERT INTO ric_cache
                (cache_key, query, response, destination, tokens, latency_ms, created_at, expires_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(cache_key) DO UPDATE SET
                response=excluded.response,
                destination=excluded.destination,
                tokens=excluded.tokens,
                latency_ms=excluded.latency_ms,
                expires_at=excluded.expires_at
            """,
            (key, query[:500], response, destination, tokens, latency_ms, now, expires_at),
        )
        await self._db.commit()
        logger.debug(f"Cache SET | key={key[:8]} | tokens={tokens} | dest={destination}")

        # Evict if over limit
        await self._enforce_limit()

    async def stats_summary(self) -> Dict[str, Any]:
        """Get cache statistics including DB entry count."""
        async with self._db.execute("SELECT COUNT(*) FROM ric_cache") as cur:
            row = await cur.fetchone()
        count = row[0] if row else 0
        summary = self.stats.to_dict()
        summary["db_entries"] = count
        return summary

    async def clear(self):
        """Clear all cached entries."""
        await self._db.execute("DELETE FROM ric_cache")
        await self._db.commit()
        logger.warning("Cache CLEARED by user request.")

    # ── Internal ──────────────────────────────

    @staticmethod
    def _hash(query: str) -> str:
        """Normalize and hash query string."""
        normalized = " ".join(query.lower().strip().split())
        return hashlib.md5(normalized.encode()).hexdigest()

    async def _delete(self, key: str):
        await self._db.execute("DELETE FROM ric_cache WHERE cache_key = ?", (key,))
        await self._db.commit()

    async def _evict_expired(self):
        now = time.time()
        async with self._db.execute(
            "DELETE FROM ric_cache WHERE expires_at < ?", (now,)
        ) as cur:
            deleted = cur.rowcount
        await self._db.commit()
        if deleted:
            logger.info(f"Cache evicted {deleted} expired entries.")

    async def _enforce_limit(self):
        async with self._db.execute("SELECT COUNT(*) FROM ric_cache") as cur:
            row = await cur.fetchone()
        count = row[0] if row else 0
        if count > self.max_entries:
            # Delete oldest 10%
            to_delete = count - int(self.max_entries * 0.9)
            await self._db.execute(
                "DELETE FROM ric_cache WHERE cache_key IN "
                "(SELECT cache_key FROM ric_cache ORDER BY created_at ASC LIMIT ?)",
                (to_delete,),
            )
            await self._db.commit()
            logger.info(f"Cache LRU eviction: removed {to_delete} old entries.")
