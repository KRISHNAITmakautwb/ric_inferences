"""
RIC Pipeline — ric/utils.py
Shared utilities: config loader, logger, timing
"""
import yaml
import logging
import time
import os
from pathlib import Path
from functools import wraps
from typing import Any, Dict

# ─────────────────────────────────────────────
# Config Loader
# ─────────────────────────────────────────────

_config_cache: Dict[str, Any] = {}

def load_config(path: str = None) -> Dict[str, Any]:
    """Load YAML config with caching."""
    global _config_cache
    if _config_cache:
        return _config_cache

    if path is None:
        # Search for config.yaml relative to project root
        candidates = [
            Path(__file__).parent.parent / "config.yaml",
            Path("config.yaml"),
        ]
        for c in candidates:
            if c.exists():
                path = str(c)
                break
        else:
            raise FileNotFoundError("config.yaml not found. Please specify path.")

    with open(path, "r") as f:
        _config_cache = yaml.safe_load(f)

    # Secrets come from the environment, never from the committed config file.
    cloud = _config_cache.setdefault("cloud", {})
    cloud["api_key"] = os.environ.get("OMNIROUTE_API_KEY", cloud.get("api_key") or "")
    if os.environ.get("OMNIROUTE_BASE_URL"):
        cloud["base_url"] = os.environ["OMNIROUTE_BASE_URL"]

    return _config_cache


# ─────────────────────────────────────────────
# Logger
# ─────────────────────────────────────────────

def get_logger(name: str = "ric") -> logging.Logger:
    """Get a pre-configured logger."""
    config = load_config()
    log_cfg = config.get("logging", {})
    level = getattr(logging, log_cfg.get("level", "INFO").upper(), logging.INFO)
    log_file = log_cfg.get("log_file", "./logs/ric.log")

    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)

        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Console handler
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)

        # File handler
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


# ─────────────────────────────────────────────
# Timing Decorator
# ─────────────────────────────────────────────

def timeit(func):
    """Decorator to measure execution time (sync and async)."""
    import asyncio

    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = await func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        return result, elapsed

    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        return result, elapsed

    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    return sync_wrapper


# ─────────────────────────────────────────────
# Token Estimation
# ─────────────────────────────────────────────

def estimate_tokens(text: str) -> int:
    """Rough token count estimation (~4 chars per token)."""
    return max(1, len(text) // 4)
