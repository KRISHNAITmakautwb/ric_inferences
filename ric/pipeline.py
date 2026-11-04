"""
RIC Pipeline — ric/pipeline.py
The main orchestrator: Router → Inference → Cache

This is the single entry point for all inference requests.
It ties together:
  1. CacheLayer  → check if we've seen this before
  2. QueryRouter → decide local vs cloud
  3. InferenceEngine → run the actual LLM call
  4. CacheLayer  → store the result for next time
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Dict, Any

from .router import QueryRouter, Destination
from .cache import CacheLayer
from .inference import InferenceEngine, InferenceResult
from .utils import load_config, get_logger

logger = get_logger("ric.pipeline")


# ─────────────────────────────────────────────
# Data Classes
# ─────────────────────────────────────────────

@dataclass
class PipelineResult:
    query: str
    response: str
    destination: str           # "local" | "cloud" | "cache"
    routing_reason: str
    routing_confidence: float
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    model: str
    from_cache: bool
    success: bool
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "response": self.response,
            "destination": self.destination,
            "routing_reason": self.routing_reason,
            "routing_confidence": self.routing_confidence,
            "latency_ms": round(self.latency_ms, 2),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "model": self.model,
            "from_cache": self.from_cache,
            "success": self.success,
            "error": self.error,
        }


# ─────────────────────────────────────────────
# RIC Pipeline
# ─────────────────────────────────────────────

class RICPipeline:
    """
    Main RIC (Router → Inference → Cache) Pipeline.

    Usage:
        pipeline = RICPipeline()
        await pipeline.init()
        result = await pipeline.run("Explain Transformer architecture")
        print(result.response)
        print(result.destination)    # "local" or "cloud" or "cache"
        print(result.latency_ms)
        await pipeline.close()
    """

    def __init__(self, config_path: str = None):
        self.config = load_config(config_path)
        self.router = QueryRouter(self.config)
        self.engine = InferenceEngine(self.config)
        self.cache = CacheLayer(self.config)
        self._initialized = False

    async def init(self):
        """Must be called before first use."""
        await self.cache.init()
        self._initialized = True
        logger.info("RICPipeline fully initialized and ready.")

    async def close(self):
        """Clean up resources."""
        await self.cache.close()
        logger.info("RICPipeline closed.")

    # ── Main Entry Point ──────────────────────

    async def run(self, query: str, system: str = None, skip_cache: bool = False) -> PipelineResult:
        """
        Process a query through the full RIC pipeline:
          1. Check cache
          2. Route (local / cloud)
          3. Run inference
          4. Store in cache
          5. Return PipelineResult

        Args:
            query:       The user's question or prompt
            system:      Optional system prompt
            skip_cache:  Force fresh inference even if cached

        Returns:
            PipelineResult with full metadata
        """
        if not self._initialized:
            await self.init()

        t_pipeline_start = time.perf_counter()
        query = query.strip()

        # ── Step 1: Cache Check ───────────────
        if not skip_cache:
            cached = await self.cache.get(query)
            if cached:
                total_ms = (time.perf_counter() - t_pipeline_start) * 1000
                logger.info(f"PIPELINE CACHE HIT | latency={total_ms:.1f}ms | query='{query[:50]}'")
                return PipelineResult(
                    query=query,
                    response=cached["response"],
                    destination="cache",
                    routing_reason="Served from cache",
                    routing_confidence=1.0,
                    latency_ms=total_ms,
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                    model=f"cached ({cached['destination']})",
                    from_cache=True,
                    success=True,
                )

        # ── Step 2: Router ────────────────────
        decision = self.router.route(query)

        # ── Step 3: Inference ─────────────────
        if decision.destination == Destination.LOCAL:
            result: InferenceResult = await self.engine.run_local(query, system)
            # Auto-fallback to cloud if local fails and config allows it
            if not result.success and self.config["local"].get("fallback_to_cloud", True):
                logger.warning(
                    f"Local inference failed ({result.error}). "
                    f"Falling back to cloud automatically."
                )
                result = await self.engine.run_cloud(query, system)
                decision.reason += " [local failed → cloud fallback]"
        else:
            result: InferenceResult = await self.engine.run_cloud(query, system)
            # Auto-fallback to local if cloud fails (OmniRoute unreachable)
            if not result.success:
                logger.warning(
                    f"Cloud inference failed ({result.error}). "
                    f"Falling back to local automatically."
                )
                result = await self.engine.run_local(query, system)
                decision.reason += " [cloud failed → local fallback]"

        # ── Step 4: Cache Result ──────────────
        if result.success and not skip_cache:
            await self.cache.set(
                query=query,
                response=result.response,
                destination=result.destination,
                tokens=result.total_tokens,
                latency_ms=result.latency_ms,
            )

        # ── Step 5: Build PipelineResult ──────
        total_ms = (time.perf_counter() - t_pipeline_start) * 1000
        pipeline_result = PipelineResult(
            query=query,
            response=result.response,
            destination=result.destination,
            routing_reason=decision.reason,
            routing_confidence=decision.confidence,
            latency_ms=total_ms,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            total_tokens=result.total_tokens,
            model=result.model,
            from_cache=False,
            success=result.success,
            error=result.error,
        )

        logger.info(
            f"PIPELINE DONE | dest={result.destination} | "
            f"latency={total_ms:.0f}ms | tokens={result.total_tokens} | "
            f"success={result.success}"
        )
        return pipeline_result

    # ── Utility Methods ───────────────────────

    async def get_stats(self) -> Dict[str, Any]:
        """Return cache and pipeline statistics."""
        cache_stats = await self.cache.stats_summary()
        return {
            "cache": cache_stats,
            "router_mode": self.router.mode,
            "local_model": self.config["local"]["model"],
            "cloud_model": self.config["cloud"]["model"],
        }

    async def clear_cache(self):
        """Clear all cached responses."""
        await self.cache.clear()
