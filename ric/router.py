"""
RIC Pipeline — ric/router.py
Decides whether a query goes to LOCAL (Mac Mini Gemma) or CLOUD (OmniRoute).

Routing modes:
  heuristic    → rule-based on word count + keyword matching
  hybrid       → heuristic + confidence scoring
  always_local → force all queries to local
  always_cloud → force all queries to cloud
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple

from .utils import load_config, get_logger

logger = get_logger("ric.router")


# ─────────────────────────────────────────────
# Data Classes
# ─────────────────────────────────────────────

class Destination(str, Enum):
    LOCAL = "local"
    CLOUD = "cloud"


@dataclass
class RoutingDecision:
    destination: Destination
    reason: str
    confidence: float          # 0.0 → 1.0 (1.0 = very certain)
    estimated_tokens: int = 0
    word_count: int = 0
    matched_keywords: List[str] = field(default_factory=list)


# ─────────────────────────────────────────────
# Router
# ─────────────────────────────────────────────

class QueryRouter:
    """
    Routes incoming queries to local or cloud inference.

    Example:
        router = QueryRouter()
        decision = router.route("What is the capital of France?")
        print(decision.destination)  # Destination.LOCAL
    """

    def __init__(self, config: dict = None):
        cfg = config or load_config()
        self.router_cfg = cfg["router"]
        self.mode = self.router_cfg.get("mode", "heuristic")
        self.threshold = self.router_cfg.get("complexity_threshold", 60)
        self.local_kw: List[str] = [
            kw.lower() for kw in self.router_cfg.get("local_keywords", [])
        ]
        self.cloud_kw: List[str] = [
            kw.lower() for kw in self.router_cfg.get("cloud_keywords", [])
        ]
        logger.info(f"QueryRouter initialized | mode={self.mode} | threshold={self.threshold} words")

    # ── Public API ────────────────────────────

    def route(self, query: str) -> RoutingDecision:
        """Main routing function. Returns a RoutingDecision."""
        query = query.strip()
        word_count = len(query.split())
        estimated_tokens = max(1, len(query) // 4)

        if self.mode == "always_local":
            return RoutingDecision(
                destination=Destination.LOCAL,
                reason="Forced local mode (config)",
                confidence=1.0,
                estimated_tokens=estimated_tokens,
                word_count=word_count,
            )

        if self.mode == "always_cloud":
            return RoutingDecision(
                destination=Destination.CLOUD,
                reason="Forced cloud mode (config)",
                confidence=1.0,
                estimated_tokens=estimated_tokens,
                word_count=word_count,
            )

        # Heuristic and hybrid both use _heuristic_route
        destination, reason, confidence, matched_kw = self._heuristic_route(
            query, word_count
        )

        decision = RoutingDecision(
            destination=destination,
            reason=reason,
            confidence=confidence,
            estimated_tokens=estimated_tokens,
            word_count=word_count,
            matched_keywords=matched_kw,
        )

        logger.info(
            f"ROUTE → {decision.destination.value.upper():5s} | "
            f"words={word_count:4d} | conf={confidence:.2f} | reason='{reason}' | "
            f"query='{query[:60]}{'...' if len(query) > 60 else ''}'"
        )
        return decision

    # ── Internal ──────────────────────────────

    def _heuristic_route(
        self, query: str, word_count: int
    ) -> Tuple[Destination, str, float, List[str]]:
        """
        Rule-based routing:
        1. Check cloud keywords (high priority) → cloud
        2. Check local keywords → local
        3. Check word count threshold → local if short, cloud if long
        """
        q_lower = query.lower()
        matched_cloud = [kw for kw in self.cloud_kw if kw in q_lower]
        matched_local = [kw for kw in self.local_kw if kw in q_lower]

        # Cloud keywords take priority
        if matched_cloud:
            return (
                Destination.CLOUD,
                f"cloud keyword match: {matched_cloud}",
                0.90,
                matched_cloud,
            )

        # Local keywords
        if matched_local:
            return (
                Destination.LOCAL,
                f"local keyword match: {matched_local}",
                0.85,
                matched_local,
            )

        # Fall back to word count
        if word_count <= self.threshold:
            conf = 1.0 - (word_count / self.threshold) * 0.3  # longer = less confident
            return (
                Destination.LOCAL,
                f"short query ({word_count} words ≤ {self.threshold} threshold)",
                round(conf, 2),
                [],
            )
        else:
            conf = min(1.0, 0.6 + (word_count - self.threshold) / 200)
            return (
                Destination.CLOUD,
                f"long query ({word_count} words > {self.threshold} threshold)",
                round(conf, 2),
                [],
            )
