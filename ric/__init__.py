"""
RIC Pipeline — ric/__init__.py
Router → Inference → Cache
"""
from .pipeline import RICPipeline
from .router import QueryRouter
from .inference import InferenceEngine
from .cache import CacheLayer

__all__ = ["RICPipeline", "QueryRouter", "InferenceEngine", "CacheLayer"]
__version__ = "1.0.0"
