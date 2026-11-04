"""
RIC Pipeline — ric/inference.py
Handles actual inference calls to:
  - LOCAL: Mac Mini Ollama + Gemma 3 12B
  - CLOUD: OmniRoute API (OpenAI-compatible)

Both adapters expose the same async interface:
    result = await engine.run_local(prompt)
    result = await engine.run_cloud(prompt)
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import httpx

from .utils import load_config, get_logger, estimate_tokens

logger = get_logger("ric.inference")


# ─────────────────────────────────────────────
# Data Classes
# ─────────────────────────────────────────────

@dataclass
class InferenceResult:
    response: str
    destination: str          # "local" or "cloud"
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    model: str
    success: bool
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "response": self.response,
            "destination": self.destination,
            "latency_ms": round(self.latency_ms, 2),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "model": self.model,
            "success": self.success,
            "error": self.error,
        }


# ─────────────────────────────────────────────
# Inference Engine
# ─────────────────────────────────────────────

class InferenceEngine:
    """
    Dual-backend inference engine.

    LOCAL  → Ollama HTTP API (Mac Mini M3 + Gemma 3 12B)
    CLOUD  → OmniRoute OpenAI-compatible API

    Usage:
        engine = InferenceEngine()
        result = await engine.run_local("What is photosynthesis?")
        result = await engine.run_cloud("Write a full research report on...")
    """

    def __init__(self, config: dict = None):
        cfg = config or load_config()
        self.local_cfg = cfg["local"]
        self.cloud_cfg = cfg["cloud"]

        self.local_url = (
            f"http://{self.local_cfg['host']}:{self.local_cfg['port']}/api/generate"
        )
        self.cloud_url = f"{self.cloud_cfg['base_url'].rstrip('/')}/chat/completions"

        logger.info(
            f"InferenceEngine ready | "
            f"local={self.local_cfg['host']}:{self.local_cfg['port']} model={self.local_cfg['model']} | "
            f"cloud={self.cloud_cfg['base_url']} model={self.cloud_cfg['model']}"
        )

    # ── Local (Ollama / Gemma) ─────────────────

    async def run_local(self, prompt: str, system: str = None) -> InferenceResult:
        """
        Call the local Ollama server running Gemma 3 on Mac Mini M3.
        """
        t_start = time.perf_counter()

        # Build full prompt (Ollama generate API)
        full_prompt = f"{system}\n\n{prompt}" if system else prompt

        payload = {
            "model": self.local_cfg["model"],
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "num_predict": 1024,
            },
        }

        try:
            async with httpx.AsyncClient(timeout=self.local_cfg.get("timeout", 60)) as client:
                resp = await client.post(self.local_url, json=payload)
                resp.raise_for_status()
                data = resp.json()

            latency_ms = (time.perf_counter() - t_start) * 1000
            response_text = data.get("response", "")

            # Ollama provides token counts
            prompt_tokens = data.get("prompt_eval_count", estimate_tokens(prompt))
            completion_tokens = data.get("eval_count", estimate_tokens(response_text))

            logger.info(
                f"LOCAL inference | latency={latency_ms:.0f}ms | "
                f"tokens={prompt_tokens}+{completion_tokens}={prompt_tokens + completion_tokens}"
            )

            return InferenceResult(
                response=response_text,
                destination="local",
                latency_ms=latency_ms,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                model=self.local_cfg["model"],
                success=True,
            )

        except httpx.ConnectError:
            logger.error("LOCAL inference FAILED: Cannot connect to Ollama. Is it running?")
            return self._error_result("local", "Ollama not reachable. Run: ollama serve", t_start)
        except Exception as e:
            logger.error(f"LOCAL inference FAILED: {e}")
            return self._error_result("local", str(e), t_start)

    # ── Cloud (OmniRoute) ──────────────────────

    async def run_cloud(self, prompt: str, system: str = None) -> InferenceResult:
        """
        Call OmniRoute (OpenAI-compatible) cloud API.
        """
        t_start = time.perf_counter()

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.cloud_cfg["model"],
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 2048,
        }

        headers = {
            "Authorization": f"Bearer {self.cloud_cfg['api_key']}",
            "Content-Type": "application/json",
        }

        try:
            async with httpx.AsyncClient(timeout=self.cloud_cfg.get("timeout", 120)) as client:
                resp = await client.post(self.cloud_url, json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()

            latency_ms = (time.perf_counter() - t_start) * 1000
            response_text = data["choices"][0]["message"]["content"]

            usage = data.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", estimate_tokens(prompt))
            completion_tokens = usage.get("completion_tokens", estimate_tokens(response_text))

            logger.info(
                f"CLOUD inference | latency={latency_ms:.0f}ms | "
                f"tokens={prompt_tokens}+{completion_tokens}={prompt_tokens + completion_tokens}"
            )

            return InferenceResult(
                response=response_text,
                destination="cloud",
                latency_ms=latency_ms,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                model=self.cloud_cfg["model"],
                success=True,
            )

        except httpx.ConnectError:
            logger.error(f"CLOUD inference FAILED: Cannot connect to OmniRoute at {self.cloud_url}")
            return self._error_result("cloud", f"OmniRoute not reachable at {self.cloud_url}", t_start)
        except httpx.HTTPStatusError as e:
            logger.error(f"CLOUD inference FAILED: HTTP {e.response.status_code} from {self.cloud_url}")
            return self._error_result("cloud", f"OmniRoute returned HTTP {e.response.status_code}", t_start)
        except httpx.TimeoutException:
            logger.error(f"CLOUD inference FAILED: Timeout after {self.cloud_cfg.get('timeout', 120)}s")
            return self._error_result("cloud", f"OmniRoute timed out after {self.cloud_cfg.get('timeout', 120)}s", t_start)
        except Exception as e:
            logger.error(f"CLOUD inference FAILED: {e}")
            return self._error_result("cloud", str(e), t_start)

    # ── Helpers ───────────────────────────────

    @staticmethod
    def _error_result(destination: str, error: str, t_start: float) -> InferenceResult:
        latency_ms = (time.perf_counter() - t_start) * 1000
        return InferenceResult(
            response=f"[ERROR] {error}",
            destination=destination,
            latency_ms=latency_ms,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            model="unknown",
            success=False,
            error=error,
        )
