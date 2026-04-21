"""
llm_client.py  —  minimal LLM wrapper.
ai_10022200118 — Obour Adomako Tawiah — CS4241 (2026)

Provider-agnostic by design. We expose a single `chat(system, user)` call.
Under the hood we use OpenRouter — an aggregator exposing many
open-weight models via an OpenAI-compatible REST API.

Resilience: OpenRouter's free-tier endpoints share an upstream rate pool,
so any single model may 429. We therefore keep a fallback list and try
each model in sequence until one accepts the request.
"""
from __future__ import annotations

import time
from typing import List, Optional

from .config import settings
from .logger_config import get_logger, log_stage, Timer

log = get_logger("llm_client")


# Ordered list of free OpenRouter models to try. All are free tier;
# we fall through on 429 (rate-limited upstream) or 404 (model retired).
# The first entry is OpenRouter's auto-router — it picks any healthy
# free model for us, which makes it our best default.
FALLBACK_MODELS: List[str] = [
    "openrouter/free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "google/gemma-4-31b-it:free",
    "google/gemma-4-26b-a4b-it:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
]


class LLMClient:
    """Thin wrapper over OpenRouter's OpenAI-compatible Chat API."""

    def __init__(self, model: Optional[str] = None):
        self.primary_model = model or settings.llm_model
        self._client = None

    # ------------------------------------------------------------- lazy init
    def _client_lazy(self):
        if self._client is not None:
            return self._client
        if not settings.llm_api_key:
            raise RuntimeError(
                "LLM_API_KEY (OpenRouter) is not set. Put it in .env or the "
                "Streamlit Cloud 'Secrets' panel."
            )
        from openai import OpenAI
        self._client = OpenAI(
            api_key=settings.llm_api_key,
            base_url="https://openrouter.ai/api/v1",
        )
        return self._client

    # -------------------------------------------------- build fallback order
    def _models_to_try(self) -> List[str]:
        order = [self.primary_model]
        for m in FALLBACK_MODELS:
            if m not in order:
                order.append(m)
        return order

    # ----------------------------------------------------------------- chat
    def chat(self, system: str, user: str,
             temperature: float = settings.llm_temperature,
             max_tokens: int = 700) -> str:
        """Try each fallback model up to `retries` times with a short
        exponential back-off. The per-minute free-tier limit on
        OpenRouter resets after ~60 s, so two retries with 20s/40s
        waits usually succeed without user intervention."""
        client = self._client_lazy()
        last_err: Optional[Exception] = None
        retries = 3
        backoff = [0, 20, 40]   # seconds

        for attempt in range(retries):
            if backoff[attempt]:
                log_stage(log, "llm.retry",
                          f"Retry {attempt}/{retries-1} after "
                          f"{backoff[attempt]}s back-off")
                time.sleep(backoff[attempt])

            for model in self._models_to_try():
                try:
                    with Timer(log, "llm.call",
                               f"OpenRouter chat ({model})"):
                        resp = client.chat.completions.create(
                            model=model,
                            messages=[
                                {"role": "system", "content": system},
                                {"role": "user",   "content": user},
                            ],
                            temperature=temperature,
                            max_tokens=max_tokens,
                            top_p=1.0,
                            extra_headers={
                                "HTTP-Referer": "https://github.com/ai_10022200118",
                                "X-Title": "ai_10022200118 Academic City RAG",
                            },
                        )
                    text = (resp.choices[0].message.content or "") if resp.choices else ""
                    if text.strip():
                        self.last_served_by = model
                        log_stage(log, "llm.response",
                                  f"Received {len(text)} chars from {model}")
                        return text
                except Exception as e:
                    last_err = e
                    msg = str(e)
                    # 429 = rate-limited, 404 = model retired,
                    # 503 = upstream overloaded. All are recoverable.
                    if any(c in msg for c in ("429", "404", "503")) \
                            or "rate" in msg.lower():
                        log_stage(log, "llm.fallback",
                                  f"{model} unavailable; trying next.")
                        continue
                    raise

        raise RuntimeError(
            "All free OpenRouter models are currently rate-limited even "
            "after retries. Wait a minute and ask again, or add $10 credit "
            "to your OpenRouter account to raise the limit from 50/day to "
            "1000/day. Last error: " + str(last_err)[:200]
        )

    # ------------------------------------------------------ pure-LLM baseline
    def complete_no_rag(self, question: str,
                        temperature: float = 0.2,
                        max_tokens: int = 500) -> str:
        """Used by evaluator.py for the 'pure LLM, no retrieval' baseline."""
        system = ("You are an AI assistant. Answer the following question "
                  "using only your parametric knowledge. Do not mention that "
                  "you have no retrieval; just answer.")
        return self.chat(system, question,
                         temperature=temperature, max_tokens=max_tokens)
