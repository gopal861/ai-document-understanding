# app/llm/multi_model_client.py

import os
import logging
import time
from typing import Optional, Dict

from openai import OpenAI
import google.generativeai as genai

from app.prompts.system_prompts import (
    DOCUMENT_QA_SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)


class MultiModelLLMClient:
    """
    Production-grade multi-provider LLM client.

    Fallback order:

    1. OpenAI (primary)
    2. Gemini (secondary)

    Local model removed for deployment memory safety.

    Guarantees:
    • Never crashes system
    • Fully observable
    • Provider latency tracking
    • Deployment-safe memory usage
    """

    # ============================================================
    # INIT
    # ============================================================

    def __init__(self):

        self.openai: Optional[OpenAI] = None
        self.gemini_model = None

        self.openai_available = False
        self.gemini_available = False

        self._init_openai()
        self._init_gemini()

        logger.info(
            "LLM initialization complete",
            extra={
                "openai_available": self.openai_available,
                "gemini_available": self.gemini_available,
            },
        )

    # ============================================================
    # INITIALIZATION
    # ============================================================

    def _init_openai(self):

        try:

            key = os.getenv("OPENAI_API_KEY")

            if key and key.startswith("sk-"):

                self.openai = OpenAI(api_key=key)

                self.openai_available = True

                logger.info("OpenAI initialized successfully")

            else:

                logger.warning("OpenAI API key missing or invalid")

        except Exception as e:

            logger.error(
                "OpenAI initialization failed",
                extra={"error": str(e)},
            )

    def _init_gemini(self):

        try:

            key = os.getenv("GEMINI_API_KEY")

            if not key:

                logger.warning("Gemini API key missing")
                return

            genai.configure(api_key=key)

            self.gemini_model = genai.GenerativeModel(
                model_name="gemini-1.5-flash"
            )

            # Test call
            test = self.gemini_model.generate_content("ping")

            if test and test.text:

                self.gemini_available = True

                logger.info("Gemini initialized successfully")

            else:

                logger.warning("Gemini test call returned empty")

        except Exception as e:

            logger.error(
                "Gemini initialization failed",
                extra={"error": str(e)},
            )

    # ============================================================
    # PUBLIC API
    # ============================================================

    def generate(self, prompt: str) -> str:

        logger.info(
            "LLM request started",
            extra={
                "openai_available": self.openai_available,
                "gemini_available": self.gemini_available,
                "prompt_length": len(prompt),
            },
        )

        # ====================================================
        # OpenAI PRIMARY
        # ====================================================

        if self.openai_available:

            try:

                return self._timed_call(
                    provider="openai",
                    fn=self._generate_openai,
                    prompt=prompt,
                )

            except Exception as e:

                logger.warning(
                    "OpenAI failed",
                    extra={"error": str(e)},
                )

        # ====================================================
        # Gemini FALLBACK
        # ====================================================

        if self.gemini_available:

            try:

                return self._timed_call(
                    provider="gemini",
                    fn=self._generate_gemini,
                    prompt=prompt,
                )

            except Exception as e:

                logger.warning(
                    "Gemini failed",
                    extra={"error": str(e)},
                )

        # ====================================================
        # No providers available
        # ====================================================

        raise RuntimeError("No LLM backend available")

    # ============================================================
    # PROVIDERS
    # ============================================================

    def _generate_openai(self, prompt: str) -> str:

        response = self.openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": DOCUMENT_QA_SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0.1,
            max_tokens=500,
        )

        return response.choices[0].message.content.strip()

    def _generate_gemini(self, prompt: str) -> str:

        response = self.gemini_model.generate_content(
            f"{DOCUMENT_QA_SYSTEM_PROMPT}\n\n{prompt}"
        )

        if not response or not response.text:
            raise RuntimeError("Gemini returned empty response")

        return response.text.strip()

    # ============================================================
    # LATENCY OBSERVABILITY
    # ============================================================

    def _timed_call(self, provider: str, fn, prompt: str):

        start = time.time()

        result = fn(prompt)

        latency = time.time() - start

        logger.info(
            "LLM provider success",
            extra={
                "provider": provider,
                "latency_seconds": round(latency, 3),
            },
        )

        return result

    # ============================================================
    # STATUS
    # ============================================================

    def get_usage_stats(self) -> Dict:

        return {
            "openai_available": self.openai_available,
            "gemini_available": self.gemini_available,
        }

