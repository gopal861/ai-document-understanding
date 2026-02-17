# app/llm/multi_model_client.py

import os
import logging
import time
from typing import Optional, Dict

from openai import OpenAI
import google.generativeai as genai
from transformers import pipeline

from app.prompts.system_prompts import (
    DOCUMENT_QA_SYSTEM_PROMPT,
    LOCAL_MODEL_SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)


class MultiModelLLMClient:
    """
    Production-grade multi-provider LLM client.

    Fallback order (STRICT):

    1. OpenAI (primary)
    2. Gemini (secondary)
    3. Local FLAN-T5 (failsafe)

    Guarantees:
    • Never crashes system
    • Fully observable
    • Provider latency tracking
    • Failure-safe fallback
    """

    def __init__(self):

        self.openai: Optional[OpenAI] = None
        self.gemini_model = None
        self.local_model = None

        self.openai_available = False
        self.gemini_available = False
        self.local_available = False

        self._init_openai()
        self._init_gemini()
        self._init_local()

        logger.info(
            "LLM initialization complete",
            extra={
                "openai_available": self.openai_available,
                "gemini_available": self.gemini_available,
                "local_available": self.local_available,
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

            # Test call to validate key
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

    def _init_local(self):

        try:

            self.local_model = pipeline(
                "text2text-generation",
                model="google/flan-t5-base",
                device=-1,
            )

            self.local_available = True

            logger.info("Local model initialized successfully")

        except Exception as e:

            logger.error(
                "Local model initialization failed",
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
                "local_available": self.local_available,
                "prompt_length": len(prompt),
            },
        )

        # OpenAI
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

        # Gemini
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

        # Local fallback
        if self.local_available:

            logger.info("Using local fallback model")

            return self._timed_call(
                provider="local",
                fn=self._generate_local,
                prompt=prompt,
            )

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

        text = response.choices[0].message.content

        return text.strip()

    def _generate_gemini(self, prompt: str) -> str:

        response = self.gemini_model.generate_content(
            f"{DOCUMENT_QA_SYSTEM_PROMPT}\n\n{prompt}"
        )

        if not response or not response.text:
            raise RuntimeError("Gemini returned empty response")

        return response.text.strip()

    def _generate_local(self, prompt: str) -> str:

        simplified_prompt = (
            f"{LOCAL_MODEL_SYSTEM_PROMPT}\n\n{prompt[:1000]}"
        )

        result = self.local_model(
            simplified_prompt,
            max_length=512,
            do_sample=False,
        )

        return result[0]["generated_text"].strip()

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
            "local_available": self.local_available,
        }

