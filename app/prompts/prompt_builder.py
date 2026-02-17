# app/prompts/prompt_builder.py

from typing import List, Dict

from app.prompts.system_prompts import (
    DOCUMENT_QA_SYSTEM_PROMPT,
    LOCAL_MODEL_SYSTEM_PROMPT
)


def build_document_prompt(
    question: str,
    context_chunks: List[Dict],
    model_type: str = "cloud"
) -> str:
    """
    Build production-grade grounded prompt.

    model_type:
        "cloud" → OpenAI / Gemini
        "local" → flan-t5 fallback model
    """

    # Select correct system prompt
    if model_type == "local":
        system_prompt = LOCAL_MODEL_SYSTEM_PROMPT
    else:
        system_prompt = DOCUMENT_QA_SYSTEM_PROMPT

    # Build context block
    context_block = "\n\n".join(
        f"[Context {i+1} | Confidence: {chunk['similarity_score']:.3f}]\n{chunk['text']}"
        for i, chunk in enumerate(context_chunks)
    )

    # Final prompt (does NOT override system behavior)
    prompt = f"""
{system_prompt}

DOCUMENT CONTEXT:
----------------
{context_block}
----------------

QUESTION:
{question}

INSTRUCTIONS:

Answer using ONLY the DOCUMENT CONTEXT above.

You MAY combine information from multiple context sections.

If the answer does not exist in the context, say:
"I don't have enough information in the document to answer this."

FINAL ANSWER:
"""

    return prompt.strip()
