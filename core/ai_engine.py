"""
core/ai_engine.py - Centralised Groq LLM connector for DocuMind Analyst.

Exposes two public interfaces that co-exist without conflict:

    1. generate_response(system_prompt, user_prompt, max_tokens, temperature)
       → Stateless function used by QAEngine, Summarizer, InsightEngine, etc.

    2. AIEngine class
       → Stateful wrapper required by AgenticEngine and app.py.
         Exposes:  complete(system_prompt, user_prompt, max_tokens, temperature)
         which delegates to generate_response() internally.

All LLM traffic is routed through Groq's OpenAI-compatible chat completions
endpoint.  The Groq client is constructed lazily on first use.

Credentials are read (in priority order) from:
    1. st.secrets["GROQ_API_KEY"]          — Streamlit secrets (preferred)
    2. os.environ["GROQ_API_KEY"]          — environment variable fallback
    3. core.config.GROQ_API_KEY            — project config fallback

Model name is read from:
    1. st.secrets.get("GROQ_MODEL_NAME")
    2. os.environ.get("GROQ_MODEL_NAME")
    3. core.config.MODEL_NAME
    4. Hard-coded default: "llama-3.1-8b-instant"
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

try:
    from groq import Groq
except ImportError as exc:
    raise ImportError(
        "The groq package is required: pip install groq"
    ) from exc

# ---------------------------------------------------------------------------
# Optional project config import (non-fatal if absent)
# ---------------------------------------------------------------------------

_cfg_api_key:    str | None = None
_cfg_model:      str | None = None
_cfg_max_tokens: int | None = None
_cfg_temperature: float | None = None

try:
    from core.config import GROQ_API_KEY as _cfg_api_key        # type: ignore[assignment]
except ImportError:
    pass

try:
    from core.config import MODEL_NAME as _cfg_model             # type: ignore[assignment]
except ImportError:
    pass

try:
    from core.config import MAX_TOKENS as _cfg_max_tokens        # type: ignore[assignment]
except ImportError:
    pass

try:
    from core.config import TEMPERATURE as _cfg_temperature      # type: ignore[assignment]
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Module defaults
# ---------------------------------------------------------------------------

_DEFAULT_MODEL       = "llama-3.1-8b-instant"
_DEFAULT_MAX_TOKENS  = 1024
_DEFAULT_TEMPERATURE = 0.1


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_api_key() -> str:
    """
    Resolve the Groq API key from the first available source.

    Priority: st.secrets → os.environ → core.config

    Raises:
        RuntimeError: If no key is found in any source.
    """
    # 1. Streamlit secrets (most common in deployed apps)
    try:
        import streamlit as st
        key = st.secrets.get("GROQ_API_KEY", "")
        if key and key.strip():
            return key.strip()
    except Exception:
        pass

    # 2. Environment variable
    key = os.environ.get("GROQ_API_KEY", "")
    if key and key.strip():
        return key.strip()

    # 3. core.config
    if _cfg_api_key and str(_cfg_api_key).strip():
        return str(_cfg_api_key).strip()

    raise RuntimeError(
        "GROQ_API_KEY is not set. "
        "Add it to .streamlit/secrets.toml, set the GROQ_API_KEY environment "
        "variable, or define it in core/config.py."
    )


def _resolve_model() -> str:
    """Return the active model name from the first available source."""
    try:
        import streamlit as st
        name = st.secrets.get("GROQ_MODEL_NAME", "")
        if name and name.strip():
            return name.strip()
    except Exception:
        pass

    name = os.environ.get("GROQ_MODEL_NAME", "")
    if name and name.strip():
        return name.strip()

    if _cfg_model and str(_cfg_model).strip():
        return str(_cfg_model).strip()

    return _DEFAULT_MODEL


def _build_client() -> Groq:
    """Construct an authenticated Groq client. Raises RuntimeError on failure."""
    api_key = _resolve_api_key()
    try:
        return Groq(api_key=api_key)
    except Exception as exc:
        raise RuntimeError(f"Failed to instantiate Groq client: {exc}") from exc


# ---------------------------------------------------------------------------
# Stateless public function  (used by QAEngine, Summarizer, InsightEngine …)
# ---------------------------------------------------------------------------

def generate_response(
    system_prompt: str,
    user_prompt:   str,
    max_tokens:    int   = _DEFAULT_MAX_TOKENS,
    temperature:   float = _DEFAULT_TEMPERATURE,
) -> str:
    """
    Send a single chat completion request to Groq and return the text.

    This is the canonical stateless entry point for all LLM calls in
    DocuMind Analyst.  Callers supply fully-formed prompts; this function
    handles client construction, request formatting, and error surfacing.

    Args:
        system_prompt: Instruction context shaping model behaviour.
        user_prompt:   The user-facing content or query.
        max_tokens:    Upper bound on response tokens.
        temperature:   Sampling temperature in [0.0, 2.0].

    Returns:
        Stripped response string from the model.

    Raises:
        ValueError:   If either prompt is blank.
        RuntimeError: If the API key is missing or the Groq call fails.
    """
    if not system_prompt.strip():
        raise ValueError("system_prompt must not be blank.")
    if not user_prompt.strip():
        raise ValueError("user_prompt must not be blank.")

    client = _build_client()
    model  = _resolve_model()

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        raise RuntimeError(
            f"Groq API request failed (model={model}, "
            f"max_tokens={max_tokens}): {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Stateful AIEngine class  (used by AgenticEngine and app.py)
# ---------------------------------------------------------------------------

class AIEngine:
    """
    Stateful LLM wrapper for use by AgenticEngine and the pipeline layer.

    Wraps generate_response() with an OOP interface so AgenticEngine can
    hold a typed reference to the engine without importing generate_response
    directly.

    Usage:
        ai_engine = AIEngine()
        response  = ai_engine.complete(system_prompt, user_prompt)

    The Groq client is NOT constructed in __init__; it is built lazily
    on the first call to complete() to avoid blocking at import time and
    to prevent startup errors when the API key is not yet available.
    """

    def __init__(self) -> None:
        # Validate that a key is resolvable at construction time so callers
        # get an early, clear error rather than a failure on first query.
        _resolve_api_key()

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def complete(
        self,
        system_prompt: str,
        user_prompt:   str,
        max_tokens:    int   | None = None,
        temperature:   float | None = None,
    ) -> str:
        """
        Generate a completion via Groq and return the response text.

        Args:
            system_prompt: Instruction context for the model.
            user_prompt:   The user query or content.
            max_tokens:    Token limit (defaults to config / 1024).
            temperature:   Sampling temperature (defaults to config / 0.1).

        Returns:
            Stripped response string.

        Raises:
            ValueError:   If either prompt is blank.
            RuntimeError: If the API key is missing or the Groq call fails.
        """
        resolved_max_tokens  = (
            max_tokens
            if max_tokens is not None
            else (_cfg_max_tokens or _DEFAULT_MAX_TOKENS)
        )
        resolved_temperature = (
            temperature
            if temperature is not None
            else (_cfg_temperature or _DEFAULT_TEMPERATURE)
        )

        return generate_response(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=int(resolved_max_tokens),
            temperature=float(resolved_temperature),
        )

    # -------------------------------------------------------------------------
    # Convenience alias expected by some AgenticEngine implementations
    # -------------------------------------------------------------------------

    def generate(
        self,
        system_prompt: str,
        user_prompt:   str,
        max_tokens:    int   | None = None,
        temperature:   float | None = None,
    ) -> str:
        """Alias for complete(); satisfies alternate AgenticEngine call sites."""
        return self.complete(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def __repr__(self) -> str:
        return f"AIEngine(model={_resolve_model()!r})"