"""
config.py - Runtime configuration for DocuMind Analyst.

All settings are sourced from Streamlit secrets (.streamlit/secrets.toml)
with sensible production defaults applied when a key is absent. No dotenv
dependency is used. GROQ_API_KEY is never exposed as a module-level
constant; it is read exclusively inside core.ai_engine.

Secrets file template (.streamlit/secrets.toml):

    GROQ_API_KEY     = "your_key_here"
    GROQ_MODEL_NAME  = "llama-3.1-8b-instant"
    CHUNK_SIZE       = 800
    CHUNK_OVERLAP    = 150
    EMBEDDING_MODEL  = "all-MiniLM-L6-v2"
    TOP_K_RESULTS    = 4
    UPLOAD_DIR       = "data/uploads"
    MAX_TOKENS       = 1024
    TEMPERATURE      = 0.2
"""

from __future__ import annotations

import streamlit as st


def _get(key: str, default: object) -> object:
    """
    Read a value from st.secrets with a typed fallback default.

    Args:
        key:     Secret key to look up.
        default: Value returned when the key is absent.

    Returns:
        The secret value if present, otherwise ``default``.
    """
    return st.secrets.get(key, default)


# -- Groq API -----------------------------------------------------------------
GROQ_MODEL_NAME: str = str(_get("GROQ_MODEL_NAME", "llama-3.1-8b-instant"))

# -- Chunking -----------------------------------------------------------------
CHUNK_SIZE: int    = int(_get("CHUNK_SIZE", 800))
CHUNK_OVERLAP: int = int(_get("CHUNK_OVERLAP", 150))

# -- Embeddings ---------------------------------------------------------------
EMBEDDING_MODEL: str = str(_get("EMBEDDING_MODEL", "all-MiniLM-L6-v2"))

# -- FAISS Retrieval ----------------------------------------------------------
TOP_K_RESULTS: int = int(_get("TOP_K_RESULTS", 4))

# -- File Storage -------------------------------------------------------------
UPLOAD_DIR: str = str(_get("UPLOAD_DIR", "data/uploads"))

# -- LLM Generation -----------------------------------------------------------
MAX_TOKENS: int    = int(_get("MAX_TOKENS", 1024))
TEMPERATURE: float = float(_get("TEMPERATURE", 0.2))


def validate_config() -> None:
    """
    Assert that all required secrets are present and non-empty.

    Must be called once at application startup before any AI client
    is constructed. Raises early to prevent misleading downstream errors.

    Raises:
        EnvironmentError: If GROQ_API_KEY is absent or empty.
    """
    api_key: str = str(st.secrets.get("GROQ_API_KEY", ""))
    if not api_key.strip():
        raise EnvironmentError(
            "GROQ_API_KEY is not configured. "
            "Add the following to .streamlit/secrets.toml:\n"
            '  GROQ_API_KEY = "your_api_key_here"'
        )