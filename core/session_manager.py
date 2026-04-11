"""
session_manager.py - Streamlit session state management for DocuMind Analyst.

Centralises all session state initialisation, access, and mutation.
No module outside this one reads or writes st.session_state keys using
raw string literals. All keys are declared on the _Keys registry class.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import streamlit as st

from core.insight_engine import ActionableTakeaways, InsightResult, SuggestedQuestions


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHAT_HISTORY_LIMIT: int = 10  # Maximum number of Q&A turns retained


# ---------------------------------------------------------------------------
# Key registry
# ---------------------------------------------------------------------------

class _Keys:
    PDF_PATH             = "pdf_path"
    METADATA             = "metadata"
    FILE_SIZE            = "file_size"
    PAGES                = "pages"
    CHUNKS               = "chunks"
    VECTOR_DB            = "vector_db"
    QA_ENGINE            = "qa_engine"
    SUMMARIZER           = "summarizer"
    INDEXED              = "indexed"
    SUMMARY              = "summary"
    CHAT_HISTORY         = "chat_history"
    LAST_ANSWER          = "last_answer"
    INSIGHTS             = "insights"
    SUGGESTED_QUESTIONS  = "suggested_questions"
    ACTION_ITEMS         = "action_items"
    ENHANCED_EXPLANATION = "enhanced_explanation"


K = _Keys


# ---------------------------------------------------------------------------
# Safe defaults factory
# ---------------------------------------------------------------------------

def _defaults() -> Dict[str, Any]:
    """
    Return a fresh dict of every session key mapped to its safe default.
    Centralised here so initialise() and reset_session() share one source
    of truth.
    """
    return {
        K.PDF_PATH:             None,
        K.METADATA:             None,
        K.FILE_SIZE:            0,
        K.PAGES:                [],
        K.CHUNKS:               [],
        K.VECTOR_DB:            None,
        K.QA_ENGINE:            None,
        K.SUMMARIZER:           None,
        K.INDEXED:              False,
        K.SUMMARY:              None,
        K.CHAT_HISTORY:         [],
        K.LAST_ANSWER:          None,
        K.INSIGHTS:             None,
        K.SUGGESTED_QUESTIONS:  None,
        K.ACTION_ITEMS:         None,
        K.ENHANCED_EXPLANATION: None,
    }


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

def initialise() -> None:
    """
    Ensure every session key exists with a safe default value.

    Idempotent: existing values are never overwritten. Safe to call on
    every Streamlit rerun — it only writes keys that are genuinely absent.
    """
    for key, default in _defaults().items():
        if key not in st.session_state:
            # Use a copy for mutable defaults to prevent shared-reference bugs
            st.session_state[key] = (
                list(default) if isinstance(default, list)
                else dict(default) if isinstance(default, dict)
                else default
            )


# ---------------------------------------------------------------------------
# Full session reset
# ---------------------------------------------------------------------------

def reset_session() -> None:
    """
    Wipe ALL managed session state keys and restore every key to its safe
    default, including chat history.

    Use this for a full application restart (e.g. "Start Over" button).
    To preserve chat history while clearing document state, call
    reset_document_state() instead.
    """
    for key, default in _defaults().items():
        st.session_state[key] = (
            list(default) if isinstance(default, list)
            else dict(default) if isinstance(default, dict)
            else default
        )


# ---------------------------------------------------------------------------
# Document state helpers
# ---------------------------------------------------------------------------

def is_document_indexed() -> bool:
    """Return True when a document has been fully processed and indexed."""
    return bool(st.session_state.get(K.INDEXED, False))


def is_new_upload(file_name: str) -> bool:
    """
    Return True when the supplied file name differs from the document
    currently held in session state.

    Args:
        file_name: uploaded_file.name from st.file_uploader.
    """
    current: Optional[str] = st.session_state.get(K.PDF_PATH)
    if not current:
        return True
    return os.path.basename(current) != file_name


def store_pipeline_result(result: Dict[str, Any]) -> None:
    """
    Persist the output of document_service.run_pipeline to session state.
    Clears all derived outputs so stale content from a previous document
    is not shown alongside new content.

    Args:
        result: Dict returned by document_service.run_pipeline().
    """
    st.session_state[K.PDF_PATH]   = result["pdf_path"]
    st.session_state[K.METADATA]   = result["metadata"]
    st.session_state[K.FILE_SIZE]  = result["file_size"]
    st.session_state[K.PAGES]      = result["pages"]
    st.session_state[K.CHUNKS]     = result["chunks"]
    st.session_state[K.VECTOR_DB]  = result["vector_db"]
    st.session_state[K.QA_ENGINE]  = result["qa_engine"]
    st.session_state[K.SUMMARIZER] = result["summarizer"]
    st.session_state[K.INDEXED]    = True

    for key in (
        K.SUMMARY, K.LAST_ANSWER,
        K.INSIGHTS, K.SUGGESTED_QUESTIONS,
        K.ACTION_ITEMS, K.ENHANCED_EXPLANATION,
    ):
        st.session_state[key] = None


def reset_document_state() -> None:
    """
    Clear all document-scoped and derived session keys and restore
    safe defaults. Chat history is preserved unless reset_chat_history
    is called explicitly.
    """
    document_keys = [
        K.PDF_PATH, K.METADATA, K.FILE_SIZE, K.PAGES, K.CHUNKS,
        K.VECTOR_DB, K.QA_ENGINE, K.SUMMARIZER, K.INDEXED,
        K.SUMMARY, K.LAST_ANSWER,
        K.INSIGHTS, K.SUGGESTED_QUESTIONS,
        K.ACTION_ITEMS, K.ENHANCED_EXPLANATION,
    ]
    for key in document_keys:
        st.session_state.pop(key, None)
    initialise()


# ---------------------------------------------------------------------------
# Insight state helpers
# ---------------------------------------------------------------------------

def store_insight_result(result: InsightResult) -> None:
    """
    Distribute InsightResult fields into individual session state keys.

    Args:
        result: Output of insight_engine.generate_insights().
    """
    st.session_state[K.INSIGHTS]             = result.key_insights
    st.session_state[K.SUGGESTED_QUESTIONS]  = result.suggested_questions
    st.session_state[K.ACTION_ITEMS]         = result.actionable_takeaways
    st.session_state[K.ENHANCED_EXPLANATION] = result.enhanced_explanation


def get_insights() -> Optional[List[str]]:
    """Return cached key insights, or None if not yet generated."""
    return st.session_state.get(K.INSIGHTS)


def get_suggested_questions() -> Optional[SuggestedQuestions]:
    """Return cached suggested questions, or None if not yet generated."""
    return st.session_state.get(K.SUGGESTED_QUESTIONS)


def get_action_items() -> Optional[ActionableTakeaways]:
    """Return cached actionable takeaways, or None if not yet generated."""
    return st.session_state.get(K.ACTION_ITEMS)


def get_enhanced_explanation() -> Optional[str]:
    """Return cached enhanced explanation, or None if not yet generated."""
    return st.session_state.get(K.ENHANCED_EXPLANATION)


def insights_generated() -> bool:
    """Return True if the insight engine has run for the current document."""
    return st.session_state.get(K.INSIGHTS) is not None


# ---------------------------------------------------------------------------
# Chat history helpers
# ---------------------------------------------------------------------------

def _is_duplicate_turn(history: List[Dict[str, Any]], question: str) -> bool:
    """
    Return True if the most recent turn in history has an identical
    question string. Prevents double-appending on Streamlit reruns.

    Args:
        history:  The current chat history list.
        question: The incoming question string.
    """
    if not history:
        return False
    return history[-1].get("question", "") == question


def append_chat_turn(
    question: str,
    answer: str,
    sources: List[Dict[str, Any]],
) -> None:
    """
    Append one Q&A exchange to the persistent chat history.

    Deduplication: if the last recorded turn carries the same question
    string, the new turn is silently dropped to prevent double-writes
    caused by Streamlit reruns.

    History limit: only the most recent CHAT_HISTORY_LIMIT turns are
    retained. Older turns are discarded (FIFO) to bound memory usage.

    Args:
        question: The user's question string.
        answer:   The LLM-generated answer string.
        sources:  Retrieved chunk dicts augmented with a score key.
    """
    history: List[Dict[str, Any]] = st.session_state.get(K.CHAT_HISTORY, [])

    # Guard: do not append a duplicate of the immediately preceding turn
    if _is_duplicate_turn(history, question):
        return

    history.append(
        {
            "turn":     len(history) + 1,
            "question": question,
            "answer":   answer,
            "sources":  sources,
        }
    )

    # Enforce sliding-window limit: keep only the last N turns
    if len(history) > CHAT_HISTORY_LIMIT:
        history = history[-CHAT_HISTORY_LIMIT:]

    st.session_state[K.CHAT_HISTORY] = history


def get_chat_history() -> List[Dict[str, Any]]:
    """Return the full chat history list, oldest turn first."""
    return st.session_state.get(K.CHAT_HISTORY, [])


def reset_chat_history() -> None:
    """Erase all recorded Q&A exchanges from session state."""
    st.session_state[K.CHAT_HISTORY] = []


def get_chat_history_count() -> int:
    """Return the number of turns currently stored in chat history."""
    return len(st.session_state.get(K.CHAT_HISTORY, []))


def get_chat_history_limit() -> int:
    """Return the maximum number of turns retained in chat history."""
    return CHAT_HISTORY_LIMIT


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

def store_summary(summary: Optional[str]) -> None:
    """Persist a generated summary to session state."""
    st.session_state[K.SUMMARY] = summary


def get_summary() -> Optional[str]:
    """Return the cached summary, or None if not yet generated."""
    return st.session_state.get(K.SUMMARY)


# ---------------------------------------------------------------------------
# Generic getter / setter
# ---------------------------------------------------------------------------

def get(key: str) -> Any:
    """
    Generic session state accessor.

    Args:
        key: A key string, preferably a constant from K.

    Returns:
        The stored value, or None if the key is absent.
    """
    return st.session_state.get(key)


def set(key: str, value: Any) -> None:  # noqa: A001
    """
    Generic session state setter. Prefer the typed helpers above for
    well-known keys; use this only for ad-hoc or dynamic keys.

    Args:
        key:   A key string, preferably a constant from K.
        value: The value to store.
    """
    st.session_state[key] = value