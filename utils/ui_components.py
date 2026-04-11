"""
ui_components.py - Reusable Streamlit UI components for DocuMind Analyst.

All components are stateless: they accept data as explicit arguments and
render a self-contained section. Session state is never read here.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import streamlit as st

from core.insight_engine import ActionableTakeaways, SuggestedQuestions
from utils.helpers import (
    format_chat_export,
    format_file_size,
    format_summary_export,
)

# ---------------------------------------------------------------------------
# Shared style constants
# ---------------------------------------------------------------------------

_CARD_BASE = (
    "background:#f9fafb; border:1px solid #e5e7eb; border-radius:6px; "
    "padding:1rem 1.2rem; font-size:0.92rem; line-height:1.75; color:#111827;"
)
_ACCENT_INDIGO  = "border-left:4px solid #6366f1;"
_ACCENT_GREEN   = "border-left:4px solid #10b981;"
_ACCENT_AMBER   = "border-left:4px solid #f59e0b;"
_ACCENT_ROSE    = "border-left:4px solid #f43f5e;"
_ACCENT_NEUTRAL = "border-left:4px solid #9ca3af;"

_LABEL_STYLE = (
    "font-size:0.72rem; font-weight:700; letter-spacing:0.08em; "
    "text-transform:uppercase; color:#6b7280; margin-bottom:0.4rem;"
)


def _card(body: str, accent: str = _ACCENT_NEUTRAL) -> str:
    return (
        f"<div style='{_CARD_BASE} {accent}'>{body}</div>"
    )


def _section_label(text: str) -> str:
    return f"<p style='{_LABEL_STYLE}'>{text}</p>"


# ---------------------------------------------------------------------------
# Page chrome
# ---------------------------------------------------------------------------

def render_header() -> None:
    """Render the application masthead and horizontal divider."""
    st.markdown(
        """
        <div style="padding:1.6rem 0 0.5rem 0;">
            <h1 style="font-size:2rem; font-weight:800; letter-spacing:-0.3px;
                       margin-bottom:0.1rem; color:#111827;">
                DocuMind Analyst
            </h1>
            <p style="font-size:0.9rem; color:#6b7280; margin-top:0;">
                AI Document Intelligence &nbsp;&middot;&nbsp;
                Decision-Support Platform
            </p>
        </div>
        <hr style="border:none; border-top:1px solid #e5e7eb;
                   margin:0 0 1rem 0;" />
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(
    model_name: str,
    chunk_size: int,
    chunk_overlap: int,
    top_k: int,
    is_indexed: bool,
    on_reset: Optional[Any] = None,
) -> None:
    """
    Render the sidebar with runtime configuration, document status,
    and an optional document reset control.

    Args:
        model_name:    Active Groq model identifier.
        chunk_size:    Configured chunk size in characters.
        chunk_overlap: Configured chunk overlap in characters.
        top_k:         Retrieval chunks per query.
        is_indexed:    Whether a document is currently loaded.
        on_reset:      Zero-argument callable invoked on reset click.
    """
    with st.sidebar:
        st.markdown(
            "<h2 style='font-size:1.1rem; font-weight:700; color:#111827;"
            " margin-bottom:0.1rem;'>DocuMind Analyst</h2>",
            unsafe_allow_html=True,
        )
        st.caption("AI Document Intelligence Platform")
        st.divider()

        st.markdown(
            _section_label("Runtime Configuration"),
            unsafe_allow_html=True,
        )
        st.markdown("**LLM Model**")
        st.code(model_name, language=None)

        col_a, col_b = st.columns(2)
        col_a.metric("Chunk Size",    chunk_size)
        col_b.metric("Chunk Overlap", chunk_overlap)
        col_a.metric("Top-K",         top_k)

        st.divider()
        st.markdown(
            _section_label("Document Status"),
            unsafe_allow_html=True,
        )
        if is_indexed:
            st.success("Document indexed. All features active.")
        else:
            st.info("No document loaded.")

        if is_indexed and on_reset is not None:
            st.divider()
            if st.button(
                "Reset Document",
                use_container_width=True,
                type="secondary",
                help="Unload the current document and return to the upload screen.",
            ):
                on_reset()

        st.divider()
        st.caption(
            "Streamlit · Groq · FAISS · "
            "sentence-transformers · PyMuPDF"
        )


# ---------------------------------------------------------------------------
# Metrics row
# ---------------------------------------------------------------------------

def render_metrics_row(
    metadata: Dict[str, Any],
    file_size: int,
    chunk_count: int,
) -> None:
    """
    Render a four-column row of document metric cards.

    Args:
        metadata:    Dict from pdf_reader.get_document_metadata().
        file_size:   Raw file size in bytes.
        chunk_count: Number of indexed chunks.
    """
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Pages",     metadata.get("page_count", "—"))
    col2.metric("Chunks",    chunk_count)
    col3.metric("File Size", format_file_size(file_size))
    col4.metric("Author",    _truncate(metadata.get("author", "Unknown"), 24))


# ---------------------------------------------------------------------------
# Overview tab
# ---------------------------------------------------------------------------

def render_document_overview(
    metadata: Dict[str, Any],
    pages: List[Dict[str, Any]],
    max_preview_pages: int = 3,
    max_chars: int = 600,
) -> None:
    """
    Render document metadata details and page text previews.

    Args:
        metadata:          Dict from pdf_reader.get_document_metadata().
        pages:             Extracted page list.
        max_preview_pages: Maximum pages to display.
        max_chars:         Maximum characters per page preview.
    """
    st.markdown("#### Document Details")
    _render_metadata_table(metadata)
    st.divider()

    st.markdown("#### Extracted Text Preview")
    st.caption(
        f"Displaying {min(max_preview_pages, len(pages))} "
        f"of {len(pages)} extracted pages."
    )
    for page in pages[:max_preview_pages]:
        page_num = page.get("page_number", "?")
        text     = page.get("text", "").strip()
        with st.expander(f"Page {page_num}", expanded=(page_num == 1)):
            if text:
                display = (
                    text[:max_chars] + " ..."
                    if len(text) > max_chars
                    else text
                )
                st.text(display)
            else:
                st.caption("No extractable text on this page.")


# ---------------------------------------------------------------------------
# Q&A components
# ---------------------------------------------------------------------------

def render_answer_panel(answer: str) -> None:
    """
    Render the primary LLM answer in a styled accent card.

    Args:
        answer: Answer string from QAEngine.answer().
    """
    st.markdown("#### Answer")
    st.markdown(_card(answer, _ACCENT_INDIGO), unsafe_allow_html=True)


def render_source_evidence(sources: List[Dict[str, Any]]) -> None:
    """
    Render collapsible source chunk evidence panels.

    Args:
        sources: Chunk dicts from QAEngine.answer()["sources"],
                 each augmented with a score key.
    """
    if not sources:
        st.caption("No source chunks are available for this answer.")
        return

    st.markdown("#### Source Evidence")
    st.caption(f"{len(sources)} chunk(s) retrieved and used for grounding.")

    for i, chunk in enumerate(sources, start=1):
        page      = chunk.get("page_number", "?")
        score     = chunk.get("score", 0.0)
        text      = chunk.get("text", "").strip()
        label     = (
            f"[{i}]  Page {page}  "
            f"—  Relevance {min(score * 100, 100):.1f}%"
        )
        with st.expander(label, expanded=False):
            st.markdown(
                f"<div style='font-size:0.875rem; line-height:1.65; "
                f"color:#374151;'>{text}</div>",
                unsafe_allow_html=True,
            )
            st.caption(
                f"Source: {chunk.get('source', '—')}  |  "
                f"Page {page}  |  "
                f"Characters {chunk.get('char_start', '?')}"
                f"–{chunk.get('char_end', '?')}"
            )


# ---------------------------------------------------------------------------
# Insight components
# ---------------------------------------------------------------------------

def render_key_insights(insights: List[str]) -> None:
    """
    Render a numbered list of key document insights.

    Args:
        insights: List of insight strings from InsightResult.key_insights.
    """
    if not insights:
        render_empty_state(
            "No key insights were extracted.",
            "The document may lack sufficient analytical content.",
        )
        return

    st.markdown("#### Key Insights")
    st.caption(f"{len(insights)} material finding(s) identified.")

    for i, insight in enumerate(insights, start=1):
        st.markdown(
            f"<div style='display:flex; gap:0.75rem; margin-bottom:0.6rem;'>"
            f"<div style='min-width:1.6rem; font-weight:700; color:#6366f1;"
            f" font-size:0.9rem; padding-top:0.05rem;'>{i:02d}</div>"
            f"<div style='font-size:0.92rem; line-height:1.7; color:#111827;'>"
            f"{insight}</div></div>",
            unsafe_allow_html=True,
        )


def render_enhanced_explanation(explanation: str) -> None:
    """
    Render the enhanced analytical explanation narrative.

    Args:
        explanation: Prose string from InsightResult.enhanced_explanation.
    """
    if not explanation.strip():
        render_empty_state(
            "No enhanced explanation was generated.",
            "Run the insight analysis to produce this output.",
        )
        return

    st.markdown("#### Enhanced Analytical Explanation")
    st.markdown(
        _card(explanation.replace("\n", "<br/>"), _ACCENT_GREEN),
        unsafe_allow_html=True,
    )


def render_suggested_questions(questions: SuggestedQuestions) -> None:
    """
    Render the four stakeholder-segmented question groups.

    Args:
        questions: Populated SuggestedQuestions dataclass instance.
    """
    groups = [
        ("Executive",     questions.executive,     _ACCENT_INDIGO),
        ("Analytical",    questions.analytical,    _ACCENT_GREEN),
        ("Risk",          questions.risk,          _ACCENT_ROSE),
        ("Visualization", questions.visualization, _ACCENT_AMBER),
    ]

    if not any(items for _, items, _ in groups):
        render_empty_state(
            "No suggested questions were generated.",
            "Run the insight analysis to populate this section.",
        )
        return

    for group_label, items, accent in groups:
        if not items:
            continue
        st.markdown(
            _section_label(group_label), unsafe_allow_html=True
        )
        for item in items:
            st.markdown(
                f"<div style='{_CARD_BASE} {accent} margin-bottom:0.4rem;'>"
                f"{item}</div>",
                unsafe_allow_html=True,
            )
        st.markdown(
            "<div style='margin-bottom:0.8rem;'></div>",
            unsafe_allow_html=True,
        )


def render_actionable_takeaways(takeaways: ActionableTakeaways) -> None:
    """
    Render role-segmented actionable takeaways and next steps.

    Args:
        takeaways: Populated ActionableTakeaways dataclass instance.
    """
    groups = [
        ("Leadership", takeaways.leadership, _ACCENT_INDIGO),
        ("Analyst",    takeaways.analyst,    _ACCENT_GREEN),
        ("Next Steps", takeaways.next_steps, _ACCENT_AMBER),
    ]

    if not any(items for _, items, _ in groups):
        render_empty_state(
            "No actionable takeaways were generated.",
            "Run the insight analysis to populate this section.",
        )
        return

    for group_label, items, accent in groups:
        if not items:
            continue
        st.markdown(
            _section_label(group_label), unsafe_allow_html=True
        )
        for idx, item in enumerate(items, start=1):
            st.markdown(
                f"<div style='{_CARD_BASE} {accent} margin-bottom:0.4rem;'>"
                f"<span style='font-weight:600; margin-right:0.5rem;'>"
                f"{idx}.</span>{item}</div>",
                unsafe_allow_html=True,
            )
        st.markdown(
            "<div style='margin-bottom:0.8rem;'></div>",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Summary panel
# ---------------------------------------------------------------------------

def render_summary_panel(summary: str) -> None:
    """
    Render a generated document summary in a styled card.

    Args:
        summary: Summary text from Summarizer.full_summary().
    """
    st.markdown("#### Document Summary")
    st.markdown(
        _card(summary.replace("\n", "<br/>"), _ACCENT_GREEN),
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Chat history
# ---------------------------------------------------------------------------

def render_chat_history(chat_history: List[Dict[str, Any]]) -> None:
    """
    Render the full Q&A session history in reverse-chronological order.

    Args:
        chat_history: Turn dicts from session_manager.get_chat_history().
    """
    if not chat_history:
        render_empty_state(
            "No questions have been asked in this session.",
            "Submit a question in the Ask Questions tab to begin.",
        )
        return

    st.caption(f"{len(chat_history)} exchange(s) recorded in this session.")

    for turn in reversed(chat_history):
        with st.container():
            st.markdown(
                f"<p style='font-size:0.78rem; font-weight:700; "
                f"letter-spacing:0.06em; text-transform:uppercase; "
                f"color:#6b7280; margin-bottom:0.2rem;'>"
                f"Turn {turn['turn']}</p>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<p style='font-size:0.92rem; font-weight:600; "
                f"color:#111827; margin-bottom:0.4rem;'>"
                f"{turn['question']}</p>",
                unsafe_allow_html=True,
            )
            st.markdown(
                _card(turn["answer"], _ACCENT_NEUTRAL),
                unsafe_allow_html=True,
            )
            sources = turn.get("sources", [])
            if sources:
                pages = sorted(
                    {s.get("page_number", "?") for s in sources}
                )
                st.caption(
                    f"Sourced from: Page(s) "
                    f"{', '.join(str(p) for p in pages)}"
                )
            st.divider()


# ---------------------------------------------------------------------------
# Download section
# ---------------------------------------------------------------------------

def render_download_section(
    summary: Optional[str],
    chat_history: List[Dict[str, Any]],
    metadata: Dict[str, Any],
) -> None:
    """
    Render download controls for summary and chat history exports.

    Buttons are disabled when the corresponding content has not yet
    been generated.

    Args:
        summary:      Cached summary string, or None.
        chat_history: Current chat history list.
        metadata:     Document metadata dict.
    """
    st.markdown("#### Export")
    file_stem = _safe_filename(metadata.get("file_name", "document"))
    col1, col2 = st.columns(2)

    with col1:
        if summary:
            st.download_button(
                label="Download Summary (.txt)",
                data=format_summary_export(summary, metadata),
                file_name=f"{file_stem}_summary.txt",
                mime="text/plain",
                use_container_width=True,
            )
        else:
            st.button(
                "Download Summary (.txt)",
                disabled=True,
                use_container_width=True,
                help="Generate a summary in the Summary tab to enable this export.",
            )

    with col2:
        if chat_history:
            st.download_button(
                label="Download Chat History (.txt)",
                data=format_chat_export(chat_history, metadata),
                file_name=f"{file_stem}_chat.txt",
                mime="text/plain",
                use_container_width=True,
            )
        else:
            st.button(
                "Download Chat History (.txt)",
                disabled=True,
                use_container_width=True,
                help="Ask at least one question to enable this export.",
            )


# ---------------------------------------------------------------------------
# Empty states and error notices
# ---------------------------------------------------------------------------

def render_empty_state(message: str, detail: str = "") -> None:
    """
    Render a neutral empty-state notice.

    Args:
        message: Primary notice line.
        detail:  Optional secondary guidance.
    """
    body = f"**{message}**"
    if detail:
        body += f"\n\n{detail}"
    st.info(body)


def render_processing_error(error: str) -> None:
    """
    Render a user-facing processing error notice.

    Args:
        error: Error message string.
    """
    st.error(f"Processing error: {error}")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _render_metadata_table(metadata: Dict[str, Any]) -> None:
    """Render document metadata as a two-column label/value layout."""
    display_keys = [
        ("file_name",  "File Name"),
        ("title",      "Title"),
        ("author",     "Author"),
        ("subject",    "Subject"),
        ("page_count", "Page Count"),
    ]
    rows = [
        (label, str(metadata[key]))
        for key, label in display_keys
        if metadata.get(key) and str(metadata.get(key)) not in ("Unknown", "")
    ]
    if not rows:
        st.caption("No metadata available for this document.")
        return
    for label, value in rows:
        col_l, col_r = st.columns([1, 2])
        col_l.markdown(f"**{label}**")
        col_r.markdown(value)


def _truncate(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len - 1] + "..."


def _safe_filename(name: str) -> str:
    import os
    import re
    stem = os.path.splitext(name)[0] if "." in name else name
    return re.sub(r"\s+", "_", stem)


