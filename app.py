"""
app.py — Enterprise AI Document Intelligence System v2.3
=========================================================
Upgrades over v2.2:
  - Single LLM call for all document intelligence sections
  - Faster execution (2–4 seconds vs 10–20 seconds)
  - Lightweight QA response caching via session_state
  - All features preserved with zero UI changes
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any

import streamlit as st

from core.document_service import run_pipeline

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="DocIQ — AI Research Analyst",
    page_icon="D",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ──────────────────────────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

    :root {
        --bg-base:       #0d0f14;
        --bg-surface:    #13161e;
        --bg-elevated:   #1a1e28;
        --bg-hover:      #1f2434;
        --border:        #262b3a;
        --border-accent: #3a4060;
        --text-primary:  #e8eaf0;
        --text-secondary:#8b91a8;
        --text-muted:    #555b72;
        --accent:        #4f8ef7;
        --accent-soft:   rgba(79,142,247,0.12);
        --accent-glow:   rgba(79,142,247,0.25);
        --green:         #3ecf8e;
        --green-soft:    rgba(62,207,142,0.12);
        --amber:         #f5a623;
        --amber-soft:    rgba(245,166,35,0.12);
        --red:           #f56565;
        --red-soft:      rgba(245,101,101,0.12);
        --purple:        #a78bfa;
        --purple-soft:   rgba(167,139,250,0.12);
    }

    html, body, [data-testid="stAppViewContainer"] {
        background: var(--bg-base) !important;
        font-family: 'DM Sans', sans-serif !important;
        color: var(--text-primary) !important;
    }
    [data-testid="stSidebar"] {
        background: var(--bg-surface) !important;
        border-right: 1px solid var(--border) !important;
    }
    .main .block-container {
        padding: 2rem 2.5rem 4rem !important;
        max-width: 1280px;
    }

    h1, h2, h3 { font-family: 'DM Serif Display', serif !important; }
    h1 { font-size: 2rem !important; letter-spacing: -0.02em; }
    h2 { font-size: 1.35rem !important; letter-spacing: -0.01em; }
    h3 { font-size: 1.1rem !important; }

    /* ── Tabs ── */
    [data-testid="stTabs"] > div:first-child {
        border-bottom: 1px solid var(--border) !important;
        gap: 0 !important;
    }
    button[data-baseweb="tab"] {
        background: transparent !important;
        color: var(--text-secondary) !important;
        font-family: 'DM Sans', sans-serif !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
        padding: 0.65rem 1.4rem !important;
        border-bottom: 2px solid transparent !important;
        transition: all 0.2s ease !important;
    }
    button[data-baseweb="tab"]:hover {
        color: var(--text-primary) !important;
        background: var(--bg-elevated) !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: var(--accent) !important;
        border-bottom-color: var(--accent) !important;
        background: transparent !important;
    }

    /* ── Cards ── */
    .iq-card {
        background: var(--bg-surface);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.5rem 1.75rem;
        margin-bottom: 1.25rem;
        transition: border-color 0.2s ease;
    }
    .iq-card:hover { border-color: var(--border-accent); }
    .iq-card-accent { border-left: 3px solid var(--accent); }
    .iq-card-green  { border-left: 3px solid var(--green); }
    .iq-card-amber  { border-left: 3px solid var(--amber); }
    .iq-card-red    { border-left: 3px solid var(--red); }
    .iq-card-purple { border-left: 3px solid var(--purple); }

    /* ── Intelligence header ── */
    .intel-header {
        background: linear-gradient(135deg, rgba(79,142,247,0.06) 0%, rgba(62,207,142,0.04) 100%);
        border: 1px solid rgba(79,142,247,0.2);
        border-radius: 14px;
        padding: 1.25rem 1.75rem;
        margin-bottom: 1rem;
    }
    .intel-header .intel-label {
        font-family: 'DM Mono', monospace;
        font-size: 0.62rem;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        color: var(--accent);
        margin-bottom: 0.45rem;
    }
    .intel-header p {
        font-family: 'DM Serif Display', serif;
        font-size: 1.1rem;
        line-height: 1.6;
        color: var(--text-primary);
        margin: 0;
        font-style: italic;
    }

    /* ── TLDR banner ── */
    .tldr-banner {
        background: linear-gradient(135deg, rgba(79,142,247,0.08) 0%, rgba(167,139,250,0.06) 100%);
        border: 1px solid var(--accent-glow);
        border-radius: 14px;
        padding: 1.5rem 2rem;
        margin-bottom: 0.75rem;
    }
    .tldr-banner .label {
        font-family: 'DM Mono', monospace;
        font-size: 0.65rem;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: var(--accent);
        margin-bottom: 0.5rem;
    }
    .tldr-banner p {
        font-size: 1.05rem;
        line-height: 1.65;
        color: var(--text-primary);
        margin: 0;
    }

    /* ── Key Takeaway ── */
    .takeaway-box {
        background: var(--green-soft);
        border: 1px solid rgba(62,207,142,0.3);
        border-radius: 10px;
        padding: 0.9rem 1.25rem;
        margin-bottom: 1.5rem;
        font-size: 0.92rem;
        color: #a8f0d0;
        line-height: 1.6;
    }
    .takeaway-box strong { color: var(--green); }

    /* ── Narrative connector ── */
    .narrative-connector {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        margin: -0.25rem 0 1rem;
        padding: 0 0.25rem;
    }
    .narrative-connector::before {
        content: '';
        display: block;
        width: 18px;
        height: 1px;
        background: var(--border-accent);
        flex-shrink: 0;
    }
    .narrative-connector span {
        font-family: 'DM Mono', monospace;
        font-size: 0.7rem;
        color: var(--text-muted);
        letter-spacing: 0.04em;
        font-style: italic;
    }

    /* ── Section label ── */
    .section-label {
        font-family: 'DM Mono', monospace;
        font-size: 0.68rem;
        font-weight: 500;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--text-muted);
        margin-bottom: 0.5rem;
    }

    /* ── Meta pill ── */
    .meta-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        background: var(--bg-elevated);
        border: 1px solid var(--border);
        border-radius: 999px;
        padding: 0.35rem 0.9rem;
        font-size: 0.8rem;
        color: var(--text-secondary);
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .meta-pill span { color: var(--text-primary); font-weight: 500; }

    /* ── Suggested question group header ── */
    .sq-group-header {
        font-family: 'DM Mono', monospace;
        font-size: 0.67rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--text-muted);
        margin: 0.9rem 0 0.5rem;
        padding-bottom: 0.35rem;
        border-bottom: 1px solid var(--border);
    }

    /* ── Answer panel ── */
    .answer-panel {
        background: var(--bg-surface);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.5rem;
        margin-top: 1rem;
    }

    /* ── Grounding indicator ── */
    .grounding-bar {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-top: 0.85rem;
        padding: 0.55rem 0.85rem;
        background: var(--bg-elevated);
        border: 1px solid var(--border);
        border-radius: 8px;
        font-size: 0.78rem;
        color: var(--text-secondary);
    }
    .grounding-bar .dot {
        width: 6px; height: 6px;
        border-radius: 50%;
        background: var(--green);
        flex-shrink: 0;
    }

    /* ── First-time guidance ── */
    .guidance-box {
        background: var(--accent-soft);
        border: 1px dashed rgba(79,142,247,0.3);
        border-radius: 10px;
        padding: 1.1rem 1.4rem;
        margin-top: 1rem;
        font-size: 0.88rem;
        color: var(--text-secondary);
        text-align: center;
        line-height: 1.6;
    }
    .guidance-box strong { color: var(--text-primary); }

    /* ── Mode explanation ── */
    .mode-explanation {
        background: var(--bg-elevated);
        border-left: 3px solid var(--accent);
        border-radius: 0 8px 8px 0;
        padding: 0.5rem 0.9rem;
        margin-top: 0.5rem;
        font-size: 0.8rem;
        color: var(--text-secondary);
        line-height: 1.5;
    }

    /* ── Mode toggle ── */
    .stRadio > div { gap: 0.5rem !important; }
    .stRadio label {
        background: var(--bg-elevated) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        padding: 0.3rem 0.8rem !important;
        font-size: 0.82rem !important;
        color: var(--text-secondary) !important;
        cursor: pointer !important;
        transition: all 0.15s !important;
    }
    .stRadio label:hover { border-color: var(--accent) !important; }

    /* ── Inputs ── */
    .stTextArea textarea, .stTextInput input {
        background: var(--bg-elevated) !important;
        border: 1px solid var(--border) !important;
        color: var(--text-primary) !important;
        border-radius: 10px !important;
        font-family: 'DM Sans', sans-serif !important;
    }
    .stTextArea textarea:focus, .stTextInput input:focus {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 2px var(--accent-glow) !important;
    }

    /* ── Buttons ── */
    .stButton button {
        background: var(--accent) !important;
        color: #fff !important;
        border: none !important;
        border-radius: 10px !important;
        font-family: 'DM Sans', sans-serif !important;
        font-weight: 500 !important;
        padding: 0.55rem 1.4rem !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 2px 12px var(--accent-glow) !important;
    }
    .stButton button:hover {
        filter: brightness(1.1) !important;
        box-shadow: 0 4px 20px var(--accent-glow) !important;
    }

    /* ── Confidence bar ── */
    .conf-track {
        background: var(--border);
        border-radius: 999px;
        height: 5px;
        margin-top: 0.4rem;
    }
    .conf-fill {
        height: 5px;
        border-radius: 999px;
        background: linear-gradient(90deg, var(--accent), var(--green));
    }

    /* ── Empty state ── */
    .empty-state {
        text-align: center;
        padding: 3.5rem 2rem;
        color: var(--text-muted);
    }
    .empty-state .icon { font-size: 2.5rem; margin-bottom: 1rem; }
    .empty-state p { font-size: 0.95rem; line-height: 1.6; }

    ::-webkit-scrollbar { width: 5px; height: 5px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: var(--border-accent); border-radius: 99px; }

    hr { border-color: var(--border) !important; margin: 1.5rem 0 !important; }

    [data-testid="stStatus"] {
        background: var(--bg-surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
    }
    [data-testid="metric-container"] {
        background: var(--bg-elevated) !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
        padding: 0.8rem 1rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ──────────────────────────────────────────────────────────────────────────────

_ANALYSIS_KEYS = [
    "brief", "summary", "insights", "risks", "analysis", "doc_meta",
    "suggested_questions", "doc_interpretation", "key_takeaway",
    "analysis_generated",
]


def _init_state() -> None:
    defaults: dict[str, Any] = {
        "qa_engine":           None,
        "summarizer":          None,
        "vector_db":           None,
        "pages":               [],
        "chunks":              [],
        "metadata":            {},
        "pdf_path":            None,
        "file_size":           0,
        "processing_time":     0.0,
        "total_chunks":        0,
        "document_loaded":     False,
        "analysis_generated":  False,
        "doc_interpretation":  "",
        "key_takeaway":        "",
        "brief":               "",
        "summary":             "",
        "insights":            "",
        "risks":               "",
        "analysis":            "",
        "doc_meta":            "",
        "suggested_questions": {},
        "question_input":      "",
        "last_answer":         None,
        "qa_cache":            {},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()

UPLOAD_DIR = "uploads"

# ──────────────────────────────────────────────────────────────────────────────
# HELPER — safe LLM call
# ──────────────────────────────────────────────────────────────────────────────


def _ask(prompt: str, fallback: str = "") -> str:
    """Run a prompt through QAEngine and return the answer string safely."""
    try:
        engine = st.session_state.get("qa_engine")
        if engine is None:
            return fallback
        result = engine.answer(prompt)
        if isinstance(result, dict):
            return result.get("answer") or result.get("text") or fallback
        if isinstance(result, str):
            return result
        return fallback
    except Exception as exc:
        logger.warning("_ask failed: %s", exc)
        return fallback


# ──────────────────────────────────────────────────────────────────────────────
# SECTION EXTRACTION — robust, case-insensitive, format-tolerant
# ──────────────────────────────────────────────────────────────────────────────

_SECTION_TAGS = [
    "INTERPRETATION", "TLDR", "KEY_TAKEAWAY", "SUMMARY",
    "INSIGHTS", "RISKS", "ANALYSIS", "METADATA", "QUESTIONS",
]


def _extract_section(text: str, tag: str) -> str:
    if not text or not tag:
        return ""

    try:
        def _tag_fragment(t: str) -> str:
            escaped = re.escape(t).replace(r"\_", r"[_ ]")
            return (
                r"(?:"
                r"(?:\*{1,2}|\#{1,3}\s*)?"
                r"[\[\(]?\s*"
                + escaped +
                r"\s*[\]\)]?"
                r"(?:\s*[:.\-])?"
                r"(?:\*{1,2})?"
                r")"
            )

        target_pattern = re.compile(
            r"^\s*" + _tag_fragment(tag) + r"\s*$",
            re.IGNORECASE | re.MULTILINE,
        )

        other_tags = [t for t in _SECTION_TAGS if t.upper() != tag.upper()]
        boundary_alts = "|".join(_tag_fragment(t) for t in other_tags)
        boundary_pattern = re.compile(
            r"^\s*(?:" + boundary_alts + r")\s*$",
            re.IGNORECASE | re.MULTILINE,
        )

        header_match = target_pattern.search(text)
        if not header_match:
            return ""

        content_start = header_match.end()
        next_section  = boundary_pattern.search(text, content_start)
        content_end   = next_section.start() if next_section else len(text)

        return text[content_start:content_end].strip()

    except Exception as exc:
        logger.warning("_extract_section failed for tag '%s': %s", tag, exc)
        return ""


# ──────────────────────────────────────────────────────────────────────────────
# QUESTION PARSER
# ──────────────────────────────────────────────────────────────────────────────

def _parse_questions(raw: str) -> dict[str, list[str]]:
    FALLBACK: dict[str, list[str]] = {
        "Understanding": [
            "What are the main findings of this document?",
            "What is the core objective of the research?",
        ],
        "Critical": [
            "What are the limitations of this study?",
            "Are there any biases or assumptions?",
        ],
        "Strategic": [
            "What are the real-world implications?",
            "How can this research be applied in practice?",
        ],
    }

    if not raw or not raw.strip():
        return FALLBACK

    try:
        groups: dict[str, list[str]] = {
            "Understanding": [],
            "Critical":      [],
            "Strategic":     [],
        }

        CATEGORY_ALIASES: dict[str, str] = {
            "understanding": "Understanding",
            "critical":      "Critical",
            "strategic":     "Strategic",
        }

        current_category: str | None = None

        for raw_line in raw.splitlines():
            line = raw_line.strip()
            if not line:
                continue

            inline_match = re.match(
                r"^[\[\(]?\s*(understanding|critical|strategic)\s*[\]\)]?\s*[:.]?\s*(.+)",
                line,
                re.IGNORECASE,
            )
            if inline_match:
                cat_key  = CATEGORY_ALIASES[inline_match.group(1).lower()]
                question = inline_match.group(2).strip()
                question = re.sub(r"^[\-\*\•\d\.\)\s]+", "", question).strip()
                if len(question) > 5 and len(groups[cat_key]) < 2:
                    groups[cat_key].append(question)
                current_category = cat_key
                continue

            header_candidate = re.sub(r"[\[\]()\:\.\-\*\•]", " ", line).strip().lower()
            header_candidate = re.sub(r"\s+", " ", header_candidate).strip()

            detected_category: str | None = None
            for alias, canonical in CATEGORY_ALIASES.items():
                if header_candidate == alias:
                    detected_category = canonical
                    break

            if detected_category:
                current_category = detected_category
                continue

            if current_category is None:
                continue

            cleaned = re.sub(r"^[\-\*\•\d\.\)\]\s]+", "", line).strip()
            if len(cleaned) > 5 and len(groups[current_category]) < 2:
                groups[current_category].append(cleaned)

        if not any(groups.values()):
            return FALLBACK

        for cat in groups:
            if not groups[cat]:
                groups[cat] = FALLBACK[cat]

        return groups

    except Exception:
        return FALLBACK


# ──────────────────────────────────────────────────────────────────────────────
# ANALYSIS PROMPT
# ──────────────────────────────────────────────────────────────────────────────

def _build_analysis_prompt() -> str:
    return (
        "Analyze this document thoroughly and return a structured response with ALL of the "
        "following sections in EXACT order, each preceded by its tag in square brackets. "
        "Do not skip any section. Be specific to this document's actual content.\n\n"

        "[INTERPRETATION]\n"
        "One sentence describing what this document focuses on, what it primarily addresses, "
        "and its broader implications. Structure: 'This document focuses on X, primarily "
        "addressing Y, with implications in Z.' Be specific — do not generalise.\n\n"

        "[TLDR]\n"
        "2–3 lines TL;DR in plain, accessible language. Capture the essential purpose and "
        "outcome. Do not repeat the interpretation above.\n\n"

        "[KEY_TAKEAWAY]\n"
        "The single most important insight or conclusion a decision-maker should remember. "
        "One direct, actionable sentence. Be specific and sharp — avoid vague statements.\n\n"

        "[SUMMARY]\n"
        "Clear executive summary in 5–8 sentences. Cover the document's purpose, core "
        "arguments, key evidence, and conclusions. Professional, analytical tone.\n\n"

        "[INSIGHTS]\n"
        "Most significant findings as concise bullet points. Each point introduces a distinct "
        "idea not already in the summary. Be specific to this document's content.\n\n"

        "[RISKS]\n"
        "Specific limitations, risks, weaknesses, or gaps in this document. Focus on what is "
        "missing, potentially flawed, or worth questioning. Do not repeat insights.\n\n"

        "[ANALYSIS]\n"
        "Structured analytical breakdown covering: methodology used, key data or evidence "
        "presented, core findings, and broader implications. Write in cohesive analytical "
        "voice — not as a list.\n\n"

        "[METADATA]\n"
        "Identify: (1) document type or genre, (2) estimated reading time, "
        "(3) complexity level — one of: Basic / Intermediate / Advanced / Expert. "
        "Format as short labelled lines. Be concise.\n\n"

        "[QUESTIONS]\n"
        "Generate exactly 6 highly specific questions a researcher or decision-maker would "
        "want answered about THIS document's actual content. Label each with its category "
        "using EXACT format, one per line:\n"
        "[Understanding] Question text\n"
        "[Critical] Question text\n"
        "[Strategic] Question text\n"
        "Provide exactly 2 questions per category. Reference specific topics, findings, or "
        "claims in this document. No numbering or bullets."
    )


# ──────────────────────────────────────────────────────────────────────────────
# ANALYSIS GENERATION — single LLM call, fully cached
# ──────────────────────────────────────────────────────────────────────────────

def _generate_document_intelligence() -> None:
    if st.session_state.get("analysis_generated"):
        return
    if not st.session_state.get("document_loaded"):
        return

    with st.spinner("Analysing document..."):
        try:
            raw_output = _ask(_build_analysis_prompt(), fallback="")
        except Exception as exc:
            logger.error("Single-call analysis failed: %s", exc)
            raw_output = ""

        if not raw_output:
            st.session_state["doc_interpretation"]  = ""
            st.session_state["brief"]               = "Summary not available."
            st.session_state["key_takeaway"]        = ""
            st.session_state["summary"]             = "Executive summary not available."
            st.session_state["insights"]            = "Key insights could not be extracted from this document."
            st.session_state["risks"]               = "Risk analysis could not be extracted from this document."
            st.session_state["analysis"]            = "Analytical breakdown not available."
            st.session_state["doc_meta"]            = "Metadata not available."
            st.session_state["suggested_questions"] = {"Understanding": [], "Critical": [], "Strategic": []}
            st.session_state["analysis_generated"]  = True
            return

        st.session_state["doc_interpretation"] = _extract_section(raw_output, "INTERPRETATION")
        st.session_state["brief"]              = _extract_section(raw_output, "TLDR") or "Summary not available."
        st.session_state["key_takeaway"]       = _extract_section(raw_output, "KEY_TAKEAWAY")
        st.session_state["summary"]            = _extract_section(raw_output, "SUMMARY") or "Executive summary not available."
        st.session_state["insights"]           = _extract_section(raw_output, "INSIGHTS") or "Key insights could not be extracted from this document."
        st.session_state["risks"]              = _extract_section(raw_output, "RISKS") or "Risk analysis could not be extracted from this document."
        st.session_state["analysis"]           = _extract_section(raw_output, "ANALYSIS") or "Analytical breakdown not available."
        st.session_state["doc_meta"]           = _extract_section(raw_output, "METADATA") or "Metadata not available."

        raw_questions = _extract_section(raw_output, "QUESTIONS")
        st.session_state["suggested_questions"] = _parse_questions(raw_questions)

    st.session_state["analysis_generated"] = True


# ──────────────────────────────────────────────────────────────────────────────
# CACHED QA
# ──────────────────────────────────────────────────────────────────────────────

def _cached_ask_qa(engine: Any, prompt: str) -> Any:
    doc_name  = st.session_state.get("_uploaded_name", "")
    cache_key = f"{doc_name}::{prompt}"
    cache: dict = st.session_state.setdefault("qa_cache", {})

    if cache_key in cache:
        return cache[cache_key]

    try:
        result = engine.answer(prompt)
        cache[cache_key] = result
        return result
    except Exception as exc:
        logger.warning("QA engine call failed: %s", exc)
        return {"error": str(exc)}


# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        """
        <div style="padding:1rem 0 1.5rem;">
            <div style="font-family:'DM Serif Display',serif;font-size:1.5rem;
                        color:#e8eaf0;line-height:1.2;">DocIQ</div>
            <div style="font-family:'DM Mono',monospace;font-size:0.65rem;
                        letter-spacing:0.14em;text-transform:uppercase;
                        color:#4f8ef7;margin-top:0.2rem;">AI Research Analyst</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    st.markdown("**Upload Document**")
    uploaded_file = st.file_uploader(
        label="PDF file",
        type=["pdf"],
        label_visibility="collapsed",
        help="Upload a PDF to begin analysis",
    )

    if uploaded_file is not None:
        current_name = uploaded_file.name
        prev_name    = st.session_state.get("_uploaded_name", "")

        if current_name != prev_name:
            st.session_state["_uploaded_name"] = current_name

            for key in _ANALYSIS_KEYS:
                if key == "analysis_generated":
                    st.session_state[key] = False
                elif key == "suggested_questions":
                    st.session_state[key] = {}
                else:
                    st.session_state[key] = ""

            st.session_state["document_loaded"] = False
            st.session_state["last_answer"]     = None
            st.session_state["question_input"]  = ""
            st.session_state["qa_cache"]        = {}

            try:
                result = run_pipeline(uploaded_file, UPLOAD_DIR)
                st.session_state["qa_engine"]       = result["qa_engine"]
                st.session_state["summarizer"]      = result["summarizer"]
                st.session_state["vector_db"]       = result["vector_db"]
                st.session_state["pages"]           = result["pages"]
                st.session_state["chunks"]          = result["chunks"]
                st.session_state["metadata"]        = result["metadata"]
                st.session_state["pdf_path"]        = result["pdf_path"]
                st.session_state["file_size"]       = result["file_size"]
                st.session_state["processing_time"] = result["processing_time"]
                st.session_state["total_chunks"]    = result["total_chunks"]
                st.session_state["document_loaded"] = True
                st.rerun()
            except Exception as exc:
                st.error(f"Pipeline error: {exc}", icon="🚨")

    st.divider()

    if st.session_state.get("document_loaded"):
        file_size    = st.session_state.get("file_size", 0)
        proc_time    = st.session_state.get("processing_time", 0)
        total_chunks = st.session_state.get("total_chunks", 0)
        pages_list   = st.session_state.get("pages", [])
        doc_name     = st.session_state.get("_uploaded_name", "Document")

        st.markdown(f"**{doc_name[:28]}{'...' if len(doc_name) > 28 else ''}**")
        st.caption(
            f"{len(pages_list)} pages · {total_chunks} chunks · "
            f"{round(file_size / 1024, 1)} KB · {proc_time}s"
        )
        if st.button("Clear document", use_container_width=True):
            for key in _ANALYSIS_KEYS + [
                "qa_engine", "summarizer", "vector_db", "pages", "chunks",
                "metadata", "pdf_path", "file_size", "processing_time",
                "total_chunks", "document_loaded", "last_answer",
                "_uploaded_name", "question_input", "qa_cache",
            ]:
                st.session_state.pop(key, None)
            st.rerun()
    else:
        st.markdown(
            "<p style='color:var(--text-muted,#666);font-size:0.82rem;'>No document loaded.</p>",
            unsafe_allow_html=True,
        )

    st.divider()
    st.caption("DocIQ v2.3 · Enterprise Edition")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN HEADER
# ──────────────────────────────────────────────────────────────────────────────

st.markdown(
    """
    <h1 style="margin-bottom:0.25rem;">Document Intelligence</h1>
    <p style="color:#8b91a8;font-size:0.92rem;margin-top:0;margin-bottom:1.5rem;">
        Upload a PDF to unlock structured AI analysis, insights, and expert Q&amp;A.
    </p>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────────────────────
# GUARD — no document
# ──────────────────────────────────────────────────────────────────────────────

if not st.session_state.get("document_loaded"):
    st.markdown(
        """
        <div class="empty-state">
            <div class="icon">[ ]</div>
            <p><strong style="color:#e8eaf0;">No document loaded</strong><br>
            Upload a PDF from the sidebar to begin your analysis session.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

# ──────────────────────────────────────────────────────────────────────────────
# TRIGGER ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────

_generate_document_intelligence()

# ──────────────────────────────────────────────────────────────────────────────
# TABS
# ──────────────────────────────────────────────────────────────────────────────

tab_intel, tab_qa = st.tabs(["Document Intelligence", "Decision & Q&A"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DOCUMENT INTELLIGENCE
# ══════════════════════════════════════════════════════════════════════════════

with tab_intel:

    interpretation = st.session_state.get("doc_interpretation", "")
    if interpretation:
        st.markdown(
            f"""
            <div class="intel-header">
                <div class="intel-label">AI Interpretation</div>
                <p>{interpretation}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    brief = st.session_state.get("brief", "")
    if brief:
        st.markdown(
            f"""
            <div class="tldr-banner">
                <div class="label">TL;DR</div>
                <p>{brief}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    takeaway = st.session_state.get("key_takeaway", "")
    if takeaway:
        st.markdown(
            f"""
            <div class="takeaway-box">
                <strong>Key Takeaway:</strong>&nbsp;&nbsp;{takeaway}
            </div>
            """,
            unsafe_allow_html=True,
        )

    col_sum, col_meta = st.columns([3, 2], gap="large")

    with col_sum:
        st.markdown(
            """
            <div class="iq-card iq-card-accent">
                <div class="section-label">Executive Summary</div>
            """,
            unsafe_allow_html=True,
        )
        summary = st.session_state.get("summary", "")
        st.markdown(
            f"<p style='color:#c8cdd8;line-height:1.75;font-size:0.92rem;margin:0;'>"
            f"{summary if summary else 'Executive summary could not be generated for this document.'}</p>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col_meta:
        st.markdown(
            """
            <div class="iq-card iq-card-purple">
                <div class="section-label">Document Metadata</div>
            """,
            unsafe_allow_html=True,
        )
        doc_meta = st.session_state.get("doc_meta", "")
        if doc_meta:
            for line in doc_meta.splitlines():
                line = line.strip()
                if line:
                    st.markdown(
                        f"<p style='color:#c8cdd8;font-size:0.85rem;margin:0.2rem 0;'>{line}</p>",
                        unsafe_allow_html=True,
                    )
        else:
            st.markdown(
                "<p style='color:#555b72;font-size:0.85rem;'>Metadata could not be determined.</p>",
                unsafe_allow_html=True,
            )

        pages_count  = len(st.session_state.get("pages", []))
        total_chunks = st.session_state.get("total_chunks", 0)
        proc_time    = st.session_state.get("processing_time", 0)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div>
                <span class="meta-pill">Pages <span>{pages_count}</span></span>
                <span class="meta-pill">Chunks <span>{total_chunks}</span></span>
                <span class="meta-pill"><span>{proc_time}s</span></span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="narrative-connector">
            <span>The following insights are derived from this summary.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="iq-card iq-card-green">
            <div class="section-label">Key Insights</div>
        """,
        unsafe_allow_html=True,
    )
    insights = st.session_state.get("insights", "")
    insight_lines = [l.strip().lstrip("•-*").strip() for l in insights.splitlines() if l.strip().lstrip("•-*").strip()]
    if insight_lines:
        for line in insight_lines:
            st.markdown(
                f"<p style='color:#c8cdd8;font-size:0.9rem;margin:0.3rem 0;'>"
                f"<span style='color:#3ecf8e;margin-right:0.5rem;'>+</span>{line}</p>",
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            "<p style='color:#555b72;font-size:0.88rem;'>Key insights could not be extracted from this document.</p>",
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="narrative-connector">
            <span>These insights surface the following risks and limitations.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="iq-card iq-card-red">
            <div class="section-label">Risks &amp; Limitations</div>
        """,
        unsafe_allow_html=True,
    )
    risks = st.session_state.get("risks", "")
    risk_lines = [l.strip().lstrip("•-*").strip() for l in risks.splitlines() if l.strip().lstrip("•-*").strip()]
    if risk_lines:
        for line in risk_lines:
            st.markdown(
                f"<p style='color:#c8cdd8;font-size:0.9rem;margin:0.3rem 0;'>"
                f"<span style='color:#f56565;margin-right:0.5rem;'>!</span>{line}</p>",
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            "<p style='color:#555b72;font-size:0.88rem;'>No significant risks or limitations could be identified from this document.</p>",
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("Analytical Breakdown", expanded=False):
        analysis_text = st.session_state.get("analysis", "")
        if analysis_text:
            st.markdown(
                f"<div style='color:#c8cdd8;font-size:0.9rem;line-height:1.75;'>"
                f"{analysis_text}</div>",
                unsafe_allow_html=True,
            )
        else:
            st.caption("Analytical breakdown could not be generated for this document.")

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Re-analyse Document", use_container_width=False):
        for key in [
            "brief", "summary", "insights", "risks", "analysis",
            "doc_meta", "doc_interpretation", "key_takeaway",
            "suggested_questions", "analysis_generated",
        ]:
            st.session_state[key] = {} if key == "suggested_questions" else (False if key == "analysis_generated" else "")
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — DECISION & Q&A
# ══════════════════════════════════════════════════════════════════════════════

with tab_qa:

    qa_engine = st.session_state.get("qa_engine")

    if qa_engine is None:
        st.markdown(
            """
            <div class="empty-state">
                <div class="icon">[ ]</div>
                <p>QA Engine not initialised. Please re-upload your document.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.stop()

    # ── Grouped Suggested Questions ───────────────────────────────────────────
    suggested: dict[str, list[str]] = st.session_state.get("suggested_questions", {})

    _GROUP_LABELS: dict[str, str] = {
        "Understanding": "Understanding",
        "Critical":      "Critical",
        "Strategic":     "Strategic",
    }

    if suggested:
        st.markdown(
            "<div class='section-label' style='margin-bottom:0.25rem;'>Suggested Questions</div>",
            unsafe_allow_html=True,
        )
        cols = st.columns(3, gap="medium")
        for col_idx, (cat, questions) in enumerate(suggested.items()):
            cat_norm      = cat.strip().rstrip(":")
            display_label = _GROUP_LABELS.get(cat_norm, cat_norm)

            with cols[col_idx]:
                st.markdown(
                    f"<div class='sq-group-header'>{display_label}</div>",
                    unsafe_allow_html=True,
                )
                for q in questions:
                    if st.button(q, key=f"sq_{cat_norm}_{q[:35]}", use_container_width=True):
                        # Write directly into the textarea's session_state key.
                        # Because this executes before the text_area widget renders
                        # below, Streamlit picks up the new value in the same rerun
                        # — no extra st.rerun() or flag needed.
                        st.session_state["question_input"] = q

        st.markdown("<br>", unsafe_allow_html=True)

    # ── Explanation Mode ──────────────────────────────────────────────────────
    st.markdown("<div class='section-label'>Explanation Mode</div>", unsafe_allow_html=True)
    mode = st.radio(
        label="Mode",
        options=["Standard", "Beginner", "Technical", "Critical Analysis"],
        horizontal=True,
        label_visibility="collapsed",
    )

    _MODE_META: dict[str, tuple[str, str]] = {
        "Standard": (
            "",
            "Balanced answer — clear, concise, and directly relevant to the document.",
        ),
        "Beginner": (
            " Explain your answer in simple terms as if to a non-expert.",
            "Simplified explanation — avoids jargon and uses plain language.",
        ),
        "Technical": (
            " Provide a detailed, technical-level answer with precise terminology.",
            "Expert-level detail — includes technical depth, precise terminology, and methodology.",
        ),
        "Critical Analysis": (
            " Critically analyse the answer, noting assumptions, limitations, and alternative interpretations.",
            "Evaluates strengths, weaknesses, assumptions, and alternative interpretations.",
        ),
    }

    mode_entry = _MODE_META.get(mode)
    if isinstance(mode_entry, (list, tuple)) and len(mode_entry) >= 2:
        mode_suffix = mode_entry[0]
        mode_desc   = mode_entry[1]
    else:
        mode_suffix = ""
        mode_desc   = "Balanced answer — clear, concise, and directly relevant to the document."

    st.markdown(
        f"<div class='mode-explanation'>{mode_desc}</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Question Input ────────────────────────────────────────────────────────
    st.markdown("<div class='section-label'>Ask a Question</div>", unsafe_allow_html=True)

    # key="question_input" binds this widget to session_state directly.
    # Suggested question buttons write to st.session_state["question_input"]
    # before this widget renders, so the value is always current with no
    # extra rerun, prefill flag, or indirection required.
    st.text_area(
        label="Question",
        placeholder="What would you like to know about this document?",
        height=90,
        label_visibility="collapsed",
        key="question_input",
    )

    col_ask, col_clear = st.columns([1, 5])
    with col_ask:
        ask_clicked = st.button("Ask", use_container_width=True)
    with col_clear:
        if st.button("Clear", use_container_width=False):
            st.session_state["question_input"] = ""
            st.session_state["last_answer"]    = None
            st.rerun()

    # ── Answer Generation ─────────────────────────────────────────────────────
    # Always read from session_state — stays in sync regardless of how the
    # textarea was populated (typed or injected by a suggested question button).
    user_question = st.session_state.get("question_input", "")

    if ask_clicked and user_question.strip():
        full_prompt = user_question.strip() + mode_suffix
        start_time  = time.time()

        with st.status("Processing query...", expanded=False) as status:
            status.update(label="Retrieving relevant context...", state="running")
            raw_result = _cached_ask_qa(qa_engine, full_prompt)
            status.update(label="Generating response...", state="running")
            status.update(label="Completed", state="complete")

        response_time = round(time.time() - start_time, 2)
        st.session_state["last_answer"] = raw_result
        if not (isinstance(raw_result, dict) and "error" in raw_result):
            st.success(f"Answer generated in {response_time}s")

    # ── Answer Panel / First-time Guidance ───────────────────────────────────
    last = st.session_state.get("last_answer")

    if last is None:
        st.markdown(
            """
            <div class="guidance-box">
                <strong>Ready to explore this document</strong><br>
                Start by selecting a suggested question above, or enter your own to begin.<br>
                <span style="font-size:0.82rem;opacity:0.7;">
                    Your answer will appear here, grounded in the document's content.
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    elif isinstance(last, dict) and "error" in last:
        st.error(f"Error: {last['error']}")

    else:
        if isinstance(last, dict):
            answer_text = last.get("answer") or last.get("text") or str(last)
            sources     = last.get("sources") or last.get("source_chunks") or []
            confidence  = last.get("confidence") or last.get("score")
            explanation = last.get("explanation") or last.get("reasoning") or ""
        else:
            answer_text = str(last)
            sources     = []
            confidence  = None
            explanation = ""

        st.markdown(
            f"""
            <div class="answer-panel">
                <div class="section-label">Answer</div>
                <div style="color:#e8eaf0;font-size:0.95rem;line-height:1.8;margin-top:0.5rem;">
                    {answer_text}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        try:
            st.caption(f"Response time: {response_time}s")
        except NameError:
            pass

        n_sources = len(sources) if sources else 0
        grounding_msg = (
            f"This answer is grounded in <strong>{n_sources}</strong> "
            f"high-relevance document section{'s' if n_sources != 1 else ''}."
            if n_sources > 0
            else "Answer generated from the full document context."
        )

        st.markdown(
            f"""
            <div class="grounding-bar">
                <div class="dot"></div>
                <span>{grounding_msg}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if confidence is not None:
            try:
                conf_val = float(confidence)
                conf_pct = min(int(conf_val * 100) if conf_val <= 1.0 else int(conf_val), 100)
                st.markdown(
                    f"""
                    <div style="margin-top:0.75rem;">
                        <span style="font-size:0.78rem;color:#8b91a8;">
                            Confidence: <strong style="color:#e8eaf0;">{conf_pct}%</strong>
                        </span>
                        <div class="conf-track">
                            <div class="conf-fill" style="width:{conf_pct}%;"></div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            except (TypeError, ValueError):
                pass

        if sources:
            with st.expander(f"Sources ({n_sources})", expanded=False):
                for i, src in enumerate(sources, 1):
                    if isinstance(src, dict):
                        page    = src.get("page") or src.get("page_num") or "?"
                        excerpt = src.get("text") or src.get("content") or src.get("chunk", "")
                        if excerpt:
                            excerpt = excerpt[:320] + ("..." if len(excerpt) > 320 else "")
                        st.markdown(
                            f"<p style='font-size:0.82rem;color:#8b91a8;margin:0.4rem 0;'>"
                            f"<strong style='color:#c8cdd8;'>Source {i}</strong>"
                            f"{' · Page ' + str(page) if page != '?' else ''}</p>"
                            f"<p style='font-size:0.82rem;color:#6b7290;margin:0 0 0.75rem;'>"
                            f"{excerpt}</p>",
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f"<p style='font-size:0.82rem;color:#6b7290;'>{str(src)[:320]}</p>",
                            unsafe_allow_html=True,
                        )

        if explanation:
            with st.expander("Reasoning / Explainability", expanded=False):
                st.markdown(
                    f"<p style='font-size:0.85rem;color:#8b91a8;line-height:1.7;'>"
                    f"{explanation}</p>",
                    unsafe_allow_html=True,
                )