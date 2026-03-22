# ── CSS injection ────────────────────────────────────────────────────────────

import streamlit as st

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

/* ── Global reset ── */
html, body, [data-testid="stAppViewContainer"] {
    background: #0a0a0f !important;
    color: #e8e4dc !important;
    font-family: 'Syne', sans-serif;
}
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stSidebar"] { display: none; }

/* ── Typography ── */
h1 { font-family: 'Syne', sans-serif; font-weight: 800; letter-spacing: -0.03em; }
h2, h3 { font-family: 'Syne', sans-serif; font-weight: 600; }
code, .mono { font-family: 'Space Mono', monospace; }

/* ── Hero header ── */
.hero {
    text-align: center;
    padding: 3rem 0 1.5rem;
}
.hero-eyebrow {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: #7c6af0;
    margin-bottom: 0.75rem;
}
.hero-title {
    font-size: clamp(2.2rem, 6vw, 3.8rem);
    font-weight: 800;
    line-height: 1.05;
    background: linear-gradient(135deg, #e8e4dc 30%, #7c6af0 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.5rem;
}
.hero-sub {
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    color: #6b6878;
    letter-spacing: 0.05em;
}

/* ── Upload zone ── */
[data-testid="stFileUploader"] {
    background: #13121a !important;
    border: 1.5px dashed #2e2b3e !important;
    border-radius: 16px !important;
    transition: border-color 0.25s;
}
[data-testid="stFileUploader"]:hover { border-color: #7c6af0 !important; }
[data-testid="stFileUploaderDropzoneInstructions"] {
    color: #6b6878 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.8rem !important;
}

/* ── Image preview card ── */
.img-card {
    background: #13121a;
    border: 1px solid #1f1d2e;
    border-radius: 16px;
    padding: 1rem;
    text-align: center;
    margin-bottom: 1.2rem;
}
.img-card img { border-radius: 10px; max-height: 280px; }
.img-meta {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: #4a4760;
    margin-top: 0.6rem;
}

/* ── Predict button ── */
[data-testid="stButton"] > button {
    width: 100%;
    background: linear-gradient(135deg, #7c6af0, #5648c4) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 12px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    letter-spacing: 0.06em !important;
    padding: 0.75rem 1.5rem !important;
    transition: opacity 0.2s, transform 0.15s !important;
    cursor: pointer !important;
}
[data-testid="stButton"] > button:hover {
    opacity: 0.88 !important;
    transform: translateY(-1px) !important;
}
[data-testid="stButton"] > button:active { transform: translateY(0) !important; }

/* ── Result card ── */
.result-card {
    background: #13121a;
    border: 1px solid #1f1d2e;
    border-radius: 20px;
    padding: 2rem 2rem 1.5rem;
    margin-top: 1.5rem;
    position: relative;
    overflow: hidden;
}
.result-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #7c6af0, #a78bfa, #7c6af0);
    background-size: 200% 100%;
    animation: shimmer 2.5s infinite linear;
}
@keyframes shimmer { to { background-position: 200% 0; } }

.result-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #7c6af0;
    margin-bottom: 0.4rem;
}
.result-class {
    font-family: 'Syne', sans-serif;
    font-size: 2.4rem;
    font-weight: 800;
    text-transform: capitalize;
    color: #e8e4dc;
    margin: 0;
    letter-spacing: -0.02em;
}
.result-super {
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    color: #6b6878;
    margin-top: 0.25rem;
}
.confidence-row {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-top: 1.2rem;
}
.conf-bar-wrap {
    flex: 1; height: 6px;
    background: #1f1d2e;
    border-radius: 3px;
    overflow: hidden;
}
.conf-bar-fill {
    height: 100%;
    border-radius: 3px;
    background: linear-gradient(90deg, #7c6af0, #a78bfa);
}
.conf-pct {
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    font-weight: 700;
    color: #a78bfa;
    white-space: nowrap;
}

/* ── Top-5 table ── */
.top5-header {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #4a4760;
    margin: 1.6rem 0 0.8rem;
    border-top: 1px solid #1f1d2e;
    padding-top: 1.2rem;
}
.top5-row {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.4rem 0;
    border-bottom: 1px solid #16151f;
}
.top5-rank {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    color: #4a4760;
    width: 1.2rem;
    text-align: right;
    flex-shrink: 0;
}
.top5-name {
    font-family: 'Syne', sans-serif;
    font-size: 0.9rem;
    font-weight: 600;
    text-transform: capitalize;
    color: #c8c3b8;
    flex: 1;
}
.top5-name.top { color: #e8e4dc; }
.top5-bar-wrap {
    width: 110px; height: 4px;
    background: #1f1d2e;
    border-radius: 2px;
    overflow: hidden;
}
.top5-bar-fill { height: 100%; border-radius: 2px; }
.top5-pct {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: #4a4760;
    width: 3.2rem;
    text-align: right;
    flex-shrink: 0;
}
.top5-pct.top { color: #7c6af0; }

/* ── Error / unrecognized ── */
.error-card {
    background: #16100f;
    border: 1px solid #3d1f1a;
    border-radius: 16px;
    padding: 1.8rem;
    text-align: center;
    margin-top: 1.5rem;
}
.error-icon { font-size: 2.5rem; }
.error-title {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 1.3rem;
    color: #e87c6a;
    margin: 0.5rem 0 0.3rem;
}
.error-sub {
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    color: #6b4a44;
}

/* ── Info chips ── */
.chips { display: flex; gap: 0.5rem; flex-wrap: wrap; margin-top: 0.8rem; }
.chip {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.08em;
    background: #1a1828;
    border: 1px solid #2e2b3e;
    color: #6b6878;
    border-radius: 6px;
    padding: 0.25rem 0.6rem;
}

/* ── Section divider ── */
.section-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #2e2b3e, transparent);
    margin: 2rem 0;
}

/* ── Model warning ── */
.model-warn {
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    background: #14120f;
    border: 1px solid #3d3010;
    border-radius: 10px;
    color: #c49a3c;
    padding: 0.8rem 1rem;
    text-align: center;
    margin-bottom: 1rem;
}

/* ── Streamlit misc ── */
[data-testid="stDecoration"] { display: none; }
.stSpinner > div { border-top-color: #7c6af0 !important; }
</style>
"""


def inject_styles() -> None:
    """Call once at app startup to inject all custom CSS."""
    st.markdown(_CSS, unsafe_allow_html=True)