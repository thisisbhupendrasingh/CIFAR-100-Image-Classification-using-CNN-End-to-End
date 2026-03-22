# ── Reusable UI components ───────────────────────────────────────────────────

import streamlit as st
from PIL import Image

from classes import CLASS_TO_SUPER
from config import BAR_COLOURS, CONFIDENCE_THRESHOLD


# ── Static layout pieces ─────────────────────────────────────────────────────

def render_hero() -> None:
    """Top hero section with title and subtitle."""
    st.markdown("""
    <div class="hero">
        <div class="hero-eyebrow">Deep Learning · Computer Vision</div>
        <h1 class="hero-title">CIFAR-100<br>Classifier</h1>
        <div class="hero-sub">100 classes · 32 × 32 input · Keras / TensorFlow</div>
    </div>
    """, unsafe_allow_html=True)


def render_divider() -> None:
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


def render_footer() -> None:
    st.markdown("""
    <div style="text-align:center;padding-bottom:2rem;">
        <span style="font-family:'Space Mono',monospace;font-size:0.65rem;color:#2e2b3e;
                     letter-spacing:0.1em;">CIFAR-100 · 100 classes across 20 superclasses</span>
    </div>
    """, unsafe_allow_html=True)


def render_upload_hint() -> None:
    """Shown when no file has been uploaded yet."""
    st.markdown("""
    <div style="text-align:center;padding:1.5rem 0 0.5rem;">
        <span style="font-family:'Space Mono',monospace;font-size:0.75rem;color:#2e2b3e;
                     letter-spacing:0.1em;">upload an image to begin</span>
    </div>
    """, unsafe_allow_html=True)


# ── Model state ──────────────────────────────────────────────────────────────

def render_model_warning(model_path: str) -> None:
    """Banner shown when the .h5 file is missing from disk."""
    st.markdown(
        f'<div class="model-warn">⚠ &nbsp;<code>{model_path}</code> not found in working directory. '
        "Place it alongside this script to enable live inference.</div>",
        unsafe_allow_html=True,
    )


# ── Image preview ────────────────────────────────────────────────────────────

def render_image_preview(img: Image.Image, filename: str, size_kb: float) -> None:
    """Card showing the uploaded image with its metadata."""
    w, h = img.size
    st.markdown('<div class="img-card">', unsafe_allow_html=True)
    st.image(img, use_container_width=False, width=min(w, 380))
    st.markdown(
        f'<div class="img-meta">{filename} &nbsp;·&nbsp; {w}×{h}px &nbsp;·&nbsp; {size_kb:.1f} KB</div>',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


# ── Error states ─────────────────────────────────────────────────────────────

def render_model_unavailable() -> None:
    """Error card when the model object is None at inference time."""
    st.markdown(
        '<div class="error-card">'
        '<div class="error-icon">🔌</div>'
        '<div class="error-title">Model Unavailable</div>'
        '<div class="error-sub">Cannot run inference — see the warning above.</div>'
        "</div>",
        unsafe_allow_html=True,
    )


def render_unrecognized(top_class: str, top_conf: float) -> None:
    """Error card shown when confidence is below the threshold."""
    st.markdown(
        '<div class="error-card">'
        '<div class="error-icon">🤷</div>'
        '<div class="error-title">Unrecognized Image</div>'
        f'<div class="error-sub">Best guess: <b>{top_class}</b> at {top_conf*100:.1f}% — '
        f"below {CONFIDENCE_THRESHOLD*100:.0f}% threshold.<br>"
        "Try a clearer or more centred photo.</div>"
        "</div>",
        unsafe_allow_html=True,
    )


# ── Prediction result ────────────────────────────────────────────────────────

def _build_top5_rows(top5: list[tuple[str, float]]) -> str:
    """Return the inner HTML for the top-5 predictions table."""
    max_conf = top5[0][1]
    rows = ""
    for rank, (cls, conf) in enumerate(top5):
        is_top   = rank == 0
        rel_w    = int(conf / max_conf * 100)
        colour   = BAR_COLOURS[rank]
        name_cls = "top5-name top" if is_top else "top5-name"
        pct_cls  = "top5-pct top"  if is_top else "top5-pct"
        rows += f"""
        <div class="top5-row">
            <span class="top5-rank">#{rank+1}</span>
            <span class="{name_cls}">{cls.replace('_', ' ')}</span>
            <div class="top5-bar-wrap">
                <div class="top5-bar-fill" style="width:{rel_w}%;background:{colour};"></div>
            </div>
            <span class="{pct_cls}">{conf*100:.1f}%</span>
        </div>"""
    return rows


def render_result(top5: list[tuple[str, float]]) -> None:
    """
    Full result card: predicted class, superclass, confidence bar, top-5 table.

    Parameters
    ----------
    top5 : list of (class_name, probability), length ≤ TOP_K, sorted desc.
    """
    top_class, top_conf = top5[0]
    super_label = CLASS_TO_SUPER.get(top_class, "—")
    pct_int     = int(top_conf * 100)
    bar_w       = min(pct_int, 100)
    top5_rows   = _build_top5_rows(top5)

    st.markdown(f"""
    <div class="result-card">
        <div class="result-label">Predicted class</div>
        <div class="result-class">{top_class.replace('_', ' ')}</div>
        <div class="result-super">Superclass &nbsp;·&nbsp; {super_label}</div>
        <div class="confidence-row">
            <div class="conf-bar-wrap">
                <div class="conf-bar-fill" style="width:{bar_w}%;"></div>
            </div>
            <span class="conf-pct">{pct_int}% confidence</span>
        </div>
        <div class="top5-header">Top-5 predictions</div>
        {top5_rows}
        <div class="chips">
            <span class="chip">input: 32×32 px</span>
            <span class="chip">classes: 100</span>
            <span class="chip">threshold: {CONFIDENCE_THRESHOLD*100:.0f}%</span>
        </div>
    </div>
    """, unsafe_allow_html=True)