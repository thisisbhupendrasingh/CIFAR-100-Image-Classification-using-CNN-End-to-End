"""
CIFAR-100 Image Classification App
Entry point — orchestrates UI flow only; all logic lives in submodules.

Run with:
    streamlit run main.py
"""

import io
import os

import streamlit as st
from PIL import Image

from config import CONFIDENCE_THRESHOLD, MODEL_PATH, SUPPORTED_FORMATS
from styles import inject_styles
from model import load_model, top_k_results
from components import (
    render_hero,
    render_divider,
    render_footer,
    render_upload_hint,
    render_model_warning,
    render_image_preview,
    render_model_unavailable,
    render_unrecognized,
    render_result,
)

import warnings
warnings.filterwarnings("ignore")

# ── Page config (must be the very first Streamlit call) ──────────────────────
st.set_page_config(
    page_title="CIFAR-100 Classifier",
    page_icon="🔬",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Styles ───────────────────────────────────────────────────────────────────
inject_styles()

# ── Hero ─────────────────────────────────────────────────────────────────────
render_hero()

# ── Model loading ─────────────────────────────────────────────────────────────
if not os.path.exists(MODEL_PATH):
    render_model_warning(MODEL_PATH)
    model = None
else:
    with st.spinner("Loading model weights…"):
        model, model_err = load_model(MODEL_PATH)
    if model_err:
        st.error(f"Failed to load model: {model_err}")
        model = None

# ── Upload section ────────────────────────────────────────────────────────────
render_divider()

uploaded = st.file_uploader(
    "Drop an image here",
    type=SUPPORTED_FORMATS,
    label_visibility="collapsed",
)

if uploaded:
    img      = Image.open(io.BytesIO(uploaded.read()))
    size_kb  = uploaded.size / 1024

    render_image_preview(img, uploaded.name, size_kb)

    if st.button("⚡  Predict", use_container_width=True):
        if model is None:
            render_model_unavailable()
        else:
            with st.spinner("Running inference…"):
                top5 = top_k_results(model, img)

            top_class, top_conf = top5[0]

            if top_conf < CONFIDENCE_THRESHOLD:
                render_unrecognized(top_class, top_conf)
            else:
                render_result(top5)

else:
    render_upload_hint()

# ── Footer ────────────────────────────────────────────────────────────────────
render_divider()
render_footer()