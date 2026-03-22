
"""
CIFAR-100 Image Classification App
A polished Streamlit UI for classifying images using a trained CIFAR-100 model.
"""

import streamlit as st
import numpy as np
from PIL import Image
import io
import os

# ── Page config (must be first Streamlit call) ──────────────────────────────
st.set_page_config(
    page_title="CIFAR-100 Classifier",
    page_icon="🔬",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── CIFAR-100 class names ────────────────────────────────────────────────────
CIFAR100_CLASSES = [
    "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle",
    "bicycle", "bottle", "bowl", "boy", "bridge", "bus", "butterfly", "camel",
    "can", "castle", "caterpillar", "cattle", "chair", "chimpanzee", "clock",
    "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur",
    "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster",
    "house", "kangaroo", "keyboard", "lamp", "lawn_mower", "leopard", "lion",
    "lizard", "lobster", "man", "maple_tree", "motorcycle", "mountain",
    "mouse", "mushroom", "oak_tree", "orange", "orchid", "otter", "palm_tree",
    "pear", "pickup_truck", "pine_tree", "plain", "plate", "poppy",
    "porcupine", "possum", "rabbit", "raccoon", "ray", "road", "rocket",
    "rose", "sea", "seal", "shark", "shrew", "skunk", "skyscraper", "snail",
    "snake", "spider", "squirrel", "streetcar", "sunflower", "sweet_pepper",
    "table", "tank", "telephone", "television", "tiger", "tractor", "train",
    "trout", "tulip", "turtle", "wardrobe", "whale", "willow_tree", "wolf",
    "woman", "worm",
]

CIFAR100_SUPERCLASSES = {
    "aquatic mammals":    ["beaver", "dolphin", "otter", "seal", "whale"],
    "fish":               ["aquarium_fish", "flatfish", "ray", "shark", "trout"],
    "flowers":            ["orchid", "poppy", "rose", "sunflower", "tulip"],
    "food containers":    ["bottle", "bowl", "can", "cup", "plate"],
    "fruit & vegetables": ["apple", "mushroom", "orange", "pear", "sweet_pepper"],
    "household furniture":["bed", "chair", "couch", "table", "wardrobe"],
    "household electrical":["clock", "keyboard", "lamp", "telephone", "television"],
    "insects":            ["bee", "beetle", "butterfly", "caterpillar", "cockroach"],
    "large carnivores":   ["bear", "leopard", "lion", "tiger", "wolf"],
    "large man-made":     ["bridge", "castle", "house", "road", "skyscraper"],
    "large natural":      ["cloud", "forest", "mountain", "plain", "sea"],
    "large omnivores":    ["camel", "cattle", "chimpanzee", "elephant", "kangaroo"],
    "medium mammals":     ["fox", "porcupine", "possum", "raccoon", "skunk"],
    "non-insect invertebrates": ["crab", "lobster", "snail", "spider", "worm"],
    "people":             ["baby", "boy", "girl", "man", "woman"],
    "reptiles":           ["crocodile", "dinosaur", "lizard", "snake", "turtle"],
    "small mammals":      ["hamster", "mouse", "rabbit", "shrew", "squirrel"],
    "trees":              ["maple_tree", "oak_tree", "palm_tree", "pine_tree", "willow_tree"],
    "vehicles 1":         ["bicycle", "bus", "motorcycle", "pickup_truck", "train"],
    "vehicles 2":         ["lawn_mower", "rocket", "streetcar", "tank", "tractor"],
}

# Reverse-lookup: class → superclass
CLASS_TO_SUPER = {
    cls: sup for sup, classes in CIFAR100_SUPERCLASSES.items() for cls in classes
}

CONFIDENCE_THRESHOLD = 0.40   # below this → "unrecognized"
INPUT_SIZE = (32, 32)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
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
    position: relative;
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
[data-testid="stFileUploader"]:hover {
    border-color: #7c6af0 !important;
}
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
    flex: 1;
    height: 6px;
    background: #1f1d2e;
    border-radius: 3px;
    overflow: hidden;
}
.conf-bar-fill {
    height: 100%;
    border-radius: 3px;
    background: linear-gradient(90deg, #7c6af0, #a78bfa);
    transition: width 0.8s cubic-bezier(0.4,0,0.2,1);
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
    width: 110px;
    height: 4px;
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

/* ── Divider ── */
.section-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #2e2b3e, transparent);
    margin: 2rem 0;
}

/* ── Model not found notice ── */
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

/* ── Streamlit misc overrides ── */
[data-testid="stDecoration"] { display: none; }
.stSpinner > div { border-top-color: #7c6af0 !important; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ─────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model(path: str):
    """Load the Keras model once and cache it."""
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(path)
        return model, None
    except Exception as e:
        return None, str(e)
    

def preprocess_image(img: Image.Image) -> np.ndarray:
    """Resize → RGB → normalise → add batch dim."""
    img = img.convert("RGB").resize(INPUT_SIZE, Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, 0)


def predict(model, img: Image.Image):
    """Return list of (class_name, probability) sorted desc."""
    x = preprocess_image(img)
    preds = model.predict(x, verbose=0)[0]          # shape (100,)
    top_idx = preds.argsort()[::-1]
    return [(CIFAR100_CLASSES[i], float(preds[i])) for i in top_idx]


def bar_color(rank: int) -> str:
    return ["#7c6af0", "#9d8ef5", "#b8aaf7", "#ccc4f9", "#dddafb"][rank]


# ── Header ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">Deep Learning · Computer Vision</div>
    <h1 class="hero-title">CIFAR-100<br>Classifier</h1>
    <div class="hero-sub">100 classes · 32 × 32 input · Keras / TensorFlow</div>
</div>
""", unsafe_allow_html=True)

# ── Model loading ────────────────────────────────────────────────────────────
MODEL_PATH = "cifar100_modelv2.h5"

model_loaded = os.path.exists(MODEL_PATH)
if not model_loaded:
    st.markdown(
        '<div class="model-warn">⚠ &nbsp;<code>cifar100_model.h5</code> not found in working directory. '
        "Place it alongside this script to enable live inference.</div>",
        unsafe_allow_html=True,
    )
    model, model_err = None, "Model file missing"
else:
    with st.spinner("Loading model weights…"):
        model, model_err = load_model(MODEL_PATH)
    if model_err:
        st.error(f"Failed to load model: {model_err}")
        model = None

# ── Upload section ───────────────────────────────────────────────────────────
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

uploaded = st.file_uploader(
    "Drop an image here",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
    label_visibility="collapsed",
)

if uploaded:
    img = Image.open(io.BytesIO(uploaded.read()))
    w, h = img.size
    size_kb = uploaded.size / 1024

    # Image preview
    st.markdown('<div class="img-card">', unsafe_allow_html=True)
    st.image(img, use_container_width=False, width=min(w, 380))
    st.markdown(
        f'<div class="img-meta">{uploaded.name} &nbsp;·&nbsp; {w}×{h}px &nbsp;·&nbsp; {size_kb:.1f} KB</div>',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # Predict button
    predict_clicked = st.button("⚡  Predict", use_container_width=True)

    if predict_clicked:
        if model is None:
            st.markdown(
                '<div class="error-card">'
                '<div class="error-icon">🔌</div>'
                '<div class="error-title">Model Unavailable</div>'
                '<div class="error-sub">Cannot run inference — see the warning above.</div>'
                "</div>",
                unsafe_allow_html=True,
            )
        else:
            with st.spinner("Running inference…"):
                results = predict(model, img)

            top_class, top_conf = results[0]
            top5 = results[:5]

            if top_conf < CONFIDENCE_THRESHOLD:
                # ── Unrecognized ──────────────────────────────────────────────
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
            else:
                # ── Result card ───────────────────────────────────────────────
                super_label = CLASS_TO_SUPER.get(top_class, "—")
                pct_int = int(top_conf * 100)
                bar_w = min(pct_int, 100)

                top5_rows = ""
                max_conf = top5[0][1]
                for rank, (cls, conf) in enumerate(top5):
                    is_top = rank == 0
                    rel_w = int(conf / max_conf * 100)
                    top5_rows += f"""
                    <div class="top5-row">
                        <span class="top5-rank">#{rank+1}</span>
                        <span class="top5-name {'top' if is_top else ''}">{cls.replace('_',' ')}</span>
                        <div class="top5-bar-wrap">
                            <div class="top5-bar-fill"
                                 style="width:{rel_w}%;background:{bar_color(rank)};"></div>
                        </div>
                        <span class="top5-pct {'top' if is_top else ''}">{conf*100:.1f}%</span>
                    </div>"""

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

else:
    # Placeholder hint when nothing uploaded yet
    st.markdown("""
    <div style="text-align:center;padding:1.5rem 0 0.5rem;">
        <span style="font-family:'Space Mono',monospace;font-size:0.75rem;color:#2e2b3e;
                     letter-spacing:0.1em;">upload an image to begin</span>
    </div>
    """, unsafe_allow_html=True)

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center;padding-bottom:2rem;">
    <span style="font-family:'Space Mono',monospace;font-size:0.65rem;color:#2e2b3e;
                 letter-spacing:0.1em;">CIFAR-100 · 100 classes across 20 superclasses</span>
</div>
""", unsafe_allow_html=True)