# ── Model utilities: load · preprocess · predict ─────────────────────────────

import numpy as np
import streamlit as st
from PIL import Image

from classes import CIFAR100_CLASSES
from config import INPUT_SIZE, TOP_K

import warnings
warnings.filterwarnings("ignore")

@st.cache_resource(show_spinner=False)
def load_model(path: str):
    """
    Load the Keras .h5 model once and cache it for the session.

    Returns
    -------
    (model, error_message)
        error_message is None on success, a string on failure.
    """
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(path)
        return model, None
    except Exception as exc:
        return None, str(exc)


def preprocess_image(img: Image.Image) -> np.ndarray:
    """
    Convert a PIL image to the tensor expected by the model.

    Pipeline: any-mode → RGB → resize to INPUT_SIZE → [0, 1] float → batch dim.
    """
    img = img.convert("RGB").resize(INPUT_SIZE, Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)   # shape: (1, 32, 32, 3)


def run_predict(model, img: Image.Image) -> list[tuple[str, float]]:
    """
    Run inference and return all classes sorted by confidence descending.

    Returns
    -------
    list of (class_name, probability) — length 100, sorted desc.
    """
    x = preprocess_image(img)
    probs = model.predict(x, verbose=0)[0]          # shape: (100,)
    ranked = probs.argsort()[::-1]
    return [(CIFAR100_CLASSES[i], float(probs[i])) for i in ranked]


def top_k_results(model, img: Image.Image) -> list[tuple[str, float]]:
    """Convenience wrapper — returns only the TOP_K predictions."""
    return run_predict(model, img)[:TOP_K]