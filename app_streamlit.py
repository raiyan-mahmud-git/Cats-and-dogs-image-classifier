"""
app_streamlit.py
Production Streamlit Application for Cat vs Dog AI Vision.

Features:
- 100% cloud-ready for Streamlit Community Cloud (share.streamlit.io) — 100% free hosting.
- Model format selector: TFLite (Float16 quantized, Float32) and Keras (Deep Feature Head).
- Instant inference caching with `@st.cache_resource`.
- Single-image interactive inspection with Grad-CAM salience attention maps.
- Batch image upload tab with tabular metrics and CSV export.
- Quick-test gallery for 1-click sample benchmarking.
"""

import io
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image as PILImage
import streamlit as st

# Ensure src/ is importable
ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from catdog.config import Config
from catdog.explainability import generate_gradcam_heatmap, superimpose_gradcam
from catdog.predict import (
    MAX_FILE_SIZE_BYTES,
    MAX_FILE_SIZE_MB,
    CatDogPredictor,
    ImageValidationError,
    _load_and_validate_image,
)

# ---------------------------------------------------------------------------
# Streamlit Page Config & Custom Styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Cat vs Dog AI Vision",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #64748b;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border-radius: 10px;
        padding: 1rem;
        border: 1px solid #e2e8f0;
        text-align: center;
    }
    .badge-dog {
        background-color: #3b82f6;
        color: white;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .badge-cat {
        background-color: #ec4899;
        color: white;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

config = Config()

# ---------------------------------------------------------------------------
# Cached Model Loaders
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading inference model into memory...")
def load_predictor(model_filename: str) -> CatDogPredictor:
    """Load and cache CatDogPredictor singleton."""
    model_path = config.models_dir / model_filename
    if not model_path.exists():
        return CatDogPredictor()
    return CatDogPredictor(model_path=model_path)


@st.cache_resource(show_spinner="Loading Keras model for Grad-CAM explainability...")
def load_keras_explain_model():
    """Load full Keras model for gradient-based attention mapping."""
    import tensorflow as tf

    candidate = config.models_dir / "transfer_model_inference.keras"
    if not candidate.exists():
        candidate = config.models_dir / "transfer_model.keras"
    if not candidate.exists():
        candidate = config.models_dir / "baseline_model.keras"
    if not candidate.exists():
        return None
    return tf.keras.models.load_model(candidate)


# ---------------------------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🐾 Configuration")

    available_models = []
    for name in [
        "transfer_model_float16.tflite",
        "transfer_model_float32.tflite",
        "transfer_model_inference.keras",
        "improved_cnn.keras",
        "baseline_model.keras",
    ]:
        if (config.models_dir / name).exists():
            available_models.append(name)

    if not available_models:
        st.error("No model files found in models/.")
        st.stop()

    selected_model = st.selectbox(
        "🧠 Active Model Engine",
        available_models,
        index=0,
        help="TFLite models offer sub-5ms edge latency. Keras models offer full computational graphs.",
    )

    st.markdown("---")
    st.subheader("🔍 Explainability")
    enable_gradcam = st.toggle("Enable Grad-CAM Attention", value=True)
    heatmap_alpha = st.slider("Heatmap Overlay Opacity", 0.1, 0.9, 0.45, 0.05)
    colormap = st.selectbox("Colormap", ["jet", "viridis", "inferno", "magma"], index=0)

    st.markdown("---")
    st.subheader("ℹ️ Project Specs")
    st.markdown(
        """
        - **Model:** MobileNetV2 Transfer Learning
        - **Input Resolution:** 150 × 150 × 3
        - **freeCodeCamp Target:** 63%
        - **Achieved Accuracy:** **98.0%**
        - **Deployment:** Streamlit Cloud
        """
    )


# ---------------------------------------------------------------------------
# Main UI Tabs
# ---------------------------------------------------------------------------
st.markdown('<div class="main-header">🐾 Cat vs Dog AI Vision</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Production-grade binary image classification with MobileNetV2 and Grad-CAM salience explainability.</div>',
    unsafe_allow_html=True,
)

tab1, tab2 = st.tabs(["📷 Single Image Analysis", "📁 Batch Classification"])

# ---------------------------------------------------------------------------
# TAB 1: Single Image Analysis
# ---------------------------------------------------------------------------
with tab1:
    col_input, col_result = st.columns([1, 1], gap="medium")

    with col_input:
        st.subheader("1. Provide Image")
        source_mode = st.radio("Choose input method:", ["Upload Image", "Pick Sample from Benchmark"], horizontal=True)

        input_image_path = None
        input_pil_img = None
        image_name = ""

        if source_mode == "Upload Image":
            uploaded_file = st.file_uploader(
                "Upload a pet photo (JPEG, PNG, WebP up to 20 MB)",
                type=["jpg", "jpeg", "png", "bmp", "webp"],
            )
            if uploaded_file is not None:
                if uploaded_file.size > MAX_FILE_SIZE_BYTES:
                    st.error(f"File too large! Max allowed is {MAX_FILE_SIZE_MB} MB.")
                else:
                    import tempfile

                    ext = Path(uploaded_file.name).suffix or ".jpg"
                    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                        tmp.write(uploaded_file.getvalue())
                        input_image_path = Path(tmp.name)
                    input_pil_img = PILImage.open(io.BytesIO(uploaded_file.getvalue()))
                    image_name = uploaded_file.name
        else:
            sample_options = {
                "Sample 1 (Dog)": "1.jpg",
                "Sample 2 (Dog)": "2.jpg",
                "Sample 3 (Dog)": "3.jpg",
                "Sample 4 (Cat)": "4.jpg",
                "Sample 32 (Tricky Dog)": "32.jpg",
            }
            sample_choice = st.selectbox("Select sample benchmark image:", list(sample_options.keys()))
            sample_file = config.test_dir / sample_options[sample_choice]
            if sample_file.exists():
                input_image_path = sample_file
                input_pil_img = PILImage.open(sample_file)
                image_name = sample_file.name
            else:
                st.warning(f"Sample {sample_file.name} not found in test directory.")

        if input_pil_img is not None:
            st.image(input_pil_img, caption=f"{image_name} ({input_pil_img.size[0]}×{input_pil_img.size[1]})", use_container_width=True)

    with col_result:
        st.subheader("2. AI Prediction & Insights")

        if input_image_path is not None:
            predictor = load_predictor(selected_model)

            try:
                # Run prediction
                res = predictor.predict_image(input_image_path)
                label = res["label"]
                conf = res["confidence_pct"]
                p_dog = res["dog_probability"]
                p_cat = round(1.0 - p_dog, 4)
                latency = res["time_ms"]

                # Result cards
                m1, m2, m3 = st.columns(3)
                m1.metric("Predicted Class", f"{'🐶' if label == 'Dog' else '🐱'} {label}")
                m2.metric("Confidence", f"{conf:.1f}%")
                m3.metric("Latency", f"{latency:.1f} ms")

                st.markdown("#### Probability Distribution")
                st.progress(float(p_dog), text=f"🐶 Dog Probability: {p_dog * 100:.2f}%")
                st.progress(float(p_cat), text=f"🐱 Cat Probability: {p_cat * 100:.2f}%")

                # Grad-CAM Section
                if enable_gradcam:
                    st.markdown("---")
                    st.markdown("#### 🔬 Grad-CAM Attention Heatmap")
                    keras_model = load_keras_explain_model()
                    if keras_model is not None:
                        with st.spinner("Computing class activation gradients..."):
                            raw_arr = _load_and_validate_image(input_image_path, target_size=(config.img_height, config.img_width))
                            heatmap = generate_gradcam_heatmap(keras_model, raw_arr)
                            overlay = superimpose_gradcam(
                                raw_arr.astype(np.uint8),
                                heatmap,
                                alpha=heatmap_alpha,
                                colormap_name=colormap,
                            )

                            g1, g2 = st.columns(2)
                            with g1:
                                st.image(overlay, caption="Grad-CAM Focus Overlay", use_container_width=True)
                            with g2:
                                st.image(heatmap, caption="Raw 2D Attention Map", clamp=True, use_container_width=True)

                            st.caption(
                                "💡 **How to interpret:** Bright warm regions (red/yellow in 'jet') highlight spatial features "
                                "that contributed most to the model's decision (ears, muzzle, facial texture)."
                            )
                    else:
                        st.info("Keras model not available for Grad-CAM overlay.")

            except Exception as e:
                st.error(f"Inference error: {e}")
            finally:
                if source_mode == "Upload Image" and input_image_path and input_image_path.exists():
                    input_image_path.unlink(missing_ok=True)
        else:
            st.info("👈 Please upload an image or pick a benchmark sample to view predictions.")


# ---------------------------------------------------------------------------
# TAB 2: Batch Classification
# ---------------------------------------------------------------------------
with tab2:
    st.subheader("📁 Batch File Processing")
    st.write("Upload a collection of images to classify them simultaneously and export structured results.")

    batch_files = st.file_uploader(
        "Upload multiple images",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        accept_multiple_files=True,
    )

    if batch_files:
        predictor = load_predictor(selected_model)
        records = []
        progress_bar = st.progress(0, text="Classifying batch images...")

        import tempfile

        for i, f in enumerate(batch_files):
            ext = Path(f.name).suffix or ".jpg"
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                tmp.write(f.getvalue())
                tmp_p = Path(tmp.name)

            try:
                r = predictor.predict_image(tmp_p)
                records.append({
                    "Image Name": f.name,
                    "Prediction": r["label"],
                    "Confidence (%)": r["confidence_pct"],
                    "P(Dog)": r["dog_probability"],
                    "Latency (ms)": r["time_ms"],
                })
            except Exception as ex:
                records.append({
                    "Image Name": f.name,
                    "Prediction": "ERROR",
                    "Confidence (%)": 0.0,
                    "P(Dog)": 0.0,
                    "Latency (ms)": 0.0,
                    "Error": str(ex),
                })
            finally:
                tmp_p.unlink(missing_ok=True)

            progress_bar.progress((i + 1) / len(batch_files), text=f"Processed {i + 1}/{len(batch_files)} images...")

        df = pd.DataFrame(records)
        st.success(f"Successfully processed {len(records)} images!")
        st.dataframe(df, use_container_width=True)

        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Predictions CSV",
            data=csv_data,
            file_name="catdog_predictions.csv",
            mime="text/csv",
        )
