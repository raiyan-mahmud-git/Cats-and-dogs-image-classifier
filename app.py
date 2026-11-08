"""Gradio web application for Cat vs Dog image classification."""

import sys
from pathlib import Path
from typing import Dict, Tuple, Union

# Ensure src/ is on Python path (supports standalone execution & Hugging Face Spaces)
SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import gradio as gr
import numpy as np
from PIL import Image as PILImage

from catdog.config import Config
from catdog.explainability import generate_gradcam_heatmap, superimpose_gradcam
from catdog.predict import CatDogPredictor, _load_and_validate_image

# ---------------------------------------------------------------------------
# Global State & Lazy Model Loading
# ---------------------------------------------------------------------------
config = Config()
_predictor: Union[CatDogPredictor, None] = None
_keras_model = None


def get_predictor() -> CatDogPredictor:
    """Singleton getter for the CatDog predictor."""
    global _predictor
    if _predictor is None:
        _predictor = CatDogPredictor()
    return _predictor


def get_keras_model():
    """Singleton getter for the full Keras model (needed for Grad-CAM gradients)."""
    global _keras_model
    if _keras_model is None:
        import tensorflow as tf

        model_path = config.models_dir / "transfer_model_inference.keras"
        if not model_path.exists():
            model_path = config.models_dir / "transfer_model.keras"
        if not model_path.exists():
            model_path = config.models_dir / "baseline_model.keras"
        _keras_model = tf.keras.models.load_model(model_path)
    return _keras_model


# ---------------------------------------------------------------------------
# Prediction & Explanation Logic
# ---------------------------------------------------------------------------
def predict_and_explain(
    input_image: PILImage.Image,
    show_gradcam: bool,
) -> Tuple[Dict[str, float], Union[np.ndarray, None], str]:
    """Process an uploaded PIL image and return probabilities, optional Grad-CAM, and summary.

    Returns:
        (confidences_dict, gradcam_overlay_or_none, status_message)
    """
    if input_image is None:
        return {}, None, "Please upload an image to classify."

    try:
        # Save temporary image for standard predictor pipeline
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            if input_image.mode != "RGB":
                input_image = input_image.convert("RGB")
            input_image.save(tmp_path, format="JPEG")

        predictor = get_predictor()
        res = predictor.predict_image(tmp_path)
        dog_prob = res["dog_probability"]
        cat_prob = round(1.0 - dog_prob, 4)

        confidences = {
            "Cat": cat_prob,
            "Dog": dog_prob,
        }

        # Grad-CAM overlay generation
        overlay_img = None
        if show_gradcam:
            keras_model = get_keras_model()
            raw_array = _load_and_validate_image(
                tmp_path, target_size=(config.img_height, config.img_width)
            )
            heatmap = generate_gradcam_heatmap(keras_model, raw_array)
            overlay = superimpose_gradcam(raw_array.astype(np.uint8), heatmap, alpha=0.45)
            overlay_img = overlay

        # Clean up temporary file
        tmp_path.unlink(missing_ok=True)

        summary = (
            f"**Prediction:** {res['label']} ({res['confidence_pct']}% confidence)\n"
            f"**Inference latency:** {res['time_ms']:.1f} ms"
        )
        return confidences, overlay_img, summary

    except Exception as e:
        return {}, None, f"Error processing image: {str(e)}"


# ---------------------------------------------------------------------------
# Gradio UI Construction
# ---------------------------------------------------------------------------
def build_gradio_app() -> gr.Blocks:
    """Build the Blocks-based interactive web UI."""
    sample_images = []
    test_dir = config.test_dir
    if test_dir.exists():
        for fname in ["1.jpg", "2.jpg", "3.jpg", "4.jpg"]:
            p = test_dir / fname
            if p.exists():
                sample_images.append([str(p), True])

    with gr.Blocks(
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="amber",
            neutral_hue="slate",
        ),
        title="Cat vs Dog AI Classifier",
    ) as demo:
        gr.Markdown(
            """
            # 🐾 Cat vs Dog Image Classifier
            ### Portfolio-Quality Deep Learning with MobileNetV2 & Explainable AI (Grad-CAM)
            
            Upload any photo of a cat or dog, or select an example below. The model analyzes the image
            in real time and visualizes its attention using **Grad-CAM**.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(
                    type="pil",
                    label="Input Image (Drag & Drop or Click)",
                    sources=["upload", "clipboard"],
                )
                show_gradcam_chk = gr.Checkbox(
                    value=True,
                    label="Generate Grad-CAM Explainability Heatmap",
                    info="Highlights the pixels that contributed most to the prediction.",
                )
                submit_btn = gr.Button("Classify Image", variant="primary", size="lg")

            with gr.Column(scale=1):
                output_label = gr.Label(label="Prediction Probabilities", num_top_classes=2)
                summary_markdown = gr.Markdown(label="Inference Details")
                gradcam_image = gr.Image(
                    label="Grad-CAM Salience Map Overlay",
                    type="numpy",
                    interactive=False,
                )

        submit_btn.click(
            fn=predict_and_explain,
            inputs=[input_image, show_gradcam_chk],
            outputs=[output_label, gradcam_image, summary_markdown],
        )

        # Trigger on image upload automatically for seamless UX
        input_image.change(
            fn=predict_and_explain,
            inputs=[input_image, show_gradcam_chk],
            outputs=[output_label, gradcam_image, summary_markdown],
        )

        if sample_images:
            gr.Examples(
                examples=sample_images,
                inputs=[input_image, show_gradcam_chk],
                outputs=[output_label, gradcam_image, summary_markdown],
                fn=predict_and_explain,
                cache_examples=False,
                label="Sample Test Images",
            )

        gr.Markdown(
            """
            ---
            **Project Highlights:**
            - **Model:** Pretrained MobileNetV2 fine-tuned with 2-stage transfer learning.
            - **Accuracy:** 98.0% on official 50-image test set (freeCodeCamp pass mark: 63%).
            - **Explainability:** Grad-CAM (Gradient-Weighted Class Activation Mapping) on final convolutional layer (`out_relu`).
            - **Deployment:** Sub-10ms CPU inference with automated channel/format validation.
            """
        )

    return demo


demo = build_gradio_app()

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
