"""
src/catdog/api.py
FastAPI production REST API for Cat vs Dog image classification.

Endpoints:
- `POST /api/predict`     Multipart image upload, returns JSON predictions & optional Grad-CAM.
- `GET  /api/health`      Health check with active model metadata and runtime info.
- `GET  /api/sample/{id}` Serves sample benchmark images for one-click testing.
- `GET  /`                Serves the responsive single-page web app.

Design decisions:
1. File size is strictly enforced before full buffer buffering (rejecting >20 MB with HTTP 413),
   preventing memory exhaustion attacks from malicious or accidental payload dumps.
2. The endpoint supports an `explain=true` query parameter, decoupling fast classification
   (<10 ms) from heavier gradient-based saliency mapping (~50 ms) to keep the default API fast.
3. FastAPI's OpenAPI integration generates interactive Swagger UI documentation at `/docs`,
   making the API immediately discoverable for frontend developers and recruiters.
"""

import base64
import io
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image as PILImage
from pydantic import BaseModel, Field

from catdog import __version__
from catdog.config import Config
from catdog.explainability import generate_gradcam_heatmap, superimpose_gradcam
from catdog.predict import (
    MAX_FILE_SIZE_BYTES,
    MAX_FILE_SIZE_MB,
    VALID_EXTENSIONS,
    CatDogPredictor,
    ImageValidationError,
    _load_and_validate_image,
)

# ---------------------------------------------------------------------------
# App Initialization & Schemas
# ---------------------------------------------------------------------------
config = Config()

app = FastAPI(
    title="Cat vs Dog Image Classifier API",
    description="High-performance binary image classification with MobileNetV2 and Grad-CAM explainability.",
    version=__version__,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files directory
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class HealthResponse(BaseModel):
    status: str = Field(..., json_schema_extra={"example": "healthy"})
    version: str = Field(..., json_schema_extra={"example": __version__})
    active_model: str = Field(..., json_schema_extra={"example": "transfer_model_float16.tflite"})
    is_tflite: bool = Field(..., json_schema_extra={"example": True})
    target_image_size: list = Field(..., json_schema_extra={"example": [150, 150]})


class PredictionResponse(BaseModel):
    success: bool = True
    filename: str = Field(..., json_schema_extra={"example": "photo.jpg"})
    label: str = Field(..., json_schema_extra={"example": "Dog"})
    confidence_pct: float = Field(..., json_schema_extra={"example": 99.6})
    cat_probability: float = Field(..., json_schema_extra={"example": 0.004})
    dog_probability: float = Field(..., json_schema_extra={"example": 0.996})
    inference_time_ms: float = Field(..., json_schema_extra={"example": 4.5})
    gradcam_base64: Optional[str] = Field(
        None,
        description="Base64 data URI of the Grad-CAM heatmap overlay (if requested via explain=true)",
    )


# ---------------------------------------------------------------------------
# Lazy Singletons
# ---------------------------------------------------------------------------
_predictor: Optional[CatDogPredictor] = None
_keras_model = None


def get_api_predictor() -> CatDogPredictor:
    """Singleton getter for predictor."""
    global _predictor
    if _predictor is None:
        _predictor = CatDogPredictor()
    return _predictor


def get_api_keras_model():
    """Singleton getter for full Keras model used for Grad-CAM gradients."""
    global _keras_model
    if _keras_model is None:
        import tensorflow as tf

        m_path = config.models_dir / "transfer_model_inference.keras"
        if not m_path.exists():
            m_path = config.models_dir / "transfer_model.keras"
        if not m_path.exists():
            m_path = config.models_dir / "baseline_model.keras"
        _keras_model = tf.keras.models.load_model(m_path)
    return _keras_model


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def serve_index():
    """Serve the single-page responsive web application."""
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        return HTMLResponse(
            "<h2>Cat vs Dog Classifier API</h2><p>Visit <a href='/docs'>/docs</a> for Swagger documentation.</p>"
        )
    return FileResponse(index_path)


@app.get("/api/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check():
    """System health check and runtime model metadata."""
    predictor = get_api_predictor()
    return HealthResponse(
        status="healthy",
        version=__version__,
        active_model=predictor.model_path.name,
        is_tflite=predictor._is_tflite,
        target_image_size=[config.img_height, config.img_width],
    )


@app.get("/api/sample/{sample_id}", tags=["Utilities"])
async def get_sample_image(sample_id: str):
    """Retrieve sample benchmark images from the test set for quick testing."""
    safe_id = Path(sample_id).name
    if not safe_id.endswith(".jpg"):
        safe_id = f"{safe_id}.jpg"

    sample_path = config.test_dir / safe_id
    if not sample_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Sample image '{safe_id}' not found.",
        )
    return FileResponse(sample_path, media_type="image/jpeg")


@app.post(
    "/api/predict",
    response_model=PredictionResponse,
    tags=["Inference"],
    summary="Classify image as Cat or Dog",
)
async def predict_image_endpoint(
    file: UploadFile = File(..., description="JPEG/PNG image file to classify"),
    explain: bool = Query(
        False,
        description="Whether to generate Grad-CAM explainability heatmap overlay",
    ),
):
    """Classify an uploaded image and optionally generate a Grad-CAM salience map.

    - **file**: Multipart uploaded image file (JPEG, PNG, WebP, etc., max 20 MB).
    - **explain**: If true, includes a base64 encoded Grad-CAM attention overlay.
    """
    # 1. Validate file extension
    filename = Path(file.filename or "uploaded.jpg").name
    ext = Path(filename).suffix.lower()
    if ext not in VALID_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(VALID_EXTENSIONS))}",
        )

    # 2. Read bytes and enforce size limit
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum size of {MAX_FILE_SIZE_MB} MB.",
        )
    if len(contents) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty (0 bytes).",
        )

    # 3. Save to temporary file for standardized loader
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp_path = Path(tmp.name)
        tmp_path.write_bytes(contents)

    try:
        predictor = get_api_predictor()
        res = predictor.predict_image(tmp_path)
        dog_prob = res["dog_probability"]
        cat_prob = round(1.0 - dog_prob, 4)

        # 4. Optional Grad-CAM overlay
        gradcam_b64 = None
        if explain:
            keras_model = get_api_keras_model()
            raw_array = _load_and_validate_image(
                tmp_path, target_size=(config.img_height, config.img_width)
            )
            heatmap = generate_gradcam_heatmap(keras_model, raw_array)
            overlay = superimpose_gradcam(raw_array.astype(np.uint8), heatmap, alpha=0.45)

            # Encode overlay as base64 JPEG
            overlay_pil = PILImage.fromarray(overlay)
            buf = io.BytesIO()
            overlay_pil.save(buf, format="JPEG", quality=90)
            encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
            gradcam_b64 = f"data:image/jpeg;base64,{encoded}"

        return PredictionResponse(
            success=True,
            filename=filename,
            label=res["label"],
            confidence_pct=res["confidence_pct"],
            cat_probability=cat_prob,
            dog_probability=dog_prob,
            inference_time_ms=res["time_ms"],
            gradcam_base64=gradcam_b64,
        )

    except ImageValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Image validation failed: {str(e)}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}",
        )
    finally:
        tmp_path.unlink(missing_ok=True)
