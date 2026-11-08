"""Unit tests for FastAPI REST API and web application."""

import io
import pytest
from fastapi.testclient import TestClient
from PIL import Image as PILImage

from catdog.api import app
from catdog.config import Config


@pytest.fixture
def client():
    """Create FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def sample_image_bytes():
    """Generate in-memory RGB JPEG bytes for testing."""
    img = PILImage.new("RGB", (150, 150), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


class TestFastAPIEndpoints:
    def test_health_endpoint(self, client):
        """GET /api/health returns healthy status and model metadata."""
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert "active_model" in data
        assert "target_image_size" in data
        assert data["target_image_size"] == [150, 150]

    def test_predict_valid_image(self, client, sample_image_bytes):
        """POST /api/predict accepts a valid JPEG and returns structured predictions."""
        files = {"file": ("test.jpg", sample_image_bytes, "image/jpeg")}
        resp = client.post("/api/predict", files=files)
        assert resp.status_code == 200
        data = resp.json()

        assert data["success"] is True
        assert data["filename"] == "test.jpg"
        assert data["label"] in ("Cat", "Dog")
        assert 0.0 <= data["confidence_pct"] <= 100.0
        assert 0.0 <= data["dog_probability"] <= 1.0
        assert 0.0 <= data["cat_probability"] <= 1.0
        assert round(data["dog_probability"] + data["cat_probability"], 2) == 1.0
        assert data["inference_time_ms"] > 0
        assert data["gradcam_base64"] is None  # explain not requested

    def test_predict_with_explain(self, client, sample_image_bytes):
        """POST /api/predict?explain=true returns base64 Grad-CAM overlay."""
        files = {"file": ("test.jpg", sample_image_bytes, "image/jpeg")}
        resp = client.post("/api/predict?explain=true", files=files)
        assert resp.status_code == 200
        data = resp.json()

        assert data["gradcam_base64"] is not None
        assert data["gradcam_base64"].startswith("data:image/jpeg;base64,")

    def test_predict_unsupported_extension(self, client):
        """POST /api/predict rejects non-image extension with HTTP 415."""
        files = {"file": ("document.pdf", b"fake pdf content", "application/pdf")}
        resp = client.post("/api/predict", files=files)
        assert resp.status_code == 415
        assert "Unsupported file type" in resp.json()["detail"]

    def test_predict_empty_file(self, client):
        """POST /api/predict rejects 0-byte file with HTTP 400."""
        files = {"file": ("empty.jpg", b"", "image/jpeg")}
        resp = client.post("/api/predict", files=files)
        assert resp.status_code == 400
        assert "empty" in resp.json()["detail"].lower()

    def test_sample_image_endpoint(self, client):
        """GET /api/sample/1.jpg returns a valid sample JPEG."""
        config = Config()
        sample_file = config.test_dir / "1.jpg"
        if not sample_file.exists():
            pytest.skip("Test image 1.jpg not found")

        resp = client.get("/api/sample/1.jpg")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/jpeg"
        assert len(resp.content) > 0


class TestGradioApp:
    def test_gradio_app_construction(self):
        """Verify Gradio app builds properly from app.py."""
        from app import build_gradio_app
        import gradio as gr

        demo = build_gradio_app()
        assert isinstance(demo, gr.Blocks)
        assert demo.title == "Cat vs Dog AI Classifier"
