# 🐾 Cat vs Dog AI Vision: Production Classifier & Explainability

[![CI](https://github.com/raiyan-mahmud-git/Cats-and-dogs-image-classifier/actions/workflows/ci.yml/badge.svg)](https://github.com/raiyan-mahmud-git/Cats-and-dogs-image-classifier/actions/workflows/ci.yml)
[![Python 3.11 | 3.12](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![TensorFlow 2.16+](https://img.shields.io/badge/TensorFlow-2.16+-FF6F00.svg?logo=tensorflow)](https://tensorflow.org/)
[![Keras 3](https://img.shields.io/badge/Keras-3.x-D00000.svg?logo=keras)](https://keras.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg?logo=streamlit)](https://streamlit.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-REST%20API-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Accuracy](https://img.shields.io/badge/FCC%20Test%20Accuracy-98.0%25-brightgreen.svg)](#-empirical-benchmarks)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end, production-grade deep learning computer vision system built for the **freeCodeCamp Machine Learning Certification Challenge**. 

Starting from a fragile baseline CNN (~74% accuracy), this project systematically engineers modern computer vision architectures—incorporating **Global Average Pooling**, **in-graph data augmentation**, **two-stage MobileNetV2 transfer learning**, **TFLite Float16 quantization**, and **Grad-CAM visual explainability**—surpassing the 63% certification pass mark to achieve **98.0% accuracy**.

---

## 🌟 Key Highlights

- **🎯 State-of-the-Art Accuracy:** Achieved **98.0% accuracy** on the official 50-image freeCodeCamp test set (benchmark threshold: 63%).
- **🔬 Explainable AI (XAI):** Built-in **Grad-CAM** (Gradient-Weighted Class Activation Mapping) on MobileNetV2's final convolutional layer (`out_relu`), generating real-time visual attention overlays highlighting discriminatory features (eyes, snout, ears).
- **⚡ Ultra-Fast Edge Inference:** Exported clean **TFLite Float16** models offering **~3.5 ms CPU latency** with an 82% footprint reduction (from 25.1 MB down to **4.57 MB**).
- **🖥️ Dual Production Interfaces:**
  - **Streamlit Dashboard ([`app_streamlit.py`](app_streamlit.py)):** Drag-and-drop prediction, Grad-CAM opacity sliders, benchmark sample gallery, and batch image classification with CSV export.
  - **FastAPI REST API ([`src/catdog/api.py`](src/catdog/api.py)):** Production REST API with auto-generated OpenAPI / Swagger docs at `/docs`.
- **🛡️ Battle-Tested Architecture:** 42 automated tests across unit, integration, and CI suites with GitHub Actions multi-version matrix testing.

---

## 📊 Empirical Benchmarks

| Model Architecture | Parameters | Disk Size | CPU Latency | Val Accuracy | FCC Test Accuracy | Notes |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Stage 1: Baseline FCC CNN** | 19,034,177 | 72.6 MB | 28.4 ms | 73.6% | 74.0% | Heavy Flatten layer; memory-bound |
| **Stage 2: GAP CNN + Augmentation** | 421,441 | 1.6 MB | 14.2 ms | 81.2% | 82.0% | 98% parameter reduction; BatchNorm |
| **Stage 3: MobileNetV2 (Fine-Tuned)** | 2,423,169 | 25.1 MB | 50.1 ms | **97.6%** | **98.0%** | Pretrained ImageNet representations |
| **Stage 5: TFLite (Float32)** | 2,423,169 | 9.08 MB | 3.69 ms | 97.6% | **98.0%** | Intermediate SavedModel export |
| **Stage 5: TFLite (Float16 Quantized)** | 2,423,169 | **4.57 MB** | **3.55 ms** | 97.6% | **98.0%** | **82% smaller, 14x faster inference** |

---

## 🔬 Explainable AI: Grad-CAM Attention

Traditional CNNs act as "black boxes". This project implements **Grad-CAM** to compute the gradient of the predicted class score with respect to the feature activation maps of MobileNetV2's final convolutional layer:

$$A_{\text{Grad-CAM}} = \text{ReLU}\left( \sum_{k} \alpha_k A^k \right), \quad \alpha_k = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial y^c}{\partial A_{i,j}^k}$$

This produces spatial saliency heatmaps that visually verify whether the model is focusing on the actual animal morphology rather than background shortcuts or image artifacts.

---

## 🚀 Quickstart & Local Execution

### 1. Clone & Set Up Environment
```powershell
git clone https://github.com/raiyan-mahmud-git/Cats-and-dogs-image-classifier.git
cd Cats-and-dogs-image-classifier

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1  # On Windows PowerShell
# source .venv/bin/activate    # On Linux/macOS

# Install dependencies and package in editable mode
pip install -e ".[dev]"
```

### 2. Launch Streamlit Web App
```powershell
streamlit run app_streamlit.py
```
Open **`http://localhost:8501`** in your browser.

### 3. Launch FastAPI Backend
```powershell
python -m uvicorn catdog.api:app --app-dir src --host 127.0.0.1 --port 8000 --reload
```
Interactive API documentation will be available at **`http://127.0.0.1:8000/docs`**.

### 4. CLI Inference Tool
Classify a single image or a folder with automatic CSV summary:
```powershell
# Single image
python -m catdog.predict cats_and_dogs/test/1.jpg

# Batch folder mode with CSV output
python -m catdog.predict cats_and_dogs/test/ --csv results.csv
```

---

## 🌐 1-Click Free Cloud Deployment (Streamlit Community Cloud)

You can host this project online for free with **zero Docker installation**:

1. Log in to **[share.streamlit.io](https://share.streamlit.io)** with your GitHub account.
2. Click **"New app"** (or **"Create app"**).
3. Select your repository:
   - **Repository:** `raiyan-mahmud-git/Cats-and-dogs-image-classifier`
   - **Branch:** `main`
   - **Main file path:** `app_streamlit.py`
4. Click **"Deploy!"**
5. Streamlit Community Cloud will automatically build from [`requirements.txt`](requirements.txt) and provide a permanent public HTTPS URL.

---

## 🧪 Testing & Continuous Integration

This project enforces strict code quality and test coverage via **pytest** and **GitHub Actions**:

```powershell
# Run the complete test suite (42 tests)
pytest tests/ -v

# Run fast unit tests only (skipping slow weight loads)
pytest tests/ -v -m "not slow"

# Code formatting and lint verification
ruff check src/ tests/
ruff format --check src/ tests/
```

### Test Suite Structure:
- `tests/test_baseline.py`: Baseline model shapes, forward pass validity.
- `tests/test_stage2.py`: GlobalAveragePooling parameter reduction & training callbacks.
- `tests/test_stage3.py`: MobileNetV2 head freezing, rescaling, fine-tuning transitions.
- `tests/test_stage4.py`: Binary metrics, ROC-AUC, Wilson score confidence intervals, Grad-CAM generation.
- `tests/test_stage5.py`: Input validation edge cases (corrupt, grayscale, RGBA, oversized), CLI parsing, and TFLite inference.
- `tests/test_stage6.py`: FastAPI endpoints (`/api/health`, `/api/predict`, `/api/sample`) and UI construction.
- `tests/test_stage7.py`: Packaging integrity (PEP 621), CI workflow schema, and synthetic dummy model fixtures.

---

## 📁 Repository Structure

```
Cats-and-dogs-image-classifier/
├── .github/
│   └── workflows/
│       └── ci.yml               # GitHub Actions CI matrix workflow
├── models/                      # Lightweight production & TFLite models
│   ├── improved_cnn.keras
│   ├── transfer_model_float16.tflite
│   ├── transfer_model_float32.tflite
│   └── transfer_model_inference.keras
├── notebooks/                   # Step-by-step educational notebooks (Stages 1-7)
├── reports/                     # Evaluation metrics, ROC curves, and ablation plots
├── src/
│   └── catdog/                  # Core package
│       ├── api.py               # FastAPI REST service
│       ├── config.py            # Central dataclass configuration
│       ├── explainability.py    # Grad-CAM heatmap generator
│       ├── export.py            # TFLite conversion & latency benchmarking
│       ├── metrics.py           # Precision, recall, ROC-AUC, calibration
│       ├── model.py             # CNN & MobileNetV2 architectures
│       └── predict.py           # Production inference engine & CLI
├── tests/                       # 42 automated tests & conftest fixtures
├── app.py                       # Gradio entrypoint
├── app_streamlit.py             # Streamlit dashboard
├── pyproject.toml               # PEP 621 package configuration
├── requirements.txt             # Pinned production dependencies
└── README.md
```

---

## 📜 License

Distributed under the **MIT License**. See `LICENSE` for details.
