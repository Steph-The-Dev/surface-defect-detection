# Industrial Surface Defect Detection 🔍

[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Checked with mypy](https://img.shields.io/badge/mypy-checked-blue.svg)](http://mypy-lang.org/)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](https://pytest.org/)

## 🎯 Overview
This project leverages a background in visual signal processing to build an automated, interactive quality inspection tool. It evaluates metallic surfaces in real-time, detecting anomalies like fine scratches and dents using a combination of classical Computer Vision and modern software engineering patterns.

## 🏗️ Project Structure
```text
surface-defect-detection/
├── assets/             # Sample images for testing
├── src/                # Core vision logic
│   ├── __init__.py
│   └── vision.py       # OOP Pipeline & Thresholding Strategies
├── tests/              # Automated unit tests
├── app.py              # Streamlit dashboard
├── Makefile            # Automation for setup, test, and lint
├── pyproject.toml      # Tooling configuration (Ruff, Mypy, Pytest)
└── requirements.txt    # Project dependencies
```

## 🚀 Key Features & Architecture

### 1. Object-Oriented Vision Pipeline
The project follows a modular, extensible architecture. The `DefectDetectionPipeline` class encapsulates the entire vision flow, utilizing the **Strategy Pattern** for thresholding. This allows for seamless swapping between global (Otsu), local (Adaptive), and manual methods, providing a foundation for future integration of Deep Learning models.

### 2. Advanced CV: Morphological Cleaning
To ensure robustness against noise and uneven lighting, the pipeline incorporates **Morphological Operations**:
- **Opening:** Mathematically removes small, bright noise specs (salt noise) from the binary mask.
- **Closing:** Fills small holes and bridges gaps within detected defects to ensure contour continuity.

### 3. Engineering Excellence
- **Strict Typing:** 100% type-hint coverage verified with `mypy`.
- **Automated Testing:** Comprehensive test suite with `pytest` covering edge cases and pipeline determinism.
- **Modern Tooling:** Optimized linting and formatting using `ruff` for industry-standard code quality.

## 🧠 The Mathematics of Binarization

### Otsu's Automatic Thresholding
Statistical global thresholding that minimizes intra-class variance to find the optimal separation in bimodal histograms.

### Adaptive (Local) Thresholding
Calculates local thresholds for small neighborhoods (e.g., 11x11), making it highly effective for metallic surfaces with complex reflections and uneven lighting.

### Topological Structural Analysis
Uses Suzuki’s (1985) border following algorithm to extract high-precision contours from the cleaned binary mask, filtered by area to eliminate microscopic artifacts.

## 🛠️ Installation & Setup

1. **Environment:** `conda create --name surface-inspection python=3.10 && conda activate surface-inspection`
2. **Setup:** `make setup`
3. **Run:** `make run`

## 🧪 Development & QA
Use the provided `Makefile` for standard development tasks:
- `make test`: Run unit tests.
- `make lint`: Check code quality.
- `make typecheck`: Verify static types.
- `make format`: Auto-format code.

*(A synthetic test image `assets/sample_scratch.jpg` is provided for immediate validation.)*
