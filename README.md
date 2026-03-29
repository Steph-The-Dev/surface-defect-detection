# Industrial Surface Defect Detection 🔍

## 🎯 Overview
This project leverages a 20-year background in visual signal processing to build an automated, interactive quality inspection tool bridging traditional Computer Vision and modern Data Science. It evaluates metallic surfaces in real-time, detecting anomalies like fine scratches and dents.

## 🚀 Features & Architecture
- **Interactive Dashboard:** Built with Streamlit, enabling users to tweak kernel sizes and thresholds on the fly.
- **Architectural Modularity:** Core visual signal processing is isolated in a modular `src/` pipeline, decoupling the interface from the analysis logic.
- **Metrics & Visualization:** 
  - Overlays bounding contours on the original image for clarity.
  - Dynamically calculates the "Defect Area (%)" severity metric.
  - Plots the pixel intensity distribution (Histogram) to justify threshold choices.

## 🧠 The Mathematics of Binarization
The project explores several classical Computer Vision algorithms to solve the non-trivial problem of isolating microscopic defects on uneven surfaces.

### 1. Otsu's Automatic Thresholding (1979)
Developed by Nobuyuki Otsu, this global thresholding method assumes the image histogram is bimodal (contains two distinct classes of pixels: background metal and foreground defects). The algorithm exhaustively searches for a threshold value `t` that minimizes the **intra-class variance** (the spread within each group) while maximizing the **inter-class variance**. This statistically guarantees the optimal separation of dark and light pixels without human intervention.

### 2. Adaptive (Local) Thresholding
Global methods fail when a metallic surface has uneven lighting—a single threshold that isolates a dark scratch in a bright area will often completely black out a shadow. Adaptive Thresholding solves this by analyzing the image block-by-block (e.g., an 11x11 neighborhood). It calculates a dynamic threshold for each individual pixel against the mean brightness of its immediate neighbors (minus a tuning constant, `C`). This localized approach makes the algorithm incredibly sensitive to fine anomalies like hairline scratches.

### 3. Topological Structural Analysis (Suzuki, 1985)
Once the image is binarized into a mask, the application must identify the distinct structures within it. It relies on Satoshi Suzuki's famous "border following" algorithm (`cv2.findContours`). The algorithm scans the binary mask line-by-line; upon encountering a transition from the background (`0`) to the object (`1`), it mathematically traces the outer perimeter until the loop is closed. By extracting these continuous arrays of `(x, y)` coordinates, the pipeline easily filters out micro-noise (e.g., false-positives under 2 pixels in area) and renders precise red bounding contours directly over the original color image.

## 🔮 Future Scope: Deep Learning Integration
While the current version uses robust, rule-based contrast analysis and statistical filtering (OpenCV), the architecture is decoupled to allow dropping in an unsupervised Deep Learning model (e.g., PaDiM, PatchCore, or Auto-Encoders) for anomaly detection as part of upcoming Master's studies in Applied Information and Data Science.

## 🛠️ Installation & Setup
1. Create isolated environment: `conda create --name surface-inspection python=3.10`
2. Activate it: `conda activate surface-inspection`
3. Install dependencies: `pip install -r requirements.txt`
4. Run locally: `streamlit run app.py`

*(A synthetic test image `assets/sample_scratch.jpg` is provided in the repository for quick testing.)*