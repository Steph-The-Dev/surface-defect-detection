import cv2
import numpy as np
import pandas as pd
import streamlit as st

from src.vision import AdaptiveThreshold, DefectDetectionPipeline, ManualThreshold, OtsuThreshold

# Page config for a modern, wide dashboard look
st.set_page_config(layout="wide", page_title="Industrial Vision: Defect Detector", page_icon="🔍")

st.title("Industrial Vision: Surface Defect Detector")
st.markdown("""
This professional-grade tool evaluates metallic surfaces in real-time. 
Adjust the vision parameters in the sidebar to fine-tune the detection pipeline.
""")

# --- SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.header("⚙️ Pipeline Configuration")

    with st.expander("🛡️ Preprocessing", expanded=True):
        blur_kernel = st.slider(
            "Gaussian Blur Kernel",
            min_value=1,
            max_value=21,
            value=5,
            step=2,
            help="Higher values reduce background noise but might blur out tiny defects.",
        )

    with st.expander("🌓 Thresholding Strategy", expanded=True):
        thresh_method = st.selectbox(
            "Method", ["Adaptive (Local)", "Otsu (Global Auto)", "Manual (Global)"], index=0
        )

        if thresh_method == "Manual (Global)":
            thresh_val = st.slider("Threshold Value", 0, 255, 150)
            strategy = ManualThreshold(threshold_value=thresh_val)
        elif thresh_method == "Otsu (Global Auto)":
            st.info("Otsu's method determines the optimal threshold statistically.")
            strategy = OtsuThreshold()
        else:
            block_size = st.slider("Neighborhood Size", 3, 99, 11, step=2)
            c_constant = st.slider("C Constant", -50, 50, 2)
            strategy = AdaptiveThreshold(block_size=block_size, c_constant=c_constant)

    with st.expander("🧹 Post-Processing", expanded=False):
        morph_size = st.slider(
            "Morphology Kernel",
            1,
            15,
            3,
            step=2,
            help="Used for noise removal (Opening) and gap filling (Closing).",
        )
        min_area = st.number_input("Min. Defect Area (px)", min_value=0.0, value=2.0, step=0.5)

# --- MAIN CONTENT ---
uploaded_file = st.file_uploader(
    "Upload Surface Image (jpg, png, jpeg)...", type=["jpg", "png", "jpeg"]
)

if uploaded_file is not None:
    # 1. Image Loading
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    # 2. Pipeline Execution
    pipeline = DefectDetectionPipeline(
        blur_kernel=blur_kernel, morph_kernel_size=morph_size, min_defect_area=min_area
    )

    try:
        results = pipeline.process(image, strategy)

        # 3. Metrics Display
        st.subheader("📊 Analysis Metrics")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Defect Area (%)", f"{results['defect_percentage']:.2f}%")

        thresh_display = (
            f"{int(results['applied_threshold'])}"
            if results["applied_threshold"] > 0
            else "Adaptive"
        )
        m2.metric("Applied Threshold", thresh_display)

        m3.metric("Anomalies Detected", results["contours_found"])

        status = "FAIL" if results["defect_percentage"] > 1.0 else "PASS"
        m4.metric("Quality Status", status, delta_color="inverse" if status == "FAIL" else "normal")

        st.markdown("---")

        # 4. Results Visualization
        st.subheader("👁️ Vision Pipeline Stages")
        c1, c2, c3 = st.columns(3)

        with c1:
            st.image(image, channels="BGR", caption="1. Original Surface")
        with c2:
            st.image(results["threshold_mask"], caption="2. Processed Mask")
        with c3:
            result_rgb = cv2.cvtColor(results["result_image"], cv2.COLOR_BGR2RGB)
            st.image(result_rgb, caption="3. Highlighted Anomalies")

        # 5. Histogram
        with st.expander("📈 Pixel Intensity Distribution"):
            chart_data = pd.DataFrame(results["histogram"], columns=["Frequency"])
            st.line_chart(chart_data)

    except Exception as e:
        st.error(f"Error processing image: {e}")

else:
    st.info("Awaiting image upload. Use the sidebar to configure the detection sensitivity.")
