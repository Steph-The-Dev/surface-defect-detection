import streamlit as st
import cv2
import numpy as np
import pandas as pd
from src.vision import process_image

# Page config for a modern, wide dashboard look
st.set_page_config(layout="wide", page_title="Industrial Vision: Defect Detector", page_icon="🔍")

st.title("Industrial Vision: Surface Defect Detector")
st.write("Upload a metal surface image to analyze defects, or adjust the parameters in the sidebar in real-time.")

# --- SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.header("⚙️ Vision Parameters")
    st.write("Tweak the algorithm settings dynamically.")
    
    blur_kernel = st.slider("Blur Kernel Size", min_value=1, max_value=21, value=5, step=2, help="Higher values reduce background noise but might blur out tiny defects. Must be an odd number.")
    
    st.markdown("---")
    thresh_method = st.selectbox(
        "Thresholding Method", 
        ["Adaptive (Local/Fine Details)", "Otsu (Automatic Global)", "Manual (Global)"],
        index=0,
        help="Global methods apply one threshold to the entire image. Adaptive calculates it locally, which is far better for uneven lighting and fine scratches."
    )
    
    # Initialize defaults
    thresh_val = 150
    block_size = 11
    c_constant = 2
    
    # Conditional UI based on method chosen
    if thresh_method == "Manual (Global)":
        thresh_val = st.slider("Manual Threshold Value", min_value=0, max_value=255, value=150, help="Pixels darker than this value will be flagged as defects.")
        
    elif thresh_method == "Otsu (Automatic Global)":
        st.info("Otsu's method is active. The global threshold is determined dynamically from the histogram.")
        
    elif thresh_method == "Adaptive (Local/Fine Details)":
        st.write("🔍 **Fine-Tuning Adaptive Threshold**")
        block_size = st.slider("Block Size (Neighborhood)", min_value=3, max_value=99, value=11, step=2, help="The size of the local region used to calculate the threshold. Larger blocks ignore larger gradients.")
        c_constant = st.slider("Constant (C)", min_value=-50, max_value=50, value=2, help="A constant subtracted from the local mean. A lower value makes it more sensitive to very faint scratches.")

# --- MAIN CONTENT ---
uploaded_file = st.file_uploader("Upload Image (jpg, png, jpeg)...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # 1. Convert image to a format OpenCV understands
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1) 
    
    # 2. Image Processing via our new module
    result_image, threshold_mask, defect_pct, hist_data, actual_thresh = process_image(
        image=image, 
        blur_kernel=blur_kernel, 
        thresh_method=thresh_method, 
        thresh_val=thresh_val,
        block_size=block_size,
        c_constant=c_constant
    )
    
    # 3. Metrics Display
    st.subheader("📊 Analysis Metrics")
    metric_cols = st.columns(3)
    metric_cols[0].metric("Defect Area (%)", f"{defect_pct:.2f}%", "- Anomalies Detected" if defect_pct > 0 else "Clean")
    
    if thresh_method == "Adaptive (Local/Fine Details)":
        metric_cols[1].metric("Applied Threshold", "Local (Varying)", "- Adaptive", delta_color="off")
    else:
        metric_cols[1].metric("Applied Threshold", f"{int(actual_thresh)}", f"- {thresh_method.split()[0]}", delta_color="off")
        
    metric_cols[2].metric("Quality Status", "FAIL" if defect_pct > 1.0 else "PASS", delta_color="inverse")
    
    st.markdown("---")

    # 4. Display Images
    st.subheader("👁️ Vision Results")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.image(image, channels="BGR", caption="1. Original Surface")
    with col2:
        st.image(threshold_mask, caption="2. Defect Mask (B/W)")
    with col3:
        # Convert BGR (OpenCV) to RGB (Streamlit display standard)
        result_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
        st.image(result_rgb, caption="3. Detected Anomalies (Highlighted)")

    # 5. Data Visualization: Histogram
    st.markdown("---")
    st.subheader("📈 Pixel Intensity Distribution (Histogram)")
    st.write("This chart represents the image pixels. The X-axis is brightness (0=Black, 255=White).")
    
    # Convert numpy histogram to a Pandas DataFrame for Streamlit's native charting
    chart_data = pd.DataFrame(hist_data, columns=['Frequency'])
    st.line_chart(chart_data)

else:
    st.info("Please upload an image to begin testing the vision algorithm. Adjust parameters in the sidebar on the left.")