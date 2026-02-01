import streamlit as st
import cv2
import numpy as np

st.title("Industrial Vision: Surface Defect Detector")
st.write("Laden Sie ein Bild der Metalloberfläche hoch, um Defekte zu analysieren.")

# 1. Bild-Upload
uploaded_file = st.file_uploader("Bild wählen...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Bild in ein Format konvertieren, das OpenCV versteht
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    # 2. Bildverarbeitung
    # Konvertierung in Graustufen (reduziert Komplexität)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Rauschunterdrückung (Gaußscher Weichzeichner)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Schwellenwert-Analyse (Thresholding), um Defekte zu isolieren
    # Alles, was dunkler/heller als ein Wert ist, wird markiert
    _, threshold = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY_INV)

    # 3. Anzeige im Dashboard
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, channels="BGR", caption="Original-Oberfläche")
    with col2:
        st.image(threshold, caption="Erkannte Anomalien/Defekte")

    st.success("Analyse abgeschlossen. Schwarze Bereiche markieren potenzielle Kratzer oder Materialfehler.")