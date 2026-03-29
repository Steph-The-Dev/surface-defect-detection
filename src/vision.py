import cv2
import numpy as np

def process_image(image: np.ndarray, blur_kernel: int, thresh_method: str, thresh_val: int, block_size: int = 11, c_constant: int = 2):
    """
    Processes the image to detect surface defects, enhanced for fine scratches.
    """
    # 1. Convert to Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 2. Noise Reduction
    if blur_kernel % 2 == 0:
        blur_kernel += 1
    blurred = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
    
    # 3. Thresholding (Binarization)
    applied_thresh = thresh_val # Default tracking
    
    if thresh_method == "Otsu (Automatic Global)":
        ret, threshold_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        applied_thresh = ret
        
    elif thresh_method == "Adaptive (Local/Fine Details)":
        # Ensure block size is odd and > 1
        if block_size % 2 == 0:
            block_size += 1
        if block_size < 3:
            block_size = 3
            
        # Adaptive thresholding calculates the threshold for small regions
        threshold_mask = cv2.adaptiveThreshold(
            blurred, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 
            block_size, 
            c_constant
        )
        applied_thresh = 0 # Not a single global value anymore
        
    else:  # Manual
        _, threshold_mask = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY_INV)

    # 4. Contour Detection (Removing noise)
    contours, _ = cv2.findContours(threshold_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Optional filtering: Ignore completely microscopic noise specs (e.g. area < 2 pixels)
    valid_contours = [c for c in contours if cv2.contourArea(c) > 2]
    
    result_image = image.copy()
    cv2.drawContours(result_image, valid_contours, -1, (0, 0, 255), 2)
    
    # 5. Metrics
    total_pixels = image.shape[0] * image.shape[1]
    defect_pixels = cv2.countNonZero(threshold_mask) 
    defect_percentage = (defect_pixels / total_pixels) * 100
    
    # 6. Histogram
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    
    return result_image, threshold_mask, defect_percentage, hist.flatten(), applied_thresh
