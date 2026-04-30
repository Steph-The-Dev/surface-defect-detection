from abc import ABC, abstractmethod

import cv2
import numpy as np


class ThresholdStrategy(ABC):
    """Abstract base class for thresholding strategies (Strategy Pattern)."""

    @abstractmethod
    def apply(self, image: np.ndarray) -> tuple[np.ndarray, float]:
        pass


class OtsuThreshold(ThresholdStrategy):
    def apply(self, image: np.ndarray) -> tuple[np.ndarray, float]:
        val, mask = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return mask, val


class AdaptiveThreshold(ThresholdStrategy):
    def __init__(self, block_size: int = 11, c_constant: int = 2):
        self.block_size = block_size if block_size % 2 != 0 else block_size + 1
        self.c_constant = c_constant

    def apply(self, image: np.ndarray) -> tuple[np.ndarray, float]:
        mask = cv2.adaptiveThreshold(
            image,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            self.block_size,
            self.c_constant,
        )
        return mask, 0.0


class ManualThreshold(ThresholdStrategy):
    def __init__(self, threshold_value: int = 150):
        self.threshold_value = threshold_value

    def apply(self, image: np.ndarray) -> tuple[np.ndarray, float]:
        _, mask = cv2.threshold(image, self.threshold_value, 255, cv2.THRESH_BINARY_INV)
        return mask, float(self.threshold_value)


from typing import Any

class DefectDetectionPipeline:
    """
    Object-Oriented Pipeline for surface defect detection.
    Encapsulates preprocessing, thresholding, morphological cleaning, and contour analysis.
    """

    def __init__(self, blur_kernel: int = 5, morph_kernel_size: int = 3, min_defect_area: float = 2.0):
        self.blur_kernel = blur_kernel if blur_kernel % 2 != 0 else blur_kernel + 1
        self.morph_kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
        self.min_defect_area = min_defect_area

    def process(self, image: np.ndarray, strategy: ThresholdStrategy) -> dict[str, Any]:
        """
        Executes the full vision pipeline on an input image.

        Args:
            image: Input image (BGR).
            strategy: The thresholding strategy to employ.

        Returns:
            Dictionary containing processed results (image, mask, percentage, histogram, etc.).
        """
        # 1. Grayscale & Noise Reduction
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (self.blur_kernel, self.blur_kernel), 0)

        # 2. Thresholding (Binarization)
        threshold_mask, applied_thresh = strategy.apply(blurred)

        # 3. Morphological Cleaning (Phase 2 Enhancement)
        # Opening removes small noise, Closing fills small holes in defects
        clean_mask = cv2.morphologyEx(threshold_mask, cv2.MORPH_OPEN, self.morph_kernel)
        clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, self.morph_kernel)

        # 4. Contour Analysis
        contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_contours = [c for c in contours if cv2.contourArea(c) > self.min_defect_area]

        result_image = image.copy()
        cv2.drawContours(result_image, valid_contours, -1, (0, 0, 255), 2)

        # 5. Metrics & Histogram
        total_pixels = image.shape[0] * image.shape[1]
        defect_pixels = cv2.countNonZero(clean_mask)
        defect_percentage = (defect_pixels / total_pixels) * 100
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()

        return {
            "result_image": result_image,
            "threshold_mask": clean_mask,
            "defect_percentage": defect_percentage,
            "histogram": hist,
            "applied_threshold": applied_thresh,
            "contours_found": len(valid_contours),
        }


# For backward compatibility (optional but recommended during transition)
def process_image(
    image: np.ndarray,
    blur_kernel: int,
    thresh_method: str,
    thresh_val: int,
    block_size: int = 11,
    c_constant: int = 2,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, float]:
    """Bridge function to maintain compatibility with original app.py during migration."""
    strategy: ThresholdStrategy
    if thresh_method == "Otsu (Automatic Global)":
        strategy = OtsuThreshold()
    elif thresh_method == "Adaptive (Local/Fine Details)":
        strategy = AdaptiveThreshold(block_size=block_size, c_constant=c_constant)
    else:
        strategy = ManualThreshold(threshold_value=thresh_val)

    pipeline = DefectDetectionPipeline(blur_kernel=blur_kernel)
    res = pipeline.process(image, strategy)

    return (
        res["result_image"],
        res["threshold_mask"],
        res["defect_percentage"],
        res["histogram"],
        res["applied_threshold"],
    )
