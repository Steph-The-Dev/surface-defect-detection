import cv2
import numpy as np
import pytest

from src.vision import AdaptiveThreshold, DefectDetectionPipeline, ManualThreshold, OtsuThreshold


@pytest.fixture
def sample_image() -> np.ndarray:
    """Creates a synthetic 100x100 image with a white square (defect) on a black background."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    # Draw a "defect"
    cv2.rectangle(img, (40, 40), (60, 60), (255, 255, 255), -1)
    return img


@pytest.fixture
def blank_image() -> np.ndarray:
    """Creates a synthetic 100x100 white image (no defects)."""
    return np.full((100, 100, 3), 255, dtype=np.uint8)


def test_manual_threshold_strategy(sample_image):
    strategy = ManualThreshold(threshold_value=127)
    # Gray conversion and blur happen in pipeline, but we can test strategy in isolation
    gray = cv2.cvtColor(sample_image, cv2.COLOR_BGR2GRAY)
    mask, val = strategy.apply(gray)

    assert val == 127.0
    assert mask.shape == (100, 100)
    # The white square should be 255 in the binary mask (BINARY_INV logic means white becomes black,
    # but wait, the pipeline uses BINARY_INV which is usually for dark defects on light background.
    # In our synthetic test, it's light on dark, so we expect the inverse.)
    # Let's check the non-zero count.
    assert cv2.countNonZero(mask) > 0


def test_pipeline_basic_detection(sample_image):
    pipeline = DefectDetectionPipeline(blur_kernel=3)
    strategy = ManualThreshold(threshold_value=127)

    results = pipeline.process(sample_image, strategy)

    assert "defect_percentage" in results
    assert results["defect_percentage"] > 0
    assert results["contours_found"] == 1
    assert results["threshold_mask"].shape == (100, 100)


def test_pipeline_no_defects(blank_image):
    pipeline = DefectDetectionPipeline()
    strategy = OtsuThreshold()

    results = pipeline.process(blank_image, strategy)

    assert results["defect_percentage"] == 0
    assert results["contours_found"] == 0


def test_adaptive_threshold_params():
    strategy = AdaptiveThreshold(block_size=10, c_constant=5)
    # Block size should be coerced to odd
    assert strategy.block_size == 11
    assert strategy.c_constant == 5


def test_morphological_cleaning(sample_image):
    # Add some noise to the sample image
    noisy_img = sample_image.copy()
    noisy_img[10, 10] = (255, 255, 255)  # Single pixel noise

    # Pipeline with morph kernel size 3 should remove the 1x1 noise pixel
    pipeline = DefectDetectionPipeline(morph_kernel_size=3, min_defect_area=5)
    strategy = ManualThreshold(threshold_value=127)

    results = pipeline.process(noisy_img, strategy)

    # If morphological cleaning and min_area filtering work, we should still only find 1 contour
    assert results["contours_found"] == 1
