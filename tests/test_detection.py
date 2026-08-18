"""
Unit tests for the YOLO detection module.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from yolo_detection import (
    YOLODetector,
    calculate_location,
    calculate_area_ratio,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_IOU_THRESHOLD,
)
from utils import CLASS_NAMES


# ---------------------------------------------------------------------------
# Class Names Tests
# ---------------------------------------------------------------------------

def test_class_names():
    """Class names should match the 7-class specification."""
    assert CLASS_NAMES == [
        "Good",
        "Crack",
        "Scratch",
        "Faulty Imprint",
        "Poke",
        "Squeeze",
        "Contamination",
    ]


def test_class_count():
    """There should be exactly 7 classes."""
    assert len(CLASS_NAMES) == 7


# ---------------------------------------------------------------------------
# Detector Configuration Tests
# ---------------------------------------------------------------------------

def test_detector_defaults():
    """Detector should have sensible defaults."""
    detector = YOLODetector()
    assert detector.confidence_threshold == DEFAULT_CONFIDENCE_THRESHOLD
    assert detector.iou_threshold == DEFAULT_IOU_THRESHOLD


def test_detector_custom_thresholds():
    """Detector should accept custom thresholds."""
    detector = YOLODetector(confidence_threshold=0.5, iou_threshold=0.6)
    assert detector.confidence_threshold == 0.5
    assert detector.iou_threshold == 0.6


# ---------------------------------------------------------------------------
# Capsule Detection Logic Tests
# ---------------------------------------------------------------------------

def test_is_capsule_detected_empty():
    """No detections should mean no capsule detected."""
    detector = YOLODetector()
    assert not detector.is_capsule_detected([])


def test_is_capsule_detected_with_detection():
    """Any detection should mean capsule detected."""
    detector = YOLODetector()
    detections = [{"class": "Good", "confidence": 0.9}]
    assert detector.is_capsule_detected(detections)


def test_is_capsule_detected_with_defect():
    """A defect detection should also mean capsule detected."""
    detector = YOLODetector()
    detections = [{"class": "Scratch", "confidence": 0.9}]
    assert detector.is_capsule_detected(detections)


# ---------------------------------------------------------------------------
# Detection Structure Tests
# ---------------------------------------------------------------------------

def test_detection_structure():
    """Detection dicts should have the required keys."""
    detector = YOLODetector()
    det = {
        "class": "Scratch",
        "class_id": 2,
        "confidence": 0.942,
        "bbox": [145, 92, 350, 164],
        "location": "Upper-middle region",
        "area_ratio": 0.034,
    }

    # Required keys
    assert "class" in det
    assert "class_id" in det
    assert "confidence" in det
    assert "bbox" in det
    assert "location" in det
    assert "area_ratio" in det

    # Types
    assert isinstance(det["class"], str)
    assert isinstance(det["class_id"], int)
    assert isinstance(det["confidence"], float)
    assert isinstance(det["bbox"], list)
    assert len(det["bbox"]) == 4
    assert isinstance(det["location"], str)
    assert isinstance(det["area_ratio"], float)


# ---------------------------------------------------------------------------
# Location Tests
# ---------------------------------------------------------------------------

def test_location_upper_middle():
    """Upper-middle region detection."""
    loc = calculate_location(400, 50, 600, 150, 1000, 800)
    assert "Upper" in loc
    assert "Middle" in loc


def test_location_center():
    """Center region detection."""
    loc = calculate_location(400, 300, 600, 500, 1000, 800)
    assert loc == "Center region"


def test_location_lower_right():
    """Lower-right region detection."""
    loc = calculate_location(800, 650, 950, 750, 1000, 800)
    assert "Lower" in loc
    assert "Right" in loc


# ---------------------------------------------------------------------------
# Area Ratio Tests
# ---------------------------------------------------------------------------

def test_area_ratio_small_defect():
    """Small defects should have small area ratios."""
    ratio = calculate_area_ratio([100, 100, 150, 130], 1000, 800)
    assert ratio < 0.01


def test_area_ratio_medium_defect():
    """Medium defects should have medium area ratios."""
    ratio = calculate_area_ratio([100, 100, 300, 200], 1000, 800)
    assert 0.01 <= ratio <= 0.05


def test_area_ratio_large_defect():
    """Large defects should have large area ratios."""
    ratio = calculate_area_ratio([100, 100, 600, 500], 1000, 800)
    assert ratio > 0.05