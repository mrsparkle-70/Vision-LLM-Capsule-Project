"""
Unit tests for location and area calculation.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from yolo_detection import calculate_location, calculate_area_ratio


# ---------------------------------------------------------------------------
# Location Calculation Tests
# ---------------------------------------------------------------------------

def test_center_location():
    """Center of image should be 'Center region'."""
    loc = calculate_location(400, 300, 600, 500, 1000, 800)
    assert loc == "Center region"


def test_upper_left_location():
    """Upper-left region."""
    loc = calculate_location(50, 50, 150, 150, 1000, 800)
    assert loc == "Upper-Left region"


def test_upper_middle_location():
    """Upper-middle region."""
    loc = calculate_location(400, 50, 600, 150, 1000, 800)
    assert loc == "Upper-Middle region"


def test_upper_right_location():
    """Upper-right region."""
    loc = calculate_location(800, 50, 950, 150, 1000, 800)
    assert loc == "Upper-Right region"


def test_middle_left_location():
    """Middle-left region."""
    loc = calculate_location(50, 300, 150, 500, 1000, 800)
    assert loc == "Middle-Left region"


def test_middle_right_location():
    """Middle-right region."""
    loc = calculate_location(800, 300, 950, 500, 1000, 800)
    assert loc == "Middle-Right region"


def test_lower_left_location():
    """Lower-left region."""
    loc = calculate_location(50, 650, 150, 750, 1000, 800)
    assert loc == "Lower-Left region"


def test_lower_middle_location():
    """Lower-middle region."""
    loc = calculate_location(400, 650, 600, 750, 1000, 800)
    assert loc == "Lower-Middle region"


def test_lower_right_location():
    """Lower-right region."""
    loc = calculate_location(800, 650, 950, 750, 1000, 800)
    assert loc == "Lower-Right region"


def test_location_deterministic():
    """Location should be deterministic for the same input."""
    loc1 = calculate_location(100, 100, 200, 200, 500, 500)
    loc2 = calculate_location(100, 100, 200, 200, 500, 500)
    assert loc1 == loc2


# ---------------------------------------------------------------------------
# Area Ratio Tests
# ---------------------------------------------------------------------------

def test_area_ratio_full_image():
    """A box covering the full image should have ratio 1.0."""
    ratio = calculate_area_ratio([0, 0, 1000, 800], 1000, 800)
    assert abs(ratio - 1.0) < 1e-6


def test_area_ratio_quarter():
    """A box covering a quarter of the image should have ratio 0.25."""
    ratio = calculate_area_ratio([0, 0, 500, 400], 1000, 800)
    assert abs(ratio - 0.25) < 1e-6


def test_area_ratio_small():
    """A small box should have a small ratio."""
    ratio = calculate_area_ratio([100, 100, 200, 200], 1000, 800)
    assert ratio < 0.05


def test_area_ratio_zero_size():
    """A zero-size box should have ratio 0."""
    ratio = calculate_area_ratio([100, 100, 100, 100], 1000, 800)
    assert ratio == 0.0


def test_area_ratio_in_bounds():
    """Area ratio should always be in [0, 1]."""
    for bbox in [
        [0, 0, 100, 100],
        [50, 50, 500, 400],
        [200, 100, 800, 700],
    ]:
        ratio = calculate_area_ratio(bbox, 1000, 800)
        assert 0.0 <= ratio <= 1.0