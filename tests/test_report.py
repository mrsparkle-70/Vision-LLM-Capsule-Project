"""
Unit tests for the PDF report generator.
"""

import io
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from report_generator import generate_pdf_report
from utils import generate_inspection_id, get_timestamp_string


# ---------------------------------------------------------------------------
# Report Generation Tests
# ---------------------------------------------------------------------------

def test_generate_pdf_returns_bytes():
    """PDF generation should return bytes."""
    inspection_data = {
        "inspection_id": generate_inspection_id(),
        "timestamp": get_timestamp_string(),
        "image_name": "test.jpg",
        "prediction": "Scratch",
        "confidence": 0.942,
        "bbox": [145, 92, 350, 164],
        "location": "Upper-middle region",
        "severity": "Moderate",
        "status": "REVIEW",
        "description": "A surface scratch was detected.",
        "possible_causes": ["Mechanical abrasion", "Handling"],
        "impact": "May affect surface quality.",
        "recommended_action": "Flag for inspection.",
    }

    pdf_bytes = generate_pdf_report(inspection_data=inspection_data)
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 0


def test_generate_pdf_with_image():
    """PDF generation should work with an annotated image."""
    from PIL import Image

    # Create a simple test image
    img = Image.new("RGB", (100, 100), color="white")

    inspection_data = {
        "inspection_id": generate_inspection_id(),
        "timestamp": get_timestamp_string(),
        "image_name": "test.jpg",
        "prediction": "Good",
        "confidence": 0.99,
        "bbox": None,
        "location": "N/A",
        "severity": "None",
        "status": "PASS",
        "description": "No defects detected.",
        "possible_causes": [],
        "impact": "No quality impact.",
        "recommended_action": "No action required.",
    }

    pdf_bytes = generate_pdf_report(
        inspection_data=inspection_data,
        annotated_image=img,
    )
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 0


def test_generate_pdf_good_capsule():
    """PDF should handle Good capsule case."""
    inspection_data = {
        "inspection_id": generate_inspection_id(),
        "timestamp": get_timestamp_string(),
        "image_name": "good.jpg",
        "prediction": "Good",
        "confidence": 0.99,
        "bbox": None,
        "location": "N/A",
        "severity": "None",
        "status": "PASS",
        "description": "The capsule passed visual inspection.",
        "possible_causes": [],
        "impact": "No quality impact.",
        "recommended_action": "No action required.",
    }

    pdf_bytes = generate_pdf_report(inspection_data=inspection_data)
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 0


def test_generate_pdf_multiple_defects():
    """PDF should handle multiple defects."""
    inspection_data = {
        "inspection_id": generate_inspection_id(),
        "timestamp": get_timestamp_string(),
        "image_name": "multi.jpg",
        "prediction": "Scratch",
        "confidence": 0.88,
        "bbox": [100, 100, 200, 200],
        "location": "Center region",
        "severity": "High",
        "status": "REJECT",
        "description": "Multiple defects detected.",
        "possible_causes": ["Mechanical stress", "Handling"],
        "impact": "May compromise integrity.",
        "recommended_action": "Reject and investigate.",
    }

    pdf_bytes = generate_pdf_report(inspection_data=inspection_data)
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 0


def test_generate_pdf_saves_to_file(tmp_path):
    """PDF should save to a file when output_path is provided."""
    inspection_data = {
        "inspection_id": generate_inspection_id(),
        "timestamp": get_timestamp_string(),
        "image_name": "test.jpg",
        "prediction": "Crack",
        "confidence": 0.9,
        "bbox": [50, 50, 150, 150],
        "location": "Upper-left region",
        "severity": "High",
        "status": "REJECT",
        "description": "A crack was detected.",
        "possible_causes": ["Mechanical stress"],
        "impact": "May compromise integrity.",
        "recommended_action": "Reject.",
    }

    output_path = str(tmp_path / "report.pdf")
    pdf_bytes = generate_pdf_report(
        inspection_data=inspection_data,
        output_path=output_path,
    )

    # File should exist and be non-empty
    assert Path(output_path).exists()
    assert Path(output_path).stat().st_size > 0
    assert isinstance(pdf_bytes, bytes)