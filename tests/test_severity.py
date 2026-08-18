"""
Unit tests for the severity engine.
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from severity import (
    calculate_severity,
    calculate_severities,
    overall_severity,
    inspection_status,
    severity_index,
    escalate,
    classify_area_size,
    SEVERITY_NONE,
    SEVERITY_LOW,
    SEVERITY_MODERATE,
    SEVERITY_HIGH,
    SEVERITY_CRITICAL,
)


# ---------------------------------------------------------------------------
# Severity Calculation Tests
# ---------------------------------------------------------------------------

def test_good_has_no_severity():
    """Good capsules should have no severity."""
    assert calculate_severity("Good") == SEVERITY_NONE


def test_crack_high_severity():
    """Crack should be High severity by default."""
    assert calculate_severity("Crack", confidence=0.9, area_ratio=0.02) == SEVERITY_HIGH


def test_crack_large_area_critical():
    """Large crack should escalate to Critical."""
    assert calculate_severity("Crack", confidence=0.9, area_ratio=0.06) == SEVERITY_CRITICAL


def test_poke_high_severity():
    """Poke should be High severity by default."""
    assert calculate_severity("Poke", confidence=0.9, area_ratio=0.02) == SEVERITY_HIGH


def test_poke_large_area_critical():
    """Large poke should escalate to Critical."""
    assert calculate_severity("Poke", confidence=0.9, area_ratio=0.06) == SEVERITY_CRITICAL


def test_scratch_low_severity():
    """Small scratch should be Low severity."""
    assert calculate_severity("Scratch", confidence=0.7, area_ratio=0.005) == SEVERITY_LOW


def test_scratch_medium_area_moderate():
    """Medium scratch should be Moderate severity."""
    assert calculate_severity("Scratch", confidence=0.7, area_ratio=0.02) == SEVERITY_MODERATE


def test_scratch_large_area_high():
    """Large scratch should be High severity."""
    assert calculate_severity("Scratch", confidence=0.7, area_ratio=0.06) == SEVERITY_HIGH


def test_squeeze_moderate_severity():
    """Squeeze should be Moderate severity by default."""
    assert calculate_severity("Squeeze", confidence=0.7, area_ratio=0.02) == SEVERITY_MODERATE


def test_squeeze_large_area_high():
    """Large squeeze should escalate to High."""
    assert calculate_severity("Squeeze", confidence=0.7, area_ratio=0.06) == SEVERITY_HIGH


def test_contamination_small_moderate_severity():
    """Small contamination should be Moderate severity by default."""
    assert calculate_severity("Contamination", confidence=0.7, area_ratio=0.005) == SEVERITY_MODERATE


def test_contamination_medium_area_upgrade():
    """Medium contamination should escalate to High due to area."""
    assert calculate_severity("Contamination", confidence=0.7, area_ratio=0.02) == SEVERITY_HIGH


def test_contamination_high_confidence_escalates():
    """High-confidence medium contamination should escalate to Critical."""
    assert calculate_severity("Contamination", confidence=0.9, area_ratio=0.02) == SEVERITY_CRITICAL


def test_faulty_imprint_low_severity():
    """Faulty imprint should be Low severity by default."""
    assert calculate_severity("Faulty Imprint", confidence=0.7, area_ratio=0.005) == SEVERITY_LOW


def test_faulty_imprint_medium_area_moderate():
    """Medium faulty imprint should be Moderate."""
    assert calculate_severity("Faulty Imprint", confidence=0.7, area_ratio=0.02) == SEVERITY_MODERATE


# ---------------------------------------------------------------------------
# Severity List Tests
# ---------------------------------------------------------------------------

def test_calculate_severities_adds_severity():
    """calculate_severities should add severity to each detection."""
    detections = [
        {"class": "Scratch", "confidence": 0.7, "area_ratio": 0.02},
        {"class": "Crack", "confidence": 0.8, "area_ratio": 0.01},
    ]
    result = calculate_severities(detections)
    assert len(result) == 2
    assert result[0]["severity"] == SEVERITY_MODERATE
    assert result[1]["severity"] == SEVERITY_HIGH


def test_overall_severity_empty():
    """Empty detections should have no severity."""
    assert overall_severity([]) == SEVERITY_NONE


def test_overall_severity_highest():
    """Overall severity should be the highest among detections."""
    detections = [
        {"severity": SEVERITY_LOW},
        {"severity": SEVERITY_HIGH},
        {"severity": SEVERITY_MODERATE},
    ]
    assert overall_severity(detections) == SEVERITY_HIGH


# ---------------------------------------------------------------------------
# Status Tests
# ---------------------------------------------------------------------------

def test_status_pass():
    """No severity should map to PASS."""
    assert inspection_status(SEVERITY_NONE) == "PASS"


def test_status_review():
    """Low/Moderate severity should map to REVIEW."""
    assert inspection_status(SEVERITY_LOW) == "REVIEW"
    assert inspection_status(SEVERITY_MODERATE) == "REVIEW"


def test_status_reject():
    """High/Critical severity should map to REJECT."""
    assert inspection_status(SEVERITY_HIGH) == "REJECT"
    assert inspection_status(SEVERITY_CRITICAL) == "REJECT"


# ---------------------------------------------------------------------------
# Helper Tests
# ---------------------------------------------------------------------------

def test_severity_index():
    """Severity index should be ordered correctly."""
    assert severity_index(SEVERITY_NONE) == 0
    assert severity_index(SEVERITY_LOW) == 1
    assert severity_index(SEVERITY_MODERATE) == 2
    assert severity_index(SEVERITY_HIGH) == 3
    assert severity_index(SEVERITY_CRITICAL) == 4


def test_escalate():
    """Escalation should work correctly."""
    assert escalate(SEVERITY_LOW, 1) == SEVERITY_MODERATE
    assert escalate(SEVERITY_LOW, 2) == SEVERITY_HIGH
    assert escalate(SEVERITY_HIGH, 1) == SEVERITY_CRITICAL
    assert escalate(SEVERITY_CRITICAL, 1) == SEVERITY_CRITICAL  # Capped


def test_classify_area_size():
    """Area size classification should work."""
    assert classify_area_size(0.005) == "small"
    assert classify_area_size(0.02) == "medium"
    assert classify_area_size(0.06) == "large"