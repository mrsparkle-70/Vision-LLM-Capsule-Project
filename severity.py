"""
Severity Engine for Capsule Vision Inspection.

Responsibilities:
    - Calculate severity for each detected defect
    - Severity is based on defect type + size/area + confidence
    - Severity is NOT equal to confidence
    - Deterministic, explainable rule engine
"""

# ---------------------------------------------------------------------------
# Severity Levels
# ---------------------------------------------------------------------------

SEVERITY_NONE = "None"
SEVERITY_LOW = "Low"
SEVERITY_MODERATE = "Moderate"
SEVERITY_HIGH = "High"
SEVERITY_CRITICAL = "Critical"

SEVERITY_LEVELS = [
    SEVERITY_NONE,
    SEVERITY_LOW,
    SEVERITY_MODERATE,
    SEVERITY_HIGH,
    SEVERITY_CRITICAL,
]

SEVERITY_MEANINGS = {
    SEVERITY_NONE: "No visible defect",
    SEVERITY_LOW: "Minor visible defect",
    SEVERITY_MODERATE: "Defect requiring inspection",
    SEVERITY_HIGH: "Significant quality concern",
    SEVERITY_CRITICAL: "Potentially serious integrity/quality concern",
}

# ---------------------------------------------------------------------------
# Severity Rules
# ---------------------------------------------------------------------------

# Base severity per defect type
BASE_SEVERITY = {
    "Good": SEVERITY_NONE,
    "Crack": SEVERITY_HIGH,
    "Scratch": SEVERITY_LOW,
    "Faulty Imprint": SEVERITY_LOW,
    "Poke": SEVERITY_HIGH,
    "Squeeze": SEVERITY_MODERATE,
    "Contamination": SEVERITY_MODERATE,
}

# Area-ratio modifiers per defect type
#   area_ratio < SMALL_AREA  -> small
#   SMALL_AREA <= area_ratio < LARGE_AREA -> medium
#   area_ratio >= LARGE_AREA -> large
SMALL_AREA = 0.01
LARGE_AREA = 0.05

AREA_UPGRADE_RULES = {
    "Crack": {
        "small": 0,       # High
        "medium": 0,      # High
        "large": 1,       # Critical
    },
    "Poke": {
        "small": 0,       # High
        "medium": 0,      # High
        "large": 1,       # Critical
    },
    "Scratch": {
        "small": 0,       # Low
        "medium": 1,      # Moderate
        "large": 2,       # High
    },
    "Squeeze": {
        "small": 0,       # Moderate
        "medium": 0,      # Moderate
        "large": 1,       # High
    },
    "Contamination": {
        "small": 0,       # Moderate
        "medium": 1,      # High
        "large": 1,       # High
    },
    "Faulty Imprint": {
        "small": 0,       # Low
        "medium": 1,      # Moderate
        "large": 1,       # Moderate
    },
}

# High-confidence escalation rule for ambiguous classes
HIGH_CONFIDENCE_THRESHOLD = 0.85

CONFIDENCE_ESCALATION_CLASSES = {
    "Scratch",
    "Faulty Imprint",
    "Squeeze",
    "Contamination",
}


# ---------------------------------------------------------------------------
# Severity Engine
# ---------------------------------------------------------------------------

def severity_index(severity: str) -> int:
    """Return the numeric index of a severity level."""
    return SEVERITY_LEVELS.index(severity)


def escalate(severity: str, levels: int = 1) -> str:
    """Escalate a severity level by a number of levels (capped at Critical)."""
    idx = severity_index(severity)
    new_idx = min(idx + levels, len(SEVERITY_LEVELS) - 1)
    return SEVERITY_LEVELS[new_idx]


def classify_area_size(area_ratio: float) -> str:
    """Classify an area ratio into small / medium / large."""
    if area_ratio < SMALL_AREA:
        return "small"
    elif area_ratio >= LARGE_AREA:
        return "large"
    else:
        return "medium"


def calculate_severity(
    defect_class: str,
    confidence: float = 1.0,
    area_ratio: float = 0.0,
) -> str:
    """
    Calculate the severity for a single defect detection.

    Severity is a function of:
        - defect type
        - defect area ratio (size)
        - confidence (as a secondary signal only)

    Returns one of: None, Low, Moderate, High, Critical
    """
    if defect_class == "Good":
        return SEVERITY_NONE

    base = BASE_SEVERITY.get(defect_class, SEVERITY_MODERATE)

    # Get area-based upgrade
    area_size = classify_area_size(area_ratio)
    area_rules = AREA_UPGRADE_RULES.get(defect_class, {})
    area_upgrade = area_rules.get(area_size, 0)

    severity = base
    if area_upgrade:
        severity = escalate(severity, area_upgrade)

    # Confidence-based escalation for ambiguous classes
    if (
        confidence >= HIGH_CONFIDENCE_THRESHOLD
        and defect_class in CONFIDENCE_ESCALATION_CLASSES
    ):
        severity = escalate(severity, 1)

    return severity


def calculate_severities(detections: list) -> list:
    """
    Calculate severity for every detection in a list.

    Each detection dict is augmented with a "severity" key.
    """
    out = []
    for det in detections:
        det = dict(det)
        det["severity"] = calculate_severity(
            defect_class=det.get("class", "Good"),
            confidence=det.get("confidence", 1.0),
            area_ratio=det.get("area_ratio", 0.0),
        )
        out.append(det)
    return out


def overall_severity(detections: list) -> str:
    """
    Determine the overall severity for an image given all detections.

    Uses the highest severity level among all defect detections.
    """
    if not detections:
        return SEVERITY_NONE

    highest = SEVERITY_NONE
    for det in detections:
        sev = det.get("severity", SEVERITY_NONE)
        if severity_index(sev) > severity_index(highest):
            highest = sev

    return highest


def inspection_status(severity: str) -> str:
    """
    Map severity to an inspection status recommendation.

    Returns one of: PASS, REVIEW, REJECT
    """
    if severity == SEVERITY_NONE:
        return "PASS"
    elif severity in (SEVERITY_LOW, SEVERITY_MODERATE):
        return "REVIEW"
    elif severity in (SEVERITY_HIGH, SEVERITY_CRITICAL):
        return "REJECT"
    else:
        return "REVIEW"


def severity_description(severity: str) -> str:
    """Return a human-readable description of a severity level."""
    return SEVERITY_MEANINGS.get(severity, severity)
