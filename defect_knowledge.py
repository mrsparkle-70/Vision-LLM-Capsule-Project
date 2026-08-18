"""
Defect Knowledge Base for Capsule Vision Inspection.

Responsibilities:
    - Known defect information
    - Possible causes
    - Quality implications
    - Recommended actions
"""

# ---------------------------------------------------------------------------
# Defect Knowledge
# ---------------------------------------------------------------------------

DEFECT_KNOWLEDGE = {
    "Good": {
        "description": "The capsule appears normal with no visible defects.",
        "possible_causes": [],
        "impact": "No quality impact. The capsule meets visual inspection criteria.",
        "recommended_action": "No action required. The capsule passes visual inspection.",
    },
    "Crack": {
        "description": (
            "A crack-like defect was detected. Cracks are visible fractures "
            "on the capsule surface that may compromise structural integrity."
        ),
        "possible_causes": [
            "Mechanical stress during handling",
            "Excessive handling pressure",
            "Manufacturing-related stress",
            "Packaging pressure or impact",
        ],
        "impact": (
            "Cracks may compromise capsule integrity, potentially exposing "
            "the contents and affecting product quality and safety."
        ),
        "recommended_action": (
            "Flag the capsule for immediate quality inspection. "
            "Investigate handling, packaging, and manufacturing processes."
        ),
    },
    "Scratch": {
        "description": (
            "A surface scratch was detected. Scratches are superficial "
            "abrasions on the capsule shell that may affect surface quality."
        ),
        "possible_causes": [
            "Mechanical abrasion during production",
            "Contact with manufacturing equipment",
            "Handling during sorting or packaging",
            "Transportation-related friction",
        ],
        "impact": (
            "Scratches primarily affect cosmetic surface quality and may "
            "not compromise structural integrity, but should still be "
            "evaluated against QC criteria."
        ),
        "recommended_action": (
            "Flag the capsule for quality review. Monitor equipment contact "
            "points and handling procedures."
        ),
    },
    "Faulty Imprint": {
        "description": (
            "A faulty imprint was detected. The printing on the capsule is "
            "incorrect, damaged, or unclear."
        ),
        "possible_causes": [
            "Printing alignment issues",
            "Ink or printing equipment malfunction",
            "Calibration errors in the printing process",
            "Insufficient ink or worn printing plates",
        ],
        "impact": (
            "Faulty imprints can cause identification issues and may affect "
            "traceability and regulatory compliance."
        ),
        "recommended_action": (
            "Flag for inspection. Verify printing equipment calibration "
            "and imprint quality during the next production run."
        ),
    },
    "Poke": {
        "description": (
            "A poke/puncture-like defect was detected. This indicates a "
            "hole or puncture in the capsule shell."
        ),
        "possible_causes": [
            "Sharp mechanical contact",
            "Equipment malfunction",
            "Handling damage from sharp objects",
            "Pinching during packaging",
        ],
        "impact": (
            "Punctures compromise capsule integrity and may expose or "
            "contaminate the contents. This is a significant quality concern."
        ),
        "recommended_action": (
            "Reject the capsule and flag for immediate review. "
            "Inspect equipment for sharp edges or malfunctioning parts."
        ),
    },
    "Squeeze": {
        "description": (
            "A squeeze/deformation defect was detected. The capsule shows "
            "signs of compression or deformation from its normal shape."
        ),
        "possible_causes": [
            "Mechanical pressure during handling",
            "Packaging compression",
            "Improper storage conditions",
            "Excessive force during sorting",
        ],
        "impact": (
            "Deformation may affect capsule sealing, dosage integrity, "
            "and visual appearance."
        ),
        "recommended_action": (
            "Flag for quality inspection. Review packaging and handling "
            "processes to reduce compression forces."
        ),
    },
    "Contamination": {
        "description": (
            "Contamination was detected. Foreign or undesired material "
            "is present on the capsule surface."
        ),
        "possible_causes": [
            "Foreign material in the production environment",
            "Handling or environmental contamination",
            "Packaging or process contamination",
            "Insufficient cleaning or filtration",
        ],
        "impact": (
            "Contamination can introduce foreign matter into the product, "
            "posing potential safety and quality risks."
        ),
        "recommended_action": (
            "Reject the capsule. Investigate the production environment, "
            "raw materials, and handling procedures for contamination sources."
        ),
    },
}

# ---------------------------------------------------------------------------
# Fallback knowledge for unknown classes
# ---------------------------------------------------------------------------

FALLBACK_KNOWLEDGE = {
    "description": "A defect was detected on the capsule surface.",
    "possible_causes": [
        "Manufacturing process variation",
        "Handling damage",
        "Equipment contact",
    ],
    "impact": (
        "The detected condition should be evaluated according to "
        "quality control criteria."
    ),
    "recommended_action": (
        "Flag the capsule for quality inspection and further evaluation."
    ),
}


# ---------------------------------------------------------------------------
# Accessor Functions
# ---------------------------------------------------------------------------

def get_defect_knowledge(defect_class: str) -> dict:
    """Return the knowledge entry for a defect class, or fallback if unknown."""
    return DEFECT_KNOWLEDGE.get(defect_class, FALLBACK_KNOWLEDGE)


def get_possible_causes(defect_class: str) -> list:
    """Return possible causes for a defect class."""
    return get_defect_knowledge(defect_class).get("possible_causes", [])


def get_impact(defect_class: str) -> str:
    """Return potential quality impact for a defect class."""
    return get_defect_knowledge(defect_class).get("impact", FALLBACK_KNOWLEDGE["impact"])


def get_recommended_action(defect_class: str) -> str:
    """Return recommended action for a defect class."""
    return get_defect_knowledge(defect_class).get(
        "recommended_action", FALLBACK_KNOWLEDGE["recommended_action"]
    )


def get_description(defect_class: str) -> str:
    """Return the description for a defect class."""
    return get_defect_knowledge(defect_class).get(
        "description", FALLBACK_KNOWLEDGE["description"]
    )