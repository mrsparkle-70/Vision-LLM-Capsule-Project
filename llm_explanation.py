"""
LLM Explanation Module for Capsule Vision Inspection.

Responsibilities:
    - Groq API integration
    - Prompt construction
    - LLM response parsing
    - Deterministic fallback when LLM is unavailable
"""

import json
import os
import hashlib
from typing import Optional

from defect_knowledge import (
    get_defect_knowledge,
    get_possible_causes,
    get_impact,
    get_recommended_action,
    get_description,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "llama-3.1-8b-instant"
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_TOKENS = 500

SYSTEM_PROMPT = """You are a pharmaceutical visual quality inspection assistant.

You do NOT perform visual detection. The computer vision model has already performed detection.

RULES:
- Never invent a defect.
- Never claim to have seen an image.
- Never change the detected defect class.
- Never change the confidence.
- Never change the bounding box.
- Never invent a specific machine, operator, batch, material, or manufacturing event.
- Possible causes must be presented as possibilities, not confirmed root causes.
- Generate explanations only from supplied detection data and predefined defect knowledge.

Respond in the following format:
Description: <brief description of what was detected>
Possible Causes:
- <cause 1>
- <cause 2>
- <cause 3>
Potential Impact: <impact statement>
Recommended Action: <action statement>
"""


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

class LLMCache:
    """Simple in-memory cache for LLM responses to avoid redundant API calls."""

    def __init__(self, max_entries: int = 100):
        self._cache = {}
        self._max_entries = max_entries

    def _key(self, detections: list) -> str:
        """Generate a cache key from the structured detection data."""
        data = json.dumps(detections, sort_keys=True, default=str)
        return hashlib.sha256(data.encode()).hexdigest()

    def get(self, detections: list) -> Optional[str]:
        """Retrieve a cached response if available."""
        key = self._key(detections)
        return self._cache.get(key)

    def set(self, detections: list, response: str) -> None:
        """Cache a response."""
        key = self._key(detections)
        if len(self._cache) >= self._max_entries:
            # Simple eviction: clear oldest (first inserted)
            self._cache.pop(next(iter(self._cache)))
        self._cache[key] = response


# Global cache instance
_llm_cache = LLMCache()


# ---------------------------------------------------------------------------
# Prompt Construction
# ---------------------------------------------------------------------------

def build_prompt(detections: list) -> str:
    """
    Build a structured prompt for the LLM from detection data.

    Args:
        detections: List of detection dicts with class, confidence,
                    location, severity, area_ratio.

    Returns:
        A prompt string for the LLM.
    """
    if not detections:
        return (
            "No defects were detected in the capsule image. "
            "The capsule appears to be in good condition."
        )

    # Build structured facts
    facts = []
    for det in detections:
        facts.append(
            {
                "defect": det.get("class", "Unknown"),
                "confidence": round(det.get("confidence", 0.0), 4),
                "location": det.get("location", "Unknown"),
                "severity": det.get("severity", "Unknown"),
                "area_ratio": round(det.get("area_ratio", 0.0), 4),
                "bbox": det.get("bbox", []),
            }
        )

    prompt = (
        "You are a pharmaceutical quality inspection assistant.\n\n"
        "The computer vision model has detected the following defects "
        "in a capsule image. Generate an explanation based ONLY on "
        "these facts. Do not invent additional defects.\n\n"
        f"Detection Data:\n{json.dumps(facts, indent=2)}\n\n"
        "Provide:\n"
        "1. A brief description of what was detected.\n"
        "2. Possible causes (as possibilities, not confirmed root causes).\n"
        "3. Potential quality impact.\n"
        "4. Recommended action for QC personnel.\n\n"
        "Format your response as:\n"
        "Description: ...\n"
        "Possible Causes:\n- ...\n- ...\n"
        "Potential Impact: ...\n"
        "Recommended Action: ..."
    )
    return prompt


# ---------------------------------------------------------------------------
# LLM Client
# ---------------------------------------------------------------------------

class GroqLLMClient:
    """Client for the Groq API with deterministic fallback."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ):
        self.api_key = api_key or os.environ.get("GROQ_API_KEY", "")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = None

    def _get_client(self):
        """Lazily initialize the Groq client."""
        if self._client is None:
            if not self.api_key:
                raise ValueError("GROQ_API_KEY is not set.")
            try:
                from groq import Groq
                self._client = Groq(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "Groq SDK is not installed. Run: pip install groq"
                )
        return self._client

    def generate(self, detections: list) -> str:
        """
        Generate an explanation for the given detections.

        Falls back to a deterministic template if the LLM is unavailable.
        """
        # Check cache first
        cached = _llm_cache.get(detections)
        if cached:
            return cached

        try:
            client = self._get_client()
            prompt = build_prompt(detections)

            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            text = response.choices[0].message.content.strip()
            _llm_cache.set(detections, text)
            return text

        except Exception as exc:
            # Deterministic fallback
            fallback = generate_fallback_explanation(detections)
            _llm_cache.set(detections, fallback)
            return fallback


# ---------------------------------------------------------------------------
# Deterministic Fallback
# ---------------------------------------------------------------------------

def generate_fallback_explanation(detections: list) -> str:
    """
    Generate a deterministic explanation from defect knowledge.

    Used when the LLM API is unavailable or fails.
    """
    if not detections:
        return (
            "No defects were detected in the capsule image. "
            "The capsule appears to be in good condition."
        )

    parts = []
    for det in detections:
        defect_class = det.get("class", "Unknown")
        location = det.get("location", "unknown region")
        severity = det.get("severity", "Unknown")
        confidence = det.get("confidence", 0.0)

        knowledge = get_defect_knowledge(defect_class)

        parts.append(
            f"A {defect_class.lower()} defect was detected in the {location} "
            f"with {severity.lower()} severity "
            f"(confidence: {confidence:.1%})."
        )
        parts.append("")
        parts.append(knowledge["description"])
        parts.append("")
        parts.append("Possible Causes:")
        for cause in knowledge["possible_causes"]:
            parts.append(f"- {cause}")
        parts.append("")
        parts.append(f"Potential Impact: {knowledge['impact']}")
        parts.append("")
        parts.append(f"Recommended Action: {knowledge['recommended_action']}")
        parts.append("")
        parts.append("---")
        parts.append("")

    return "\n".join(parts).strip()


# ---------------------------------------------------------------------------
# Convenience Functions
# ---------------------------------------------------------------------------

def generate_explanation(detections: list, api_key: Optional[str] = None) -> str:
    """
    Generate an explanation for detections using the LLM with fallback.

    This is the main entry point used by the application.
    """
    client = GroqLLMClient(api_key=api_key)
    return client.generate(detections)


def parse_explanation(text: str) -> dict:
    """
    Parse an LLM explanation into structured fields.

    Returns a dict with keys: description, possible_causes, impact, recommended_action.
    Falls back to extracting from the raw text if parsing fails.
    """
    result = {
        "description": "",
        "possible_causes": [],
        "impact": "",
        "recommended_action": "",
    }

    if not text:
        return result

    lines = text.strip().split("\n")
    current_section = None

    for line in lines:
        line = line.strip()
        if not line:
            continue

        lower = line.lower()

        if lower.startswith("description:"):
            current_section = "description"
            result["description"] = line.split(":", 1)[1].strip()
        elif lower.startswith("possible causes:"):
            current_section = "possible_causes"
        elif lower.startswith("potential impact:"):
            current_section = "impact"
            result["impact"] = line.split(":", 1)[1].strip()
        elif lower.startswith("recommended action:"):
            current_section = "recommended_action"
            result["recommended_action"] = line.split(":", 1)[1].strip()
        elif current_section == "possible_causes" and line.startswith("-"):
            result["possible_causes"].append(line.lstrip("- ").strip())
        elif current_section == "description" and result["description"]:
            result["description"] += " " + line
        elif current_section == "impact" and result["impact"]:
            result["impact"] += " " + line
        elif current_section == "recommended_action" and result["recommended_action"]:
            result["recommended_action"] += " " + line

    return result