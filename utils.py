"""
Utility functions for the Capsule Vision Inspection Assistant.

Responsibilities:
    - Image validation
    - File naming
    - Common utilities
"""

import os
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp", "bmp", "tiff"}
MAX_IMAGE_DIMENSION = 1280
MAX_FILE_SIZE_MB = 10

CLASS_NAMES = [
    "Good",
    "Crack",
    "Scratch",
    "Faulty Imprint",
    "Poke",
    "Squeeze",
    "Contamination",
]

CLASS_IDS = {name: idx for idx, name in enumerate(CLASS_NAMES)}

REGION_NAMES = [
    "Upper-left",
    "Upper-middle",
    "Upper-right",
    "Middle-left",
    "Center",
    "Middle-right",
    "Lower-left",
    "Lower-middle",
    "Lower-right",
]


# ---------------------------------------------------------------------------
# Image Validation
# ---------------------------------------------------------------------------

def validate_image_file(file_path: str) -> bool:
    """Validate that a file is a supported image format."""
    extension = Path(file_path).suffix.lstrip(".").lower()
    return extension in ALLOWED_EXTENSIONS


def validate_image_size(file_path: str, max_mb: int = MAX_FILE_SIZE_MB) -> bool:
    """Validate that an image file is within the maximum allowed size."""
    size_bytes = os.path.getsize(file_path)
    return size_bytes <= max_mb * 1024 * 1024


def load_validated_image(file_path: str) -> Image.Image:
    """
    Load an image and validate it.

    Returns a PIL Image in RGB mode, or raises a ValueError.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Image file not found: {file_path}")

    if not validate_image_file(file_path):
        raise ValueError("Unsupported image format. Use JPG, JPEG, PNG, or WEBP.")

    if not validate_image_size(file_path):
        raise ValueError(
            f"Image exceeds the maximum allowed size of {MAX_FILE_SIZE_MB} MB."
        )

    try:
        image = Image.open(file_path)
        image.load()
    except Exception as exc:
        raise ValueError(f"Corrupt or unreadable image: {exc}")

    if image.mode != "RGB":
        image = image.convert("RGB")

    # Cap maximum dimension for CPU performance
    max_dim = max(image.size)
    if max_dim > MAX_IMAGE_DIMENSION:
        scale = MAX_IMAGE_DIMENSION / max_dim
        new_size = (int(image.width * scale), int(image.height * scale))
        image = image.resize(new_size, Image.LANCZOS)

    return image


def image_to_numpy(image: Image.Image) -> np.ndarray:
    """Convert a PIL Image (RGB) to a NumPy array in RGB order."""
    return np.array(image)


# ---------------------------------------------------------------------------
# File Naming
# ---------------------------------------------------------------------------

def generate_inspection_id(prefix: str = "CAP") -> str:
    """Generate a unique inspection ID."""
    date_part = datetime.now().strftime("%Y%m%d")
    unique_part = uuid.uuid4().hex[:4].upper()
    return f"{prefix}-{date_part}-{unique_part}"


def safe_filename(filename: str) -> str:
    """Sanitize a filename for safe storage."""
    basename = Path(filename).name
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in basename)


def get_timestamp_string() -> str:
    """Return a formatted timestamp string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------------------
# Generic Helpers
# ---------------------------------------------------------------------------

def clamp(value: float, low: float, high: float) -> float:
    """Clamp a value between low and high."""
    return max(low, min(high, value))


def format_confidence(value: float) -> str:
    """Format a confidence value as a percentage string."""
    return f"{value * 100:.1f}%"