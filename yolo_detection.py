"""
YOLO Detection Module for Capsule Vision Inspection.

Responsibilities:
    - Load detection model
    - Receive image
    - Run inference
    - Extract detections (class, confidence, bbox)
    - Calculate defect location
    - Calculate defect area ratio
    - Draw annotations
    - Return structured results
"""

import logging
import os
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from utils import CLASS_NAMES, clamp

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_CONFIDENCE_THRESHOLD = 0.25
DEFAULT_IOU_THRESHOLD = 0.45
MODEL_PATH = Path("models/best.pt")

# Standard YOLOv8n COCO checkpoint used as a fallback so the app can run
# end-to-end before the capsule model has been trained / uploaded.
FALLBACK_MODEL_URL = (
    "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt"
)

# Colors for annotation drawing (BGR for OpenCV compatibility)
CLASS_COLORS_BGR = {
    0: (0, 255, 0),     # Good - Green
    1: (0, 0, 255),     # Crack - Red
    2: (255, 165, 0),   # Scratch - Orange
    3: (255, 0, 255),   # Faulty Imprint - Magenta
    4: (0, 255, 255),   # Poke - Yellow
    5: (255, 0, 0),     # Squeeze - Blue
    6: (128, 0, 128),   # Contamination - Purple
}

CLASS_COLORS_RGB = {k: tuple(reversed(v)) for k, v in CLASS_COLORS_BGR.items()}

CAPSULE_CLASS_IDS = {0}  # "Good" acts as the capsule detection class


# ---------------------------------------------------------------------------
# Model Availability
# ---------------------------------------------------------------------------

def ensure_model_available(model_path: Path = MODEL_PATH) -> Path:
    """
    Ensure a detection model exists at the expected path.

    If the trained 'best.pt' is not present, downloads the standard
    YOLOv8n COCO checkpoint as a fallback so the pipeline can run in any
    environment (local or Streamlit Community Cloud). Replace
    models/best.pt with your trained capsule model for real defect detection.

    Returns the resolved model path.
    """
    if model_path.exists():
        return model_path

    # Try the fallback (yolov8n.pt) in the models/ dir first
    fallback_path = model_path.parent / "yolov8n.pt"
    if fallback_path.exists():
        logger.warning(
            "Trained model %s not found. Using fallback %s (COCO classes; "
            "not capsule-specific). Replace with your trained best.pt.",
            model_path,
            fallback_path,
        )
        return fallback_path

    # Download fallback model
    os.makedirs(model_path.parent, exist_ok=True)
    logger.info("Downloading fallback YOLO model from %s ...", FALLBACK_MODEL_URL)
    urllib.request.urlretrieve(FALLBACK_MODEL_URL, fallback_path)
    logger.info("Fallback model downloaded to %s", fallback_path)
    return fallback_path


# ---------------------------------------------------------------------------
# Model Loading
# ---------------------------------------------------------------------------

class YOLODetector:
    """
    Wrapper around the Ultralytics YOLO detection model.

    If 'best.pt' does not exist yet, the standard YOLOv8n checkpoint is
    downloaded automatically so the pipeline can be tested in any
    environment (locally or in Streamlit Community Cloud).
    """

    def __init__(
        self,
        model_path: str = None,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    ):
        self.model_path = model_path or str(ensure_model_available(MODEL_PATH))
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self._model = None

    def load_model(self):
        """Load the YOLO model. Uses lazy loading for efficiency."""
        if self._model is None:
            try:
                from ultralytics import YOLO

                if not os.path.exists(self.model_path):
                    # Second chance: ensure fallback / download is available
                    self.model_path = str(ensure_model_available())
                    if not os.path.exists(self.model_path):
                        raise FileNotFoundError(
                            f"Detection model cannot be located at: {self.model_path}. "
                            "Please train the capsule model and place best.pt in models/."
                        )
                self._model = YOLO(self.model_path)
            except ImportError:
                raise ImportError(
                    "Ultralytics is not installed. Run: pip install ultralytics"
                )
        return self._model

    def detect(self, image: Image.Image) -> list:
        """
        Run YOLO inference on a PIL image and return structured detections.

        Returns:
            List of detection dicts:
            [
                {
                    "class": "Scratch",
                    "class_id": 2,
                    "confidence": 0.942,
                    "bbox": [x1, y1, x2, y2],
                    "location": "Upper-middle region",
                    "area_ratio": 0.034,
                },
                ...
            ]
        """
        model = self.load_model()
        image_width, image_height = image.size

        # Convert PIL to numpy (RGB) for Ultralytics
        img_np = np.array(image)

        results = model.predict(
            source=img_np,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            verbose=False,
        )

        detections = []
        if results is None or len(results) == 0:
            return detections

        result = results[0]
        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            return detections

        # Extract box data
        xyxy = boxes.xyxy.cpu().numpy()          # [N, 4] in pixel coords
        confs = boxes.conf.cpu().numpy()          # [N]
        class_ids = boxes.cls.cpu().numpy().astype(int)  # [N]

        for i in range(len(boxes)):
            x1, y1, x2, y2 = [float(v) for v in xyxy[i]]
            confidence = float(confs[i])
            class_id = int(class_ids[i])

            # Clamp to image bounds
            x1 = clamp(x1, 0, image_width)
            y1 = clamp(y1, 0, image_height)
            x2 = clamp(x2, x1, image_width)
            y2 = clamp(y2, y1, image_height)

            # Filter by confidence (double safety)
            if confidence < self.confidence_threshold:
                continue

            class_name = (
                CLASS_NAMES[class_id]
                if 0 <= class_id < len(CLASS_NAMES)
                else f"Unknown-{class_id}"
            )

            bbox = [int(x1), int(y1), int(x2), int(y2)]
            location = calculate_location(x1, y1, x2, y2, image_width, image_height)
            area_ratio = calculate_area_ratio(bbox, image_width, image_height)

            detections.append(
                {
                    "class": class_name,
                    "class_id": class_id,
                    "confidence": confidence,
                    "bbox": bbox,
                    "location": location,
                    "area_ratio": area_ratio,
                }
            )

        # Sort by confidence descending
        detections.sort(key=lambda d: d["confidence"], reverse=True)
        return detections

    def is_capsule_detected(self, detections: list) -> bool:
        """
        Determine whether a capsule was detected at all.

        A "Good" detection means the capsule was found but no defect.
        No detections at all means we cannot confirm a capsule is present.
        """
        return len(detections) > 0

    def draw_annotations(
        self,
        image: Image.Image,
        detections: list,
        draw_labels: bool = True,
    ) -> np.ndarray:
        """
        Draw bounding boxes and labels on an image.

        Returns an annotated numpy array (RGB).
        """
        img_np = np.array(image)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        for det in detections:
            class_id = det["class_id"]
            x1, y1, x2, y2 = det["bbox"]
            confidence = det["confidence"]
            class_name = det["class"]

            color_bgr = CLASS_COLORS_BGR.get(class_id, (255, 255, 255))
            thickness = 2

            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color_bgr, thickness)

            if draw_labels:
                label = f"{class_name} {confidence:.1%}"
                font_scale = 0.5
                (text_w, text_h), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
                )

                # Draw filled label background
                label_y1 = max(y1 - text_h - baseline - 4, 0)
                cv2.rectangle(
                    img_bgr,
                    (x1, label_y1),
                    (x1 + text_w + 6, y1),
                    color_bgr,
                    -1,
                )

                # Draw text
                cv2.putText(
                    img_bgr,
                    label,
                    (x1 + 3, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

        # Convert back to RGB for PIL/Streamlit
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# Location Calculation
# ---------------------------------------------------------------------------

def calculate_location(
    x1: float, y1: float, x2: float, y2: float,
    image_width: int, image_height: int,
) -> str:
    """
    Calculate the region of the image where the detection is centered.

    Returns a deterministic region name like "Upper-middle region".
    """
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    rx = cx / image_width
    ry = cy / image_height

    if rx < 1 / 3:
        col = "Left"
    elif rx > 2 / 3:
        col = "Right"
    else:
        col = "Middle"

    if ry < 1 / 3:
        row = "Upper"
    elif ry > 2 / 3:
        row = "Lower"
    else:
        row = "Middle"

    if row == "Middle" and col == "Middle":
        region = "Center"
    else:
        region = f"{row}-{col}"

    return f"{region} region"


# ---------------------------------------------------------------------------
# Area Ratio Calculation
# ---------------------------------------------------------------------------

def calculate_area_ratio(bbox: list, image_width: int, image_height: int) -> float:
    """
    Calculate the ratio of the bounding-box area to the image area.

    Returns a float in [0, 1].
    """
    x1, y1, x2, y2 = [float(v) for v in bbox]
    box_area = max(x2 - x1, 0) * max(y2 - y1, 0)
    image_area = max(image_width, 1) * max(image_height, 1)
    return box_area / image_area


# ---------------------------------------------------------------------------
# Convenience Functions
# ---------------------------------------------------------------------------

def load_model_cached():
    """
    Load the YOLO model (used with Streamlit's st.cache_resource).

    Returns an ultralytics YOLO instance.
    """
    from ultralytics import YOLO
    return YOLO(str(ensure_model_available(MODEL_PATH)))


def run_inspection(image: Image.Image) -> dict:
    """
    Run a full detection pass and return a structured result dict.

    This is a convenience function that wraps detector usage for testing.
    """
    detector = YOLODetector()
    detections = detector.detect(image)
    annotated = detector.draw_annotations(image, detections)

    capsule_detected = detector.is_capsule_detected(detections)

    return {
        "capsule_detected": capsule_detected,
        "detections": detections,
        "annotated_image": annotated,
    }