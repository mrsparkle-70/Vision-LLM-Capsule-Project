"""
Capsule Vision Inspection Assistant - Streamlit Application.

Main entry point for the web interface.

Responsibilities:
    - UI
    - Workflow orchestration
    - State management
    - Display
"""

import io
import os
import sys
import time
from pathlib import Path

import streamlit as st
from PIL import Image

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from utils import (
    CLASS_NAMES,
    generate_inspection_id,
    get_timestamp_string,
    load_validated_image,
    format_confidence,
)
from yolo_detection import YOLODetector, load_model_cached
from severity import (
    calculate_severities,
    overall_severity,
    inspection_status,
    severity_description,
    SEVERITY_NONE,
)
from llm_explanation import generate_explanation, parse_explanation
from report_generator import generate_pdf_report
from defect_knowledge import get_defect_knowledge


# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Capsule Vision Inspection Assistant",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Model Loading (cached)
# ---------------------------------------------------------------------------

@st.cache_resource
def get_detector():
    """Load the YOLO detector once and cache it."""
    return YOLODetector()


# ---------------------------------------------------------------------------
# Inspection Pipeline
# ---------------------------------------------------------------------------

def run_inspection(image: Image.Image) -> dict:
    """
    Run the full inspection pipeline on an image.

    Returns a structured result dict.
    """
    detector = get_detector()

    # Step 1: YOLO detection
    detections = detector.detect(image)

    # Step 2: Determine capsule presence
    capsule_detected = detector.is_capsule_detected(detections)

    if not capsule_detected:
        return {
            "capsule_detected": False,
            "prediction": "Capsule not detected",
            "confidence": 0.0,
            "bbox": None,
            "location": "N/A",
            "severity": SEVERITY_NONE,
            "status": "INVALID",
            "detections": [],
            "annotated_image": image,
            "description": (
                "No capsule was detected in the image. "
                "Please upload a clear image of a capsule."
            ),
            "possible_causes": [],
            "impact": "Unable to perform inspection.",
            "recommended_action": "Upload a valid capsule image and retry.",
        }

    # Step 3: Separate Good from defects
    defect_detections = [d for d in detections if d["class"] != "Good"]
    good_detections = [d for d in detections if d["class"] == "Good"]

    if not defect_detections:
        # Good capsule - no defects
        good_conf = good_detections[0]["confidence"] if good_detections else 0.0
        return {
            "capsule_detected": True,
            "prediction": "Good",
            "confidence": good_conf,
            "bbox": None,
            "location": "N/A",
            "severity": SEVERITY_NONE,
            "status": "PASS",
            "detections": [],
            "annotated_image": detector.draw_annotations(image, detections),
            "description": (
                "The capsule passed visual inspection with no detected defect."
            ),
            "possible_causes": [],
            "impact": "No quality impact. The capsule meets visual inspection criteria.",
            "recommended_action": "No action required. The capsule passes visual inspection.",
        }

    # Step 4: Calculate severity for each defect
    defect_detections = calculate_severities(defect_detections)

    # Step 5: Determine overall severity and status
    sev = overall_severity(defect_detections)
    status = inspection_status(sev)

    # Step 6: Generate LLM explanation (reads GROQ_API_KEY/GROQ_MODEL
    # from Streamlit secrets or the GROQ_API_KEY environment variable)
    try:
        groq_api_key = st.secrets.get("GROQ_API_KEY", "")
        groq_model = st.secrets.get("GROQ_MODEL", None)
        explanation_text = generate_explanation(
            defect_detections,
            api_key=groq_api_key or None,
            model=groq_model,
        )
        explanation = parse_explanation(explanation_text)
    except Exception:
        # Fallback to deterministic knowledge
        knowledge = get_defect_knowledge(defect_detections[0]["class"])
        explanation = {
            "description": knowledge["description"],
            "possible_causes": knowledge["possible_causes"],
            "impact": knowledge["impact"],
            "recommended_action": knowledge["recommended_action"],
        }

    # Step 7: Build final result
    primary = defect_detections[0]  # Highest confidence defect

    return {
        "capsule_detected": True,
        "prediction": primary["class"],
        "confidence": primary["confidence"],
        "bbox": primary["bbox"],
        "location": primary["location"],
        "severity": sev,
        "status": status,
        "detections": defect_detections,
        "annotated_image": detector.draw_annotations(image, detections),
        "description": explanation.get("description", ""),
        "possible_causes": explanation.get("possible_causes", []),
        "impact": explanation.get("impact", ""),
        "recommended_action": explanation.get("recommended_action", ""),
    }


# ---------------------------------------------------------------------------
# UI Components
# ---------------------------------------------------------------------------

def render_header():
    """Render the application header."""
    st.title("💊 Capsule Vision Inspection Assistant")
    st.markdown(
        """
        **YOLO-LLM based vision inspection for pharmaceutical capsule quality analysis.**
        Upload a capsule image to detect defects, assess severity, and generate
        an AI-powered quality report.
        """
    )
    st.divider()


def render_sidebar():
    """Render the sidebar with app information."""
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown(
            """
            This system uses:
            - **YOLO** for object detection
            - **Severity Engine** for defect assessment
            - **Groq LLM** for explanations
            - **Streamlit** for the interface
            """
        )

        st.header("📋 Defect Classes")
        for i, name in enumerate(CLASS_NAMES):
            st.markdown(f"**{i}** → {name}")

        st.header("🔒 Privacy")
        st.markdown(
            "Images are processed in memory and not stored permanently."
        )


def render_upload_section():
    """Render the image upload section."""
    st.subheader("📤 Upload Capsule Image")

    col1, col2 = st.columns([3, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "Choose a capsule image",
            type=["jpg", "jpeg", "png", "webp"],
            help="Supported formats: JPG, JPEG, PNG, WEBP",
        )

    with col2:
        st.markdown("")
        st.markdown("")
        run_button = st.button(
            "🔍 Run Inspection",
            type="primary",
            width="stretch",
            disabled=uploaded_file is None,
        )

    return uploaded_file, run_button


def render_results(result: dict, image: Image.Image, image_name: str):
    """Render the inspection results."""
    if not result:
        return

    # Inspection ID
    inspection_id = generate_inspection_id()
    timestamp = get_timestamp_string()

    st.divider()
    st.subheader("📊 Inspection Results")

    # Status badge
    status = result.get("status", "UNKNOWN")
    status_colors = {
        "PASS": "🟢",
        "REVIEW": "🟡",
        "REJECT": "🔴",
        "INVALID": "⚪",
    }
    status_icon = status_colors.get(status, "⚪")

    st.markdown(f"### {status_icon} Status: {status}")

    # Two-column layout: original vs annotated
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Original Image**")
        st.image(image, width="stretch")

    with col2:
        st.markdown("**Detection Result**")
        annotated = result.get("annotated_image")
        if annotated is not None:
            st.image(annotated, width="stretch")
        else:
            st.image(image, width="stretch")

    # Detection summary
    st.divider()
    st.markdown("### 🔍 Detection Summary")

    summary_cols = st.columns(4)

    with summary_cols[0]:
        st.metric("Prediction", result.get("prediction", "N/A"))

    with summary_cols[1]:
        conf = result.get("confidence", 0.0)
        st.metric("Confidence", format_confidence(conf) if isinstance(conf, float) else str(conf))

    with summary_cols[2]:
        st.metric("Severity", result.get("severity", "N/A"))

    with summary_cols[3]:
        st.metric("Location", result.get("location", "N/A"))

    # Bounding box
    bbox = result.get("bbox")
    if bbox:
        st.markdown(
            f"**Bounding Box:** `[{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]`"
        )

    # Multiple detections
    detections = result.get("detections", [])
    if len(detections) > 1:
        st.markdown("### 📦 All Detections")
        det_data = []
        for det in detections:
            det_data.append(
                {
                    "Class": det["class"],
                    "Confidence": format_confidence(det["confidence"]),
                    "Location": det["location"],
                    "Severity": det["severity"],
                    "Area Ratio": f"{det['area_ratio']:.3f}",
                }
            )
        st.dataframe(det_data, width="stretch")

    # AI Explanation
    st.divider()
    st.markdown("### 🤖 AI Quality Analysis")

    description = result.get("description", "")
    if description:
        st.markdown("**Description**")
        st.write(description)

    possible_causes = result.get("possible_causes", [])
    if possible_causes:
        st.markdown("**Possible Causes**")
        for cause in possible_causes:
            st.markdown(f"- {cause}")

    impact = result.get("impact", "")
    if impact:
        st.markdown("**Potential Impact**")
        st.write(impact)

    recommended_action = result.get("recommended_action", "")
    if recommended_action:
        st.markdown("**Recommended Action**")
        st.write(recommended_action)

    # Report download
    st.divider()
    st.markdown("### 📄 Report")

    # Build inspection data for report
    inspection_data = {
        "inspection_id": inspection_id,
        "timestamp": timestamp,
        "image_name": image_name,
        "prediction": result.get("prediction", "N/A"),
        "confidence": result.get("confidence", 0.0),
        "bbox": result.get("bbox"),
        "location": result.get("location", "N/A"),
        "severity": result.get("severity", "N/A"),
        "status": result.get("status", "N/A"),
        "description": result.get("description", ""),
        "possible_causes": result.get("possible_causes", []),
        "impact": result.get("impact", ""),
        "recommended_action": result.get("recommended_action", ""),
    }

    # Generate PDF
    try:
        annotated_img = result.get("annotated_image")
        if annotated_img is not None:
            annotated_pil = Image.fromarray(annotated_img)
        else:
            annotated_pil = image

        pdf_bytes = generate_pdf_report(
            inspection_data=inspection_data,
            annotated_image=annotated_pil,
        )

        st.download_button(
            label="📥 Download PDF Report",
            data=pdf_bytes,
            file_name=f"{inspection_id}.pdf",
            mime="application/pdf",
            width="stretch",
        )
    except Exception as exc:
        st.warning(f"PDF generation failed: {exc}")


# ---------------------------------------------------------------------------
# Main Application
# ---------------------------------------------------------------------------

def main():
    """Main application entry point."""
    render_header()
    render_sidebar()

    uploaded_file, run_button = render_upload_section()

    if uploaded_file is not None:
        try:
            # Read and validate image
            image_bytes = uploaded_file.getvalue()
            image = Image.open(io.BytesIO(image_bytes))
            if image.mode != "RGB":
                image = image.convert("RGB")

            # Cap max dimension for CPU performance
            max_dim = 1280
            if max(image.size) > max_dim:
                scale = max_dim / max(image.size)
                new_size = (int(image.width * scale), int(image.height * scale))
                image = image.resize(new_size, Image.LANCZOS)

            if run_button:
                with st.spinner("Analyzing capsule..."):
                    # Run inspection
                    result = run_inspection(image)
                    render_results(result, image, uploaded_file.name)

        except Exception as exc:
            st.error(f"Error processing image: {exc}")
            st.info("Please upload a valid capsule image and try again.")


if __name__ == "__main__":
    main()