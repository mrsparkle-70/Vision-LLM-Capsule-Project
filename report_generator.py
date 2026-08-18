"""
PDF Report Generator for Capsule Vision Inspection.

Responsibilities:
    - Generate downloadable PDF inspection reports
    - Include annotated image
    - Include all inspection details
"""

import io
import os
from datetime import datetime
from pathlib import Path

from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image as RLImage,
    Table,
    TableStyle,
)

from utils import format_confidence, get_timestamp_string


# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------

def _get_styles():
    """Create and return report styles."""
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Title"],
        fontSize=20,
        spaceAfter=12,
        textColor=colors.HexColor("#1a5276"),
    )

    heading_style = ParagraphStyle(
        "CustomHeading",
        parent=styles["Heading2"],
        fontSize=14,
        spaceBefore=12,
        spaceAfter=6,
        textColor=colors.HexColor("#1a5276"),
    )

    body_style = ParagraphStyle(
        "CustomBody",
        parent=styles["BodyText"],
        fontSize=10,
        leading=14,
        spaceAfter=6,
    )

    label_style = ParagraphStyle(
        "LabelStyle",
        parent=styles["BodyText"],
        fontSize=10,
        leading=14,
        fontName="Helvetica-Bold",
        textColor=colors.HexColor("#555555"),
    )

    value_style = ParagraphStyle(
        "ValueStyle",
        parent=styles["BodyText"],
        fontSize=10,
        leading=14,
    )

    return {
        "title": title_style,
        "heading": heading_style,
        "body": body_style,
        "label": label_style,
        "value": value_style,
    }


# ---------------------------------------------------------------------------
# Report Generation
# ---------------------------------------------------------------------------

def generate_pdf_report(
    inspection_data: dict,
    annotated_image: Image.Image = None,
    output_path: str = None,
) -> bytes:
    """
    Generate a PDF inspection report.

    Args:
        inspection_data: Dict with inspection results.
        annotated_image: PIL Image with bounding boxes drawn (optional).
        output_path: Path to save the PDF. If None, returns bytes.

    Returns:
        PDF bytes if output_path is None, otherwise writes to file.
    """
    styles = _get_styles()

    # Create a buffer
    buffer = io.BytesIO()

    # Create the PDF document
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=0.75 * inch,
        leftMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
    )

    story = []

    # Title
    story.append(Paragraph("CAPSULE INSPECTION REPORT", styles["title"]))
    story.append(Spacer(1, 0.1 * inch))

    # Inspection ID and timestamp
    inspection_id = inspection_data.get("inspection_id", "N/A")
    timestamp = inspection_data.get("timestamp", get_timestamp_string())

    meta_table = Table(
        [
            [Paragraph("Inspection ID:", styles["label"]),
             Paragraph(str(inspection_id), styles["value"])],
            [Paragraph("Date/Time:", styles["label"]),
             Paragraph(str(timestamp), styles["value"])],
            [Paragraph("Image File:", styles["label"]),
             Paragraph(str(inspection_data.get("image_name", "N/A")), styles["value"])],
        ],
        colWidths=[1.5 * inch, 5.0 * inch],
    )
    meta_table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(meta_table)
    story.append(Spacer(1, 0.2 * inch))

    # Annotated image
    if annotated_image is not None:
        story.append(Paragraph("Detection Result", styles["heading"]))
        img_buffer = io.BytesIO()
        annotated_image.save(img_buffer, format="PNG")
        img_buffer.seek(0)

        # Scale image to fit page width
        max_width = 6.5 * inch
        img_width, img_height = annotated_image.size
        scale = min(1.0, max_width / img_width)
        display_width = img_width * scale
        display_height = img_height * scale

        img = RLImage(img_buffer, width=display_width, height=display_height)
        story.append(img)
        story.append(Spacer(1, 0.2 * inch))

    # Detection summary
    story.append(Paragraph("Detection Summary", styles["heading"]))

    prediction = inspection_data.get("prediction", "N/A")
    confidence = inspection_data.get("confidence", 0.0)
    location = inspection_data.get("location", "N/A")
    severity = inspection_data.get("severity", "N/A")
    status = inspection_data.get("status", "N/A")

    summary_data = [
        [Paragraph("Prediction:", styles["label"]),
         Paragraph(str(prediction), styles["value"])],
        [Paragraph("Confidence:", styles["label"]),
         Paragraph(format_confidence(confidence) if isinstance(confidence, float) else str(confidence), styles["value"])],
        [Paragraph("Location:", styles["label"]),
         Paragraph(str(location), styles["value"])],
        [Paragraph("Severity:", styles["label"]),
         Paragraph(str(severity), styles["value"])],
        [Paragraph("Status:", styles["label"]),
         Paragraph(str(status), styles["value"])],
    ]

    # Add bounding box if present
    bbox = inspection_data.get("bbox")
    if bbox:
        bbox_str = f"[{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]"
        summary_data.append(
            [Paragraph("Bounding Box:", styles["label"]),
             Paragraph(bbox_str, styles["value"])]
        )

    summary_table = Table(
        summary_data,
        colWidths=[1.5 * inch, 5.0 * inch],
    )
    summary_table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("LINEBELOW", (0, 0), (-1, -2), 0.5, colors.HexColor("#dddddd")),
            ]
        )
    )
    story.append(summary_table)
    story.append(Spacer(1, 0.2 * inch))

    # Explanation
    story.append(Paragraph("AI Quality Analysis", styles["heading"]))

    description = inspection_data.get("description", "")
    if description:
        story.append(Paragraph("Description", styles["label"]))
        story.append(Paragraph(str(description), styles["body"]))

    possible_causes = inspection_data.get("possible_causes", [])
    if possible_causes:
        story.append(Paragraph("Possible Causes", styles["label"]))
        for cause in possible_causes:
            story.append(Paragraph(f"• {cause}", styles["body"]))

    impact = inspection_data.get("impact", "")
    if impact:
        story.append(Paragraph("Potential Impact", styles["label"]))
        story.append(Paragraph(str(impact), styles["body"]))

    recommended_action = inspection_data.get("recommended_action", "")
    if recommended_action:
        story.append(Paragraph("Recommended Action", styles["label"]))
        story.append(Paragraph(str(recommended_action), styles["body"]))

    # Footer
    story.append(Spacer(1, 0.3 * inch))
    story.append(
        Paragraph(
            "This report was generated automatically by the Capsule Vision "
            "Inspection Assistant. It is a prototype recommendation and does "
            "not replace regulated QA/QC procedures.",
            ParagraphStyle(
                "Footer",
                parent=styles["body"],
                fontSize=8,
                textColor=colors.HexColor("#999999"),
            ),
        )
    )

    # Build the PDF
    doc.build(story)

    # Return bytes or write to file
    pdf_bytes = buffer.getvalue()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(pdf_bytes)
        return pdf_bytes

    return pdf_bytes


def save_report_pdf(
    inspection_data: dict,
    annotated_image: Image.Image = None,
    output_dir: str = "outputs",
) -> str:
    """
    Save a PDF report to the outputs directory.

    Returns the path to the saved file.
    """
    inspection_id = inspection_data.get("inspection_id", "report")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{inspection_id}_{timestamp}.pdf"
    output_path = os.path.join(output_dir, filename)

    generate_pdf_report(
        inspection_data=inspection_data,
        annotated_image=annotated_image,
        output_path=output_path,
    )

    return output_path