# YOLO-LLM Capsule Inspection Assistant

A **YOLO-LLM based vision inspection system** for pharmaceutical capsule quality analysis. The system performs object detection to identify defects, calculates severity using a deterministic rule engine, and generates AI-powered explanations using the Groq LLM API.

## Overview

This project implements an end-to-end **detection + localization + explanation** pipeline for capsule quality inspection. Unlike a simple image classifier, this system:

1. **Detects** defects using YOLO object detection
2. **Localizes** defects with bounding boxes
3. **Assesses** severity using an explainable rule engine
4. **Explains** findings using a guardrailed LLM
5. **Reports** results via a Streamlit UI and downloadable PDF

## Problem Statement

Pharmaceutical capsule quality control requires automated visual inspection to detect manufacturing defects. Traditional manual inspection is:
- Time-consuming
- Subjective
- Prone to human error

This system provides an automated, consistent, and explainable inspection assistant.

## Features

- ✅ **7-class defect detection** (Good, Crack, Scratch, Faulty Imprint, Poke, Squeeze, Contamination)
- ✅ **Bounding box localization** for every detected defect
- ✅ **Confidence scores** for all predictions
- ✅ **Deterministic severity engine** (independent of confidence)
- ✅ **LLM-generated explanations** with strict guardrails
- ✅ **Multiple defect support** per image
- ✅ **Good capsule handling** (no false defect boxes)
- ✅ **PDF report generation** with annotated images
- ✅ **LLM fallback** (deterministic templates when API unavailable)
- ✅ **Streamlit web interface**
- ✅ **CPU-optimized inference** for free hosting

## Architecture

```
                 ┌───────────────────────┐
                 │       Streamlit       │
                 │      Web Interface    │
                 └───────────┬───────────┘
                             │
                             ▼
                 ┌───────────────────────┐
                 │    Image Validation   │
                 └───────────┬───────────┘
                             │
                             ▼
                 ┌───────────────────────┐
                 │      YOLO Detector    │
                 │       best.pt         │
                 └───────────┬───────────┘
                             │
             ┌───────────────┼────────────────┐
             ▼               ▼                ▼
          CLASS           BBOX           CONFIDENCE
             │               │                │
             └───────────────┼────────────────┘
                             ▼
                 ┌───────────────────────┐
                 │ Location + Area       │
                 └───────────┬───────────┘
                             │
                             ▼
                 ┌───────────────────────┐
                 │   Severity Engine     │
                 └───────────┬───────────┘
                             │
                             ▼
                 ┌───────────────────────┐
                 │ Structured Detection  │
                 │       JSON            │
                 └───────────┬───────────┘
                             │
                             ▼
                 ┌───────────────────────┐
                 │       Groq LLM        │
                 │ Explanation Only      │
                 └───────────┬───────────┘
                             │
             ┌───────────────┼────────────────┐
             ▼               ▼                ▼
        Explanation       Causes        Recommendation
             │               │                │
             └───────────────┼────────────────┘
                             ▼
                 ┌───────────────────────┐
                 │  Inspection Result    │
                 └───────────┬───────────┘
                             │
                    ┌────────┴────────┐
                    ▼                 ▼
               Streamlit UI       PDF Report
```

## Defect Classes

| ID | Class          | Meaning                           |
| -- | -------------- | --------------------------------- |
| 0  | Good           | Normal capsule                    |
| 1  | Crack          | Visible crack/fracture            |
| 2  | Scratch        | Surface scratch/abrasion          |
| 3  | Faulty Imprint | Incorrect/damaged/unclear imprint |
| 4  | Poke           | Puncture/hole-like defect         |
| 5  | Squeeze        | Compression/deformation           |
| 6  | Contamination  | Foreign/undesired material        |

## Severity Levels

| Level    | Meaning                                       |
| -------- | --------------------------------------------- |
| None     | No visible defect                             |
| Low      | Minor visible defect                          |
| Moderate | Defect requiring inspection                   |
| High     | Significant quality concern                   |
| Critical | Potentially serious integrity/quality concern |

## Project Structure

```
Vision-LLM-Capsule-Project/
│
├── app.py                    # Streamlit UI + workflow orchestration
├── yolo_detection.py         # YOLO detection, bbox, confidence, location
├── severity.py               # Deterministic severity engine
├── defect_knowledge.py       # Defect knowledge base
├── llm_explanation.py        # Groq LLM integration + fallback
├── report_generator.py       # PDF report generation
├── utils.py                  # Image validation, file naming, helpers
│
├── models/
│   └── best.pt               # Trained YOLO model (place here)
│
├── notebooks/
│   ├── 01_dataset_analysis.ipynb
│   ├── 02_training.ipynb
│   └── 03_evaluation.ipynb
│
├── scripts/
│   └── validate_dataset.py   # Dataset validation script
│
├── tests/
│   ├── test_severity.py
│   ├── test_location.py
│   ├── test_detection.py
│   └── test_report.py
│
├── .streamlit/
│   └── secrets.toml          # Local secrets (never commit)
│
├── requirements.txt
├── .gitignore
└── README.md
```

## Local Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd Vision-LLM-Capsule-Project

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Environment Variables

Create a `.streamlit/secrets.toml` file:

```toml
GROQ_API_KEY = "your-groq-api-key-here"
```

Get your API key from [Groq Console](https://console.groq.com/keys).

## Model Setup

1. Train the YOLO model using `notebooks/02_training.ipynb` on Google Colab
2. Download the `best.pt` file
3. Place it in `models/best.pt`

## Running the Application

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

## Training Pipeline

1. **Dataset Analysis** - `notebooks/01_dataset_analysis.ipynb`
   - Validate dataset structure
   - Check class distribution
   - Verify annotations

2. **Training** - `notebooks/02_training.ipynb`
   - Train YOLOv8 nano on Google Colab
   - Validate on test set
   - Download `best.pt`

3. **Evaluation** - `notebooks/03_evaluation.ipynb`
   - Per-class metrics
   - Confusion matrix
   - Error analysis

## Testing

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_severity.py
pytest tests/test_location.py
pytest tests/test_detection.py
pytest tests/test_report.py
```

## Deployment

### Streamlit Community Cloud (Recommended)

1. Push the repository to GitHub
2. Go to [Streamlit Community Cloud](https://streamlit.io/cloud)
3. Create a new app
4. Select the repository and branch
5. Set `app.py` as the main file
6. Configure secrets (GROQ_API_KEY)
7. Deploy

### Hugging Face Spaces (Alternative)

1. Create a new Space with Streamlit SDK
2. Push the code
3. Set the `GROQ_API_KEY` secret
4. Deploy

## LLM Guardrails

The system prompt enforces strict rules:

- The LLM **does not** perform visual detection
- The LLM **cannot** invent defects
- The LLM **cannot** change detection facts
- Possible causes are presented as **possibilities**, not confirmed root causes
- No specific machines, operators, or batches are invented

## LLM Fallback

If the Groq API is unavailable, the system automatically uses deterministic explanation templates from the defect knowledge base. The inspection never fails due to LLM unavailability.

## Performance Metrics

The evaluation notebook tracks:

- Precision
- Recall
- F1
- mAP@50
- mAP@50-95
- Per-class metrics
- Confusion matrix
- Inference speed

## Limitations

- This is a **prototype** for quality control assistance, not a validated pharmaceutical release system
- Severity rules are implementation starting points, not medically validated thresholds
- Model performance depends on dataset quality and coverage
- CPU inference is slower than GPU

## Future Improvements

- ONNX export for faster CPU inference
- Batch image processing
- SQLite database for inspection history
- More training data for minority classes
- Active learning for edge cases
- Integration with manufacturing systems

## License

MIT License - see LICENSE file for details.