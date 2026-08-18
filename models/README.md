# Models Directory

Place your trained YOLO detection model here.

## Required File

- `best.pt` - The trained YOLO detection model

## Training

Train the model using the notebook in `notebooks/02_training.ipynb` on Google Colab.

After training, download the `best.pt` file from the Colab output and place it here.

## Model Requirements

- YOLO detection model (not classification)
- Trained on the 7-class capsule dataset:
  - 0: Good
  - 1: Crack
  - 2: Scratch
  - 3: Faulty Imprint
  - 4: Poke
  - 5: Squeeze
  - 6: Contamination

## Note

The model file is typically 5-10 MB for YOLO nano. If it's too large for Git,
use Git LFS or download it at deployment time.