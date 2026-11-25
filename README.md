
# Pneumonia Detection from Chest X-Ray Images Using CNN and Transfer Learning

A complete, beginner-friendly deep learning project that classifies chest X-ray
images as **NORMAL** or **PNEUMONIA** using a custom Convolutional Neural Network
and a MobileNetV2 transfer learning model, with a Flask web interface.

> **Medical disclaimer:** This project is for educational and academic research
> purposes only. It is not a medical device and must not be used to make any
> clinical or treatment decisions.

## Table of Contents

1. [Project Introduction](#project-introduction)
2. [Problem Statement](#problem-statement)
3. [Objectives](#objectives)
4. [Technologies Used](#technologies-used)
5. [Dataset](#dataset)
6. [Installation](#installation)
7. [Training](#training)
8. [Evaluation](#evaluation)
9. [How to Add Your Own X-Ray Images](#how-to-add-your-own-x-ray-images)
10. [Running the Web Application](#running-the-web-application)
11. [Using the Web Interface](#using-the-web-interface)
12. [Project Limitations](#project-limitations)
13. [Troubleshooting](#troubleshooting)
14. [Academic and Medical Disclaimer](#academic-and-medical-disclaimer)

## Project Introduction

Chest X-ray images are widely used to screen for lung conditions, including
pneumonia. Machine learning models can learn patterns from these images to help
classify whether an X-ray looks normal or shows signs of pneumonia.

This project implements two deep learning approaches:

- **Model A – Custom CNN:** a small convolutional neural network built from
  scratch with convolution, batch normalization, max pooling, dropout and a
  sigmoid output.
- **Model B – MobileNetV2 Transfer Learning:** a model pretrained on ImageNet
  whose classification head is replaced with a small binary classifier trained
  on chest X-ray images.

Both models are trained on the real **Chest X-Ray Images (Pneumonia)** dataset
from Kaggle and compared.

## Problem Statement

Manual screening of chest X-rays is time-consuming and subject to
inter-reader variability. The objective of this project is to build a simple,
explainable deep learning system that automatically predicts whether a chest
X-ray belongs to the NORMAL or PNEUMONIA class, using either a custom CNN or a
transfer learning model.

## Objectives

- Load and validate real chest X-ray images from the Kaggle pneumonia dataset.
- Preprocess images consistently (resize to 224x224, normalization, augmentation).
- Build a custom CNN from scratch for binary classification.
- Build a MobileNetV2 transfer learning model for binary classification.
- Train both models and save the best checkpoints.
- Evaluate accuracy, precision, recall, specificity, F1-score and ROC AUC.
- Provide a Flask web app for uploading and predicting chest X-rays.
- Compare the two models and select the recommended one.

## Technologies Used

| Area | Technology |
| --- | --- |
| Language | Python 3 |
| Deep learning | TensorFlow 2 / Keras |
| Model 1 | Custom CNN |
| Model 2 | MobileNetV2 (ImageNet pretrained) |
| Image processing | Pillow, TensorFlow image utilities |
| Data analysis | NumPy, Pandas, Matplotlib, Seaborn |
| Evaluation | Scikit-learn |
| Backend | Flask |
| Frontend | HTML5, CSS3, JavaScript, Bootstrap 5, Chart.js |
| Charts | Chart.js / Matplotlib |

## Dataset

- **Source:** [Chest X-Ray Images (Pneumonia) on Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **Classes:** `NORMAL`, `PNEUMONIA`
- **Splits:** `train/`, `val/`, `test/` folders as provided by the dataset.

See [scripts/download_instructions.md](scripts/download_instructions.md) for how
to download the dataset manually or via the Kaggle API.

Folder layout:

```
dataset/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── test/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test_data/
    ├── README.txt
    └── .gitkeep
```

## Installation

```bash
python -m venv .venv

# Windows activation
.venv\Scripts\activate

pip install -r requirements.txt
```

## Training

Train the custom CNN:

```bash
python -m src.train --model cnn
```

Train MobileNetV2 with transfer learning:

```bash
python -m src.train --model mobilenetv2
```

Optional fine-tuning of the MobileNetV2 backbone:

```bash
python -m src.train --model mobilenetv2 --fine-tune
```

Training time depends on your hardware and dataset size. On CPU it may take a
long time; GPU acceleration is used automatically when available.

## Evaluation

Evaluate both trained models and generate comparison results:

```bash
python -m src.evaluate
```

This computes accuracy, precision, recall, specificity, F1-score, ROC AUC and
confusion matrices on the held-out test set, saves plots under `results/plots/`
and writes machine-readable metrics to `results/evaluation_metrics.json`.

## How to Add Your Own X-Ray Images

Place your own chest X-ray images (PNG/JPEG) into:

```
dataset/test_data/
```

Then use the **Predict** page in the web interface. Images in this folder are
used only for individual predictions and do not trigger retraining.

## Running the Web Application

```bash
python app.py
```

Open <http://127.0.0.1:5000>

## Using the Web Interface

- **Home:** dataset statistics, available models, last recorded metrics.
- **Predict:** upload a chest X-ray (drag & drop or file picker), choose a model,
  view prediction and probabilities.
- **Comparison:** metrics table, charts and confusion matrices for both models.
- **Dataset:** counts per split/class and preprocessing info.
- **About:** educational background and disclaimers.

## Project Limitations

- Small educational dataset; results are not clinically validated.
- Confidence is model probability, not clinical certainty.
- Intended for academic teaching only.

## Troubleshooting

- **Dataset missing:** download it and place it under `dataset/train/` etc.
- **Model missing:** run training commands first; the app will tell the user.
- **Import errors:** ensure TensorFlow/Keras versions are installed via
  `pip install -r requirements.txt`.

## Academic and Medical Disclaimer

See the disclaimer in the [About](http://127.0.0.1:5000/about) page and the
project report (`docs/project_report.md`).

---

## Full documentation

- Project report: [`docs/project_report.md`](docs/project_report.md)
- Presentation outline: [`docs/presentation_outline.md`](docs/presentation_outline.md)
