# Chest X-Ray Pneumonia Detection - Execution & Project Guide

## 1. GitHub Repository Details

- **Repository Name:** `Chest-XRay-Pneumonia-Detection`
- **GitHub URL:** [https://github.com/Preet-vora30/Chest-XRay-Pneumonia-Detection](https://github.com/Preet-vora30/Chest-XRay-Pneumonia-Detection)
- **Branch:** `main`
- **Owner:** `Preet-vora30`

---

## 2. Project Verification & Status

- **Environment:** Configured in `.venv/` (Python 3.14.3 with TensorFlow 2.22, Flask 3.1, Scikit-Learn 1.9, Pillow, Matplotlib, Seaborn, NumPy, Pandas).
- **Test Suite:** Automated tests run via `pytest` (**21 passed, 1 skipped**).
- **Flask Web Interface:** All endpoints (`/`, `/health`, `/dataset`, `/compare`, `/about`, `/predict`) verified returning HTTP `200 OK`.

---

## 3. How to Run the Project

### A. Run the Web Application (Recommended)
You can start the Flask web interface by executing:
```powershell
.\.venv\Scripts\python.exe app.py
```
Or double-click `run.bat` in the project root.

Once started, open your browser and navigate to:
```
http://127.0.0.1:5000
```

### B. Run the Test Suite
To verify the complete test suite:
```powershell
.\.venv\Scripts\python.exe -m pytest tests/
```

### C. Train Deep Learning Models
Make sure you have downloaded the Kaggle pneumonia dataset into `dataset/train/`, `dataset/val/`, and `dataset/test/` (see `scripts/download_instructions.md`).

- **Train Custom CNN:**
  ```powershell
  .\.venv\Scripts\python.exe -m src.train --model cnn
  ```

- **Train MobileNetV2 Transfer Learning:**
  ```powershell
  .\.venv\Scripts\python.exe -m src.train --model mobilenetv2
  ```

### D. Evaluate Models & Generate Metrics
```powershell
.\.venv\Scripts\python.exe -m src.evaluate
```
This produces confusion matrices, ROC curves, and classification metrics under `results/`.

---

## 4. Project Directory Structure

```text
NNDL_Mini/
├── app.py                      # Flask web application entrypoint
├── requirements.txt            # Python dependencies
├── README.md                   # Comprehensive project documentation
├── RUN_GUIDE.md                # Quick run and reference guide
├── run.bat                     # Windows 1-click launcher
├── dataset/                    # Dataset directory
│   └── test_data/              # Sample images for quick testing
├── docs/                       # Project report and presentation outline
├── models/                     # Saved Keras model checkpoints (.keras)
├── results/                    # Generated charts, confusion matrices, metrics
├── scripts/                    # Convenience training & evaluation scripts
├── src/                        # Core deep learning & pipeline source code
│   ├── cnn_model.py            # Custom CNN architecture
│   ├── transfer_learning.py    # MobileNetV2 architecture
│   ├── data_loader.py          # Dataset ingestion & validation
│   ├── preprocessing.py        # Image augmentation & resizing
│   ├── train.py                # Training loop with callbacks
│   ├── evaluate.py             # Performance evaluation & plotting
│   └── predict.py              # Inference helpers
├── static/                     # CSS, JavaScript, and static assets
├── templates/                  # Jinja2 HTML templates for Flask UI
└── tests/                      # Pytest automated test suite
```
