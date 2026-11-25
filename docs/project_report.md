# Project Report

## 1. Title Page

- **Project Title:** Pneumonia Detection from Chest X-Ray Images Using CNN and Transfer Learning
- **Student Name:** ______________________
- **Roll Number:** ______________________
- **College / Institution:** ______________________
- **Department:** ______________________
- **Guide / Supervisor:** ______________________
- **Academic Year:** ______________________

## 2. Abstract

This project investigates binary classification of chest X-ray images into
**NORMAL** and **PNEUMONIA** categories using deep learning. Two models are
developed and compared: a custom Convolutional Neural Network (CNN) built with
convolutional layers, batch normalization, max pooling, dropout and sigmoid
output; and a MobileNetV2 transfer learning model pretrained on ImageNet with a
new binary classification head. The models are trained on the public Kaggle
Chest X-Ray Images (Pneumonia) dataset and evaluated on a held-out test split.
A Flask web application allows users to upload chest X-ray images and receive
predictions.

## 3. Introduction

Pneumonia is a lung infection that can be detected with chest X-rays. Deep
learning can help automate parts of this analysis. This project implements a
custom CNN and a transfer learning model for educational purposes.

## 4. Problem Statement

Build a working system that predicts pneumonia from chest X-rays and compare
custom CNN vs. transfer learning.

## 5. Objectives

- Implement preprocessing for chest X-rays (224x224, normalization, augmentation).
- Implement custom CNN and MobileNetV2 models.
- Train and evaluate both models.
- Build a Flask prediction interface.

## 6. Existing System and Proposed System

- Existing: manual / radiologist-based screening.
- Proposed: automated CNN / MobileNetV2 classifier with web UI.

## 7. System Requirements

- Python 3.9+, TensorFlow 2, Flask, Pillow, NumPy, Scikit-learn.

## 8. Dataset Description

- Source: Kaggle Chest X-Ray Images (Pneumonia)
- Classes: NORMAL, PNEUMONIA.
- Splits: train/val/test.

## 9. Methodology

Images are loaded, resized to 224x224, normalized, augmented for training and
evaluated for test.

## 10. CNN Architecture

Custom CNN: Conv-BN-ReLU-MaxPool blocks, global average pooling, dense layers,
sigmoid.

## 11. Transfer Learning Methodology

MobileNetV2 ImageNet pretrained, head replaced, frozen initially, fine-tune.

## 12. System Architecture

Data -> Preprocessing -> Model -> Training -> Evaluation -> Flask prediction.

## 13. Implementation Details

Implemented in `src/` modules, Flask app in `app.py`, UI in `templates/`.

## 14. Evaluation Methodology

Metrics: accuracy, precision, recall, specificity, F1, ROC AUC, confusion
matrix.

## 15. Results

Results section to be populated after running training and evaluation with real
data.

## 16. Limitations

Educational use only; small dataset; no clinical validation.

## 17. Future Enhancements

- More data, hyperparameter tuning, Grad-CAM.

## 18. Conclusion

Project demonstrates CNN and transfer learning for pneumonia detection.

## 19. References

- Kaggle dataset: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
- MobileNetV2: https://arxiv.org/abs/1704.04861
- Keras: https://keras.io
