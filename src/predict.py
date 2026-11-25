"""Prediction helpers: load trained models and run inference on images."""

import os

import numpy as np

from src import config
from src.preprocessing import load_and_preprocess, load_and_preprocess_file


MODEL_CACHE = {}
MODEL_CACHE_LOCK = None


def get_model_path(model_name: str) -> str:
    """Return the file path of a trained model."""
    name = (model_name or config.DEFAULT_MODEL).lower()
    if name == "cnn":
        return str(config.CNN_MODEL_PATH)
    if name == "mobilenetv2":
        return str(config.MOBILENETV2_MODEL_PATH)
    if name in ("best", "best_model"):
        return str(config.BEST_MODEL_PATH)
    raise ValueError(f"Unknown model '{model_name}'. Choose one of: {config.MODEL_CHOICES}")


def model_names() -> list:
    """Return available model names."""
    return list(config.MODEL_CHOICES)


def model_exists(model_name: str) -> bool:
    """Check whether a trained model file exists."""
    try:
        return os.path.isfile(get_model_path(model_name))
    except Exception:
        return False


def load_model(model_name: str):
    """Load a trained Keras model (cached so the Flask app does not reload it
    on every request)."""
    name = (model_name or config.DEFAULT_MODEL).lower()
    if name not in MODEL_CACHE:
        import tensorflow as tf

        path = get_model_path(name)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Trained model not found for '{name}'. Please train the model first using: python -m src.train --model {name}"
            )
        model = tf.keras.models.load_model(path)
        MODEL_CACHE[name] = model
    return MODEL_CACHE[name]


def predict_array(x, model_name: str = "cnn") -> dict:
    """Run inference on a model-ready image array.

    Returns a dictionary with the predicted class, the confidence and the
    class probabilities. ``ok`` is False and ``error`` contains a message when
    prediction could not be completed.
    """
    try:
        import tensorflow as tf

        model = load_model(model_name)
        probs = model.predict(np.asarray(x, dtype=np.float32), verbose=0)
        prob = float(np.asarray(probs).ravel()[0])
        class_index = int(1) if prob >= 0.5 else int(0)
        predicted_class = config.CLASS_NAMES[class_index]
        confidence = float(prob if class_index == config.POSITIVE_CLASS_INDEX else (1.0 - prob))
        probabilities = {
            config.CLASS_NAMES[0]: float(1.0 - prob),
            config.CLASS_NAMES[1]: float(prob),
        }
        return {
            "ok": True,
            "model": (model_name or config.DEFAULT_MODEL).lower(),
            "predicted_class": predicted_class,
            "class_index": int(class_index),
            "confidence": confidence,
            "probabilities": probabilities,
            "raw_probability": prob,
        }
    except Exception as exc:
        return {"ok": False, "error": f"Prediction failed: {exc}", "status": 500}


def predict_image_path(image_path, model_name: str = "cnn") -> dict:
    """Load a real image from disk and run model prediction on it."""
    try:
        arr, error = load_and_preprocess(image_path, model_name=model_name)
        if error:
            return {"ok": False, "error": error, "status": 400}
        if not model_exists(model_name):
            return {
                "ok": False,
                "error": f"Trained model not found for '{model_name}'. Please train it first using: python -m src.train --model {model_name}",
                "status": 503,
            }
        result = predict_array(arr, model_name=model_name)
        if not result.get("ok"):
            return result
        result["image_path"] = str(image_path)
        return result
    except Exception as exc:
        return {"ok": False, "error": f"Prediction failed: {exc}", "status": 500}


def predict_uploaded(file_storage, model_name: str = "cnn"):
    """Validate and predict an uploaded file.

    Returns ``(result, temp_path)``. The caller must delete ``temp_path`` when
    it is not None. The result dictionary carries an ``ok`` flag, an ``error``
    message (when prediction was not possible) and a suggested ``status``.
    """
    arr, error, temp_path = load_and_preprocess_file(file_storage, model_name=model_name)
    if error or arr is None:
        return {"ok": False, "error": error or "Image could not be processed.", "status": 400}, temp_path
    if not model_exists(model_name):
        return {
            "ok": False,
            "error": f"Trained model not found for '{model_name}'. Please train it first using: python -m src.train --model {model_name}",
            "status": 503,
        }, temp_path
    try:
        result = predict_array(arr, model_name=model_name)
        if result.get("ok"):
            result["image_name"] = file_storage.filename
        else:
            result["status"] = result.get("status") or 500
        return result, temp_path
    except Exception as exc:
        return {"ok": False, "error": f"Prediction failed: {exc}", "status": 500}, temp_path
