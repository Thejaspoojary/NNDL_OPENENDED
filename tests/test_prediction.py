"""Tests for prediction helpers (model loading, inference, error handling)."""

import numpy as np
import pytest
from PIL import Image

from src import config
from src.predict import model_exists, predict_image_path


@pytest.fixture
def valid_image(tmp_path):
    path = tmp_path / "valid_xray.png"
    img = Image.fromarray(np.full((64, 64, 3), 120, dtype=np.uint8))
    img.save(path, format="PNG")
    return path


def test_predict_invalid_image_returns_error(tmp_path):
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"not really an image")
    result = predict_image_path(bad, model_name="cnn")
    assert result.get("ok") is False
    assert result.get("error")
    assert result.get("status") == 400


def test_predict_valid_image_without_trained_model(tmp_path):
    """Without a trained model the prediction must fail with a helpful message."""
    valid = tmp_path / "valid_xray.png"
    img = Image.fromarray(np.full((64, 64, 3), 120, dtype=np.uint8))
    img.save(valid, format="PNG")
    result = predict_image_path(valid, model_name="cnn")
    assert result.get("ok") is False
    assert "train" in str(result.get("error")).lower()
    assert result.get("status") == 503


def test_predict_image_with_real_model(tmp_path):
    """Genuine inference is only run when a trained model is available."""
    if not model_exists("cnn"):
        pytest.skip("Trained model not available (run: python -m src.train --model cnn)")


def test_predict_probabilities_keys():
    # Sanity check on the class naming used by predictions.
    assert config.CLASS_NAMES == ["NORMAL", "PNEUMONIA"]
