"""Tests for the image preprocessing pipeline."""

import numpy as np
import pytest
from PIL import Image

from src.preprocessing import (
    image_to_array,
    is_allowed_extension,
    load_and_preprocess,
    load_image,
    preprocess_for_cnn,
    validate_image_file,
)


@pytest.fixture
def valid_image(tmp_path):
    path = tmp_path / "valid_xray.png"
    img = Image.fromarray(np.full((64, 64, 3), 120, dtype=np.uint8))
    img.save(path, format="PNG")
    return path


@pytest.fixture
def invalid_image(tmp_path):
    path = tmp_path / "bad.xray"
    path.write_bytes(b"this is definitely not an image file")
    return path


def test_is_allowed_extension_valid():
    assert is_allowed_extension("xray.jpg") is True
    assert is_allowed_extension("xray.jpeg") is True
    assert is_allowed_extension("xray.png") is True
    assert is_allowed_extension("xray.bmp") is False
    assert is_allowed_extension("xray") is False


def test_validate_image_file_valid(valid_image):
    ok, message = validate_image_file(valid_image)
    assert ok is True
    assert message == "ok"


def test_validate_image_file_invalid(invalid_image):
    ok, message = validate_image_file(invalid_image)
    assert ok is False
    assert isinstance(message, str) and len(message) > 0


def test_load_and_preprocess_valid_image(valid_image):
    arr, error = load_and_preprocess(valid_image, model_name="cnn")
    assert error is None
    assert arr is not None
    assert arr.shape == (224, 224, 3)
    assert arr.dtype == np.float32
    assert float(arr.min()) >= 0.0
    assert float(arr.max()) <= 1.0


def test_load_and_preprocess_invalid_image(invalid_image):
    arr, error = load_and_preprocess(invalid_image, model_name="cnn")
    assert arr is None
    assert error is not None
    assert len(error) > 0


def test_image_to_array_range(valid_image):
    img = load_image(valid_image)
    arr = image_to_array(img)
    assert arr.shape == (224, 224, 3)
    assert arr.dtype == np.float32
    assert arr.min() >= 0.0
    assert arr.max() <= 1.0


def test_preprocess_for_cnn_does_not_change_range(valid_image):
    arr, error = load_and_preprocess(valid_image, model_name="cnn")
    assert error is None
    out = preprocess_for_cnn(arr)
    assert out.shape == (224, 224, 3)
    assert float(out.min()) >= 0.0
    assert float(out.max()) <= 1.0
