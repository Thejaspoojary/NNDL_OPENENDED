"""Tests for the Flask application routes and upload validation."""

import io

import numpy as np
import pytest
from PIL import Image

import app as app_module


@pytest.fixture
def client():
    app_module.app.config.update(TESTING=True)
    with app_module.app.test_client() as client:
        yield client


def _png_bytes():
    img = Image.fromarray(np.full((64, 64, 3), 120, dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


def test_home_page(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"Pneumonia" in response.data


def test_predict_page(client):
    response = client.get("/predict")
    assert response.status_code == 200


def test_compare_page(client):
    response = client.get("/compare")
    assert response.status_code == 200


def test_dataset_page(client):
    response = client.get("/dataset")
    assert response.status_code == 200


def test_about_page(client):
    response = client.get("/about")
    assert response.status_code == 200


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data.get("status") == "ok"


def test_predict_without_file_returns_error(client):
    response = client.post("/api/predict", data={})
    assert response.status_code == 400
    data = response.get_json()
    assert data.get("ok") is False
    assert data.get("error")


def test_predict_invalid_file_returns_error(client):
    data = {"image": (io.BytesIO(b"this is not an image"), "bad.png")}
    response = client.post("/api/predict", data=data, content_type="multipart/form-data")
    assert response.status_code == 400
    body = response.get_json()
    assert body.get("ok") is False
    assert body.get("error")


def test_predict_valid_file_without_model_returns_helpful_error(client):
    data = {"image": (io.BytesIO(_png_bytes().getvalue()), "xray.png")}
    response = client.post("/api/predict", data=data, content_type="multipart/form-data")
    assert response.status_code == 503
    body = response.get_json()
    assert body.get("ok") is False
    assert "train" in str(body.get("error")).lower()


def test_predict_unsupported_extension_returns_error(client):
    data = {"image": (io.BytesIO(b"fake"), "xray.gif")}
    response = client.post("/api/predict", data=data, content_type="multipart/form-data")
    assert response.status_code == 400
    body = response.get_json()
    assert body.get("ok") is False
    assert body.get("error")
