"""Reusable image loading and preprocessing pipeline.

This module is the single place where chest X-ray images are converted into
model-ready tensors.  Training, validation, testing and prediction all use
the exact same functions so that no preprocessing inconsistency can creep in.

Preprocessing per model
----------------------
- Custom CNN       : RGB image resized to 224x224, float32 in [0, 1].
- MobileNetV2     : same image, but passed through the
                   ``mobilenetv2.preprocess_input`` function (scales pixels
                   into the format expected by the ImageNet pretrained model).

Only real image files are loaded; anything that Pillow cannot open is
reported as invalid instead of raising an opaque error.
"""

import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from src import config

SUPPORTED_EXTENSIONS = tuple(sorted(config.ALLOWED_EXTENSIONS))

# Patterns accepted as image files inside class folders.
IMAGE_PATTERNS = ("*.jpg", "*.jpeg", "*.png")


def is_allowed_extension(filename: str) -> bool:
    """Return True if the file extension is an allowed image extension."""
    ext = Path(filename or "").suffix.lower()
    return ext in SUPPORTED_EXTENSIONS


def is_allowed_image_file(path) -> bool:
    """Return True if the file exists, has an allowed extension and is a
    decodable image."""
    try:
        if not Path(path).exists():
            return False
        if Path(path).suffix.lower() not in SUPPORTED_EXTENSIONS:
            return False
        with Image.open(path) as img:
            img.load()
        return True
    except Exception:
        return False


def validate_image_file(path) -> tuple:
    """Validate a local image file.

    Returns ``(ok, message)``.
    """
    p = Path(path)
    if not p.exists():
        return False, "Image file not found."
    if p.suffix.lower() not in SUPPORTED_EXTENSIONS:
        return False, f"Unsupported file type '{p.suffix or 'unknown'}'. Please upload a PNG or JPEG image."
    try:
        with Image.open(p) as img:
            img.load()
            fmt = (img.format or "").upper()
    except UnidentifiedImageError:
        return False, "The file could not be read as an image. It may be corrupted or not an image at all."
    except Exception as exc:  # pragma: no cover - defensive
        return False, f"Image could not be processed: {exc}"
    if fmt not in ("JPEG", "PNG"):
        return False, f"Unsupported image format '{fmt}'. Only JPEG and PNG images are supported."
    return True, "ok"


def load_image(path) -> Image.Image:
    """Load an image from disk as an RGB image resized to the model size."""
    img = Image.open(path).convert("RGB")
    return resize_image(img, config.IMG_SIZE)


def resize_image(img: Image.Image, target_size) -> Image.Image:
    """Resize a PIL image to the target size (width, height)."""
    size = (int(target_size[0]), int(target_size[1]))
    return img.resize(size, Image.Resampling.LANCZOS)


def image_to_array(img: Image.Image) -> np.ndarray:
    """Convert a PIL RGB image to a float32 array in [0, 1], shape (H, W, 3)."""
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return np.asarray(arr, dtype=np.float32)


def preprocess_for_cnn(image_array: np.ndarray) -> np.ndarray:
    """Normalize a float32 image in [0, 1] for the custom CNN."""
    return np.asarray(image_array, dtype=np.float32)


def preprocess_for_mobilenetv2(image_array: np.ndarray):
    """Normalize a float32 image in [0, 1] for MobileNetV2 (ImageNet pretrained)."""
    from src.transfer_learning import get_mobilenetv2_preprocess_input

    arr = np.asarray(image_array, dtype=np.float32)
    return get_mobilenetv2_preprocess_input()(arr)


def preprocess_image(image_array: np.ndarray, model_name: str = "cnn"):
    """Apply the correct preprocessing for the selected model."""
    arr = np.asarray(image_array, dtype=np.float32)
    if (model_name or "cnn").lower() == "mobilenetv2":
        return preprocess_for_mobilenetv2(arr)
    return preprocess_for_cnn(arr)


def load_and_preprocess(path, model_name: str = "cnn"):
    """Full pipeline: load file -> validate -> resize -> convert -> normalize.

    Returns ``(array, error)`` where array is model-ready and error is None on
    success.
    """
    ok, message = validate_image_file(path)
    if not ok:
        return None, message
    try:
        img = load_image(path)
        arr = image_to_array(img)
        arr = preprocess_image(arr, model_name)
        return arr, None
    except Exception as exc:
        return None, f"Failed to process the image: {exc}"


def load_and_preprocess_file(file_storage, model_name: str = "cnn") -> tuple:
    """Full pipeline for an uploaded file storage object.

    Writes the upload to a temporary file, runs the pipeline and returns
    ``(array, error, temp_path)``.  The caller is responsible for deleting the
    temporary file.
    """
    import tempfile
    import os

    from werkzeug.utils import secure_filename

    filename = secure_filename(file_storage.filename or "upload")
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        return None, f"Unsupported file type '{ext or 'unknown'}'. Please upload a PNG or JPEG image.", None
    max_bytes = config.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    data = file_storage.read()
    if len(data) > max_bytes:
        return None, f"Image is too large. Maximum allowed size is {config.MAX_UPLOAD_SIZE_MB} MB.", None
    fd, tmp_path = tempfile.mkstemp(prefix="pneumonia_xray_", suffix=ext or ".png")
    tmp = Path(tmp_path)
    try:
        with open(fd, "wb") as f:
            f.write(data)
        ok, message = validate_image_file(tmp)
        if not ok:
            return None, message, tmp
        arr, err = load_and_preprocess(tmp, model_name=model_name)
        return arr, err, tmp
    except Exception as exc:
        return None, f"Failed to process the uploaded image: {exc}", tmp
