"""Tests for dataset validation helpers."""

import pytest

from src.data_loader import validate_dataset


def test_validate_dataset_missing_returns_invalid():
    result = validate_dataset(data_root="/tmp/definitely_missing_pneumonia_dataset")
    assert result.get("valid") is False
    assert isinstance(result.get("message"), str)
    assert len(result["message"]) > 0
