"""
Pytest configuration and shared fixtures for hf-model-downloader tests.
"""

import os
import pytest
import tempfile
from pathlib import Path


@pytest.fixture
def mock_env_vars(monkeypatch):
    """
    Fixture that sets up minimal required environment variables.
    """
    env_vars = {
        "MODEL_NAME": "gpt2",
        "S3_BUCKET": "test-bucket",
        "S3_ENDPOINT_URL": "https://objectstore.example.com",
        "AWS_ACCESS_KEY_ID": "AKIATEST1234567890AB",
        "AWS_SECRET_ACCESS_KEY": "secretkey1234567890abcdefghijklmnopqrstuvwxyz",
        "S3_PREFIX": "models/",
        "DOWNLOAD_DIR": "/tmp",
    }

    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)

    return env_vars


@pytest.fixture
def temp_download_dir():
    """
    Fixture that creates a temporary directory for testing downloads.
    """
    with tempfile.TemporaryDirectory(prefix="test_hf_") as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_model_files(temp_download_dir):
    """
    Fixture that creates sample model files for testing uploads.
    """
    model_dir = Path(temp_download_dir)

    # Create some sample files
    (model_dir / "config.json").write_text('{"model_type": "gpt2"}')
    (model_dir / "pytorch_model.bin").write_bytes(b"fake model weights" * 100)

    # Create subdirectory with files
    tokenizer_dir = model_dir / "tokenizer"
    tokenizer_dir.mkdir()
    (tokenizer_dir / "vocab.txt").write_text("word1\nword2\nword3")
    (tokenizer_dir / "special_tokens_map.json").write_text('{"unk_token": "[UNK]"}')

    return model_dir


@pytest.fixture
def mock_hf_token():
    """
    Fixture that provides a mock HuggingFace token.
    """
    return "hf_test_token_1234567890abcdefghijklmnopqrstuvwxyz"
