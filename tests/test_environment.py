"""
Tests for environment validation and input validation functions.
"""

import pytest
import sys
import os

# Add parent directory to path to import cache_model
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cache_model


class TestMaskCredential:
    """Tests for credential masking function."""

    def test_mask_normal_credential(self):
        """Test masking a normal length credential."""
        result = cache_model.mask_credential("AKIATEST1234567890AB")
        assert result == "AKIA***90AB"

    def test_mask_long_credential(self):
        """Test masking a longer credential."""
        long_cred = "secretkey1234567890abcdefghijklmnopqrstuvwxyz"
        result = cache_model.mask_credential(long_cred)
        assert result == "secr***wxyz"
        assert "1234567890" not in result

    def test_mask_short_credential(self):
        """Test masking credentials that are too short."""
        assert cache_model.mask_credential("short") == "***"
        assert cache_model.mask_credential("abc") == "***"
        assert cache_model.mask_credential("") == "***"

    def test_mask_none_credential(self):
        """Test masking None value."""
        assert cache_model.mask_credential(None) == "***"


class TestValidateInputs:
    """Tests for input validation function."""

    def test_valid_model_name_with_org(self):
        """Test valid model name with organization."""
        # Should not raise
        cache_model.validate_inputs("openai/gpt-3.5-turbo", "test-bucket", "models/")

    def test_valid_model_name_without_org(self):
        """Test valid model name without organization."""
        # Should not raise
        cache_model.validate_inputs("gpt2", "test-bucket", "models/")

    def test_valid_model_name_with_dots(self):
        """Test valid model name with dots and hyphens."""
        # Should not raise
        cache_model.validate_inputs("meta-llama/Llama-2-7b-hf", "test-bucket", "models/")

    def test_invalid_model_name_path_traversal(self):
        """Test model name with path traversal attempt."""
        with pytest.raises(ValueError, match="Invalid MODEL_NAME"):
            cache_model.validate_inputs("../../../etc/passwd", "test-bucket", "models/")

    def test_invalid_model_name_special_chars(self):
        """Test model name with invalid special characters."""
        with pytest.raises(ValueError, match="Invalid MODEL_NAME"):
            cache_model.validate_inputs("model@name", "test-bucket", "models/")

    def test_invalid_model_name_multiple_slashes(self):
        """Test model name with multiple slashes."""
        with pytest.raises(ValueError, match="Invalid MODEL_NAME"):
            cache_model.validate_inputs("org/model/extra", "test-bucket", "models/")

    def test_valid_s3_bucket(self):
        """Test valid S3 bucket names."""
        # Should not raise
        cache_model.validate_inputs("gpt2", "my-bucket", "models/")
        cache_model.validate_inputs("gpt2", "test.bucket.name", "models/")
        cache_model.validate_inputs("gpt2", "bucket123", "models/")

    def test_invalid_s3_bucket_uppercase(self):
        """Test S3 bucket with uppercase (invalid)."""
        with pytest.raises(ValueError, match="Invalid S3_BUCKET"):
            cache_model.validate_inputs("gpt2", "MyBucket", "models/")

    def test_invalid_s3_bucket_too_short(self):
        """Test S3 bucket that's too short."""
        with pytest.raises(ValueError, match="Invalid S3_BUCKET"):
            cache_model.validate_inputs("gpt2", "ab", "models/")

    def test_invalid_s3_bucket_special_chars(self):
        """Test S3 bucket with invalid special characters."""
        with pytest.raises(ValueError, match="Invalid S3_BUCKET"):
            cache_model.validate_inputs("gpt2", "bucket_name", "models/")

    def test_valid_s3_prefix(self):
        """Test valid S3 prefixes."""
        # Should not raise
        cache_model.validate_inputs("gpt2", "test-bucket", "models/")
        cache_model.validate_inputs("gpt2", "test-bucket", "my-models/v1/")
        cache_model.validate_inputs("gpt2", "test-bucket", "")

    def test_invalid_s3_prefix_path_traversal(self):
        """Test S3 prefix with path traversal."""
        with pytest.raises(ValueError, match="path traversal"):
            cache_model.validate_inputs("gpt2", "test-bucket", "../models/")

        with pytest.raises(ValueError, match="path traversal"):
            cache_model.validate_inputs("gpt2", "test-bucket", "models/../other/")

    def test_invalid_s3_prefix_absolute_path(self):
        """Test S3 prefix with absolute path."""
        with pytest.raises(ValueError, match="absolute path"):
            cache_model.validate_inputs("gpt2", "test-bucket", "/models/")


class TestValidateEnvironment:
    """Tests for environment validation function."""

    def test_validate_environment_all_required_vars(self, mock_env_vars):
        """Test validation with all required environment variables."""
        config = cache_model.validate_environment()

        assert config["model_name"] == "gpt2"
        assert config["s3_bucket"] == "test-bucket"
        assert config["s3_endpoint_url"] == "https://objectstore.example.com"
        assert config["aws_access_key_id"] == "AKIATEST1234567890AB"
        assert config["aws_secret_access_key"] == "secretkey1234567890abcdefghijklmnopqrstuvwxyz"
        assert config["s3_prefix"] == "models/"
        assert config["download_dir"] == "/tmp"

    def test_validate_environment_missing_model_name(self, mock_env_vars, monkeypatch):
        """Test validation fails when MODEL_NAME is missing."""
        monkeypatch.delenv("MODEL_NAME")

        with pytest.raises(SystemExit) as exc_info:
            cache_model.validate_environment()
        assert exc_info.value.code == 1

    def test_validate_environment_missing_s3_bucket(self, mock_env_vars, monkeypatch):
        """Test validation fails when S3_BUCKET is missing."""
        monkeypatch.delenv("S3_BUCKET")

        with pytest.raises(SystemExit) as exc_info:
            cache_model.validate_environment()
        assert exc_info.value.code == 1

    def test_validate_environment_missing_aws_credentials(self, mock_env_vars, monkeypatch):
        """Test validation fails when AWS credentials are missing."""
        monkeypatch.delenv("AWS_ACCESS_KEY_ID")

        with pytest.raises(SystemExit) as exc_info:
            cache_model.validate_environment()
        assert exc_info.value.code == 1

    def test_validate_environment_with_hf_token(self, mock_env_vars, monkeypatch):
        """Test validation with optional HF_TOKEN."""
        monkeypatch.setenv("HF_TOKEN", "hf_test_token")

        config = cache_model.validate_environment()
        assert config["hf_token"] == "hf_test_token"

    def test_validate_environment_without_hf_token(self, mock_env_vars):
        """Test validation without optional HF_TOKEN."""
        config = cache_model.validate_environment()
        assert config["hf_token"] is None

    def test_validate_environment_invalid_model_name(self, mock_env_vars, monkeypatch):
        """Test validation fails with invalid model name."""
        monkeypatch.setenv("MODEL_NAME", "../../../etc/passwd")

        with pytest.raises(SystemExit) as exc_info:
            cache_model.validate_environment()
        assert exc_info.value.code == 1

    def test_validate_environment_custom_download_dir(self, mock_env_vars, monkeypatch):
        """Test validation with custom DOWNLOAD_DIR."""
        monkeypatch.setenv("DOWNLOAD_DIR", "/custom/path")

        config = cache_model.validate_environment()
        assert config["download_dir"] == "/custom/path"
