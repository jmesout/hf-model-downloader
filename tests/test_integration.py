"""
Integration tests for the full model caching workflow.
"""

import pytest
import sys
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add parent directory to path to import cache_model
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cache_model
from moto import mock_aws
import boto3


@mock_aws
class TestMainWorkflow:
    """Integration tests for the main() function workflow."""

    @patch('cache_model.cleanup_local_files')
    def test_main_exits_early_when_model_exists(self, mock_cleanup, mock_env_vars, monkeypatch):
        """Test that main exits successfully when model already exists in S3."""
        # Create mock S3 bucket with existing model
        s3_client = boto3.client("s3", region_name="us-east-1")
        s3_client.create_bucket(Bucket="test-bucket")
        s3_client.put_object(
            Bucket="test-bucket",
            Key="models/gpt2/config.json",
            Body=b"{}"
        )

        # Patch create_s3_client to return our mock client
        with patch('cache_model.create_s3_client', return_value=s3_client):
            with pytest.raises(SystemExit) as exc_info:
                cache_model.main()

            # Should exit with code 0 (success)
            assert exc_info.value.code == 0

        # Cleanup should not be called since no download occurred
        mock_cleanup.assert_not_called()

    @patch('cache_model.snapshot_download')
    @patch('cache_model.cleanup_local_files')
    def test_main_downloads_and_uploads_new_model(
        self,
        mock_cleanup,
        mock_snapshot,
        mock_env_vars,
        sample_model_files
    ):
        """Test full workflow: download from HF and upload to S3."""
        # Mock snapshot_download to return our sample files
        mock_snapshot.return_value = str(sample_model_files)

        # Create mock S3 bucket (empty - model doesn't exist)
        s3_client = boto3.client("s3", region_name="us-east-1")
        s3_client.create_bucket(Bucket="test-bucket")

        # Patch create_s3_client and tempfile.mkdtemp
        with patch('cache_model.create_s3_client', return_value=s3_client):
            with patch('cache_model.tempfile.mkdtemp', return_value=str(sample_model_files)):
                with pytest.raises(SystemExit) as exc_info:
                    cache_model.main()

                # Should exit with code 0 (success)
                assert exc_info.value.code == 0

        # Verify model was uploaded to S3
        response = s3_client.list_objects_v2(Bucket="test-bucket", Prefix="models/gpt2/")
        keys = [obj["Key"] for obj in response.get("Contents", [])]

        assert "models/gpt2/config.json" in keys
        assert "models/gpt2/pytorch_model.bin" in keys

        # Verify cleanup was called
        mock_cleanup.assert_called_once()

    @patch('cache_model.snapshot_download')
    @patch('cache_model.cleanup_local_files')
    def test_main_cleanup_called_on_error(
        self,
        mock_cleanup,
        mock_snapshot,
        mock_env_vars,
        temp_download_dir
    ):
        """Test that cleanup is called even when upload fails."""
        # Mock snapshot_download to succeed
        mock_snapshot.return_value = temp_download_dir

        # Create mock S3 client
        s3_client = boto3.client("s3", region_name="us-east-1")
        s3_client.create_bucket(Bucket="test-bucket")

        # Mock upload to raise an error
        with patch('cache_model.create_s3_client', return_value=s3_client):
            with patch('cache_model.tempfile.mkdtemp', return_value=temp_download_dir):
                with patch('cache_model.upload_directory_to_s3', side_effect=Exception("Upload failed")):
                    with pytest.raises(SystemExit) as exc_info:
                        cache_model.main()

                    # Should exit with code 1 (failure)
                    assert exc_info.value.code == 1

        # Verify cleanup was still called despite the error
        mock_cleanup.assert_called_once()

    def test_main_fails_with_invalid_env_vars(self, monkeypatch):
        """Test that main exits with error when environment variables are invalid."""
        # Set invalid model name
        monkeypatch.setenv("MODEL_NAME", "../../../etc/passwd")
        monkeypatch.setenv("S3_BUCKET", "test-bucket")
        monkeypatch.setenv("S3_ENDPOINT_URL", "https://s3.example.com")
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIATEST123")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secretkey123")

        with pytest.raises(SystemExit) as exc_info:
            cache_model.main()

        # Should exit with code 1 (failure due to validation)
        assert exc_info.value.code == 1

    def test_main_fails_with_missing_env_vars(self):
        """Test that main exits with error when required env vars are missing."""
        # Don't set any environment variables

        with pytest.raises(SystemExit) as exc_info:
            cache_model.main()

        # Should exit with code 1 (failure)
        assert exc_info.value.code == 1


@mock_aws
class TestEndToEndScenarios:
    """End-to-end scenario tests."""

    @patch('cache_model.snapshot_download')
    @patch('cache_model.cleanup_local_files')
    def test_scenario_gated_model_with_token(
        self,
        mock_cleanup,
        mock_snapshot,
        mock_env_vars,
        mock_hf_token,
        sample_model_files,
        monkeypatch
    ):
        """Test downloading a gated model with HF token."""
        # Set up for gated model
        monkeypatch.setenv("MODEL_NAME", "meta-llama/Llama-2-7b-hf")
        monkeypatch.setenv("HF_TOKEN", mock_hf_token)

        mock_snapshot.return_value = str(sample_model_files)

        # Create empty S3 bucket
        s3_client = boto3.client("s3", region_name="us-east-1")
        s3_client.create_bucket(Bucket="test-bucket")

        with patch('cache_model.create_s3_client', return_value=s3_client):
            with patch('cache_model.tempfile.mkdtemp', return_value=str(sample_model_files)):
                with pytest.raises(SystemExit) as exc_info:
                    cache_model.main()

                assert exc_info.value.code == 0

        # Verify HF token was used
        mock_snapshot.assert_called_once()
        call_kwargs = mock_snapshot.call_args[1]
        assert call_kwargs['token'] == mock_hf_token

    @patch('cache_model.snapshot_download')
    @patch('cache_model.cleanup_local_files')
    def test_scenario_custom_s3_prefix(
        self,
        mock_cleanup,
        mock_snapshot,
        mock_env_vars,
        sample_model_files,
        monkeypatch
    ):
        """Test using custom S3 prefix."""
        # Set custom prefix
        monkeypatch.setenv("S3_PREFIX", "my-models/prod/")

        mock_snapshot.return_value = str(sample_model_files)

        # Create empty S3 bucket
        s3_client = boto3.client("s3", region_name="us-east-1")
        s3_client.create_bucket(Bucket="test-bucket")

        with patch('cache_model.create_s3_client', return_value=s3_client):
            with patch('cache_model.tempfile.mkdtemp', return_value=str(sample_model_files)):
                with pytest.raises(SystemExit) as exc_info:
                    cache_model.main()

                assert exc_info.value.code == 0

        # Verify files were uploaded to custom prefix
        response = s3_client.list_objects_v2(Bucket="test-bucket", Prefix="my-models/prod/gpt2/")
        keys = [obj["Key"] for obj in response.get("Contents", [])]

        assert len(keys) > 0
        assert any("my-models/prod/gpt2/" in key for key in keys)
