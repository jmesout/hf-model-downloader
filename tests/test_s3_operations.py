"""
Tests for S3 operations using moto for mocking AWS.
"""

import pytest
import sys
import os
from pathlib import Path

# Add parent directory to path to import cache_model
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cache_model
from moto import mock_aws
import boto3
from botocore.exceptions import ClientError


@mock_aws
class TestCreateS3Client:
    """Tests for S3 client creation."""

    def test_create_s3_client(self):
        """Test creating S3 client with credentials."""
        client = cache_model.create_s3_client(
            "https://s3.example.com",
            "AKIATEST123",
            "secretkey123"
        )

        assert client is not None
        assert client._endpoint.host == "https://s3.example.com"


@mock_aws
class TestModelExistsInS3:
    """Tests for checking model existence in S3."""

    def test_model_exists_when_files_present(self):
        """Test that model exists returns True when files are in S3."""
        # Create mock S3 bucket and upload a file
        s3_client = boto3.client("s3", region_name="us-east-1")
        s3_client.create_bucket(Bucket="test-bucket")
        s3_client.put_object(Bucket="test-bucket", Key="models/gpt2/config.json", Body=b"{}")

        # Check if model exists
        exists = cache_model.model_exists_in_s3(s3_client, "test-bucket", "models/gpt2/")

        assert exists is True

    def test_model_not_exists_when_empty(self):
        """Test that model exists returns False when no files in S3."""
        # Create mock S3 bucket but don't upload anything
        s3_client = boto3.client("s3", region_name="us-east-1")
        s3_client.create_bucket(Bucket="test-bucket")

        # Check if model exists
        exists = cache_model.model_exists_in_s3(s3_client, "test-bucket", "models/gpt2/")

        assert exists is False

    def test_model_exists_with_different_prefix(self):
        """Test that model exists only finds correct prefix."""
        # Create mock S3 bucket and upload to different prefix
        s3_client = boto3.client("s3", region_name="us-east-1")
        s3_client.create_bucket(Bucket="test-bucket")
        s3_client.put_object(Bucket="test-bucket", Key="models/other-model/config.json", Body=b"{}")

        # Check if our model exists (should be False)
        exists = cache_model.model_exists_in_s3(s3_client, "test-bucket", "models/gpt2/")

        assert exists is False

    def test_model_exists_handles_client_error(self):
        """Test that model exists properly raises ClientError on S3 failures."""
        # Create S3 client but don't create bucket - this will cause an error
        s3_client = boto3.client("s3", region_name="us-east-1")

        with pytest.raises(ClientError):
            cache_model.model_exists_in_s3(s3_client, "nonexistent-bucket", "models/gpt2/")


@mock_aws
class TestUploadDirectoryToS3:
    """Tests for uploading directories to S3."""

    def test_upload_single_file(self, temp_download_dir):
        """Test uploading a directory with a single file."""
        # Create a test file
        test_file = Path(temp_download_dir) / "test.txt"
        test_file.write_text("Hello World")

        # Create mock S3 bucket
        s3_client = boto3.client("s3", region_name="us-east-1")
        s3_client.create_bucket(Bucket="test-bucket")

        # Upload directory
        cache_model.upload_directory_to_s3(
            s3_client,
            temp_download_dir,
            "test-bucket",
            "models/test/"
        )

        # Verify file was uploaded
        response = s3_client.get_object(Bucket="test-bucket", Key="models/test/test.txt")
        assert response["Body"].read() == b"Hello World"

    def test_upload_multiple_files(self, sample_model_files):
        """Test uploading a directory with multiple files and subdirectories."""
        # Create mock S3 bucket
        s3_client = boto3.client("s3", region_name="us-east-1")
        s3_client.create_bucket(Bucket="test-bucket")

        # Upload directory
        cache_model.upload_directory_to_s3(
            s3_client,
            str(sample_model_files),
            "test-bucket",
            "models/gpt2"
        )

        # Verify files were uploaded with correct structure
        response = s3_client.list_objects_v2(Bucket="test-bucket", Prefix="models/gpt2/")
        keys = [obj["Key"] for obj in response.get("Contents", [])]

        assert "models/gpt2/config.json" in keys
        assert "models/gpt2/pytorch_model.bin" in keys
        assert "models/gpt2/tokenizer/vocab.txt" in keys
        assert "models/gpt2/tokenizer/special_tokens_map.json" in keys

    def test_upload_preserves_directory_structure(self, sample_model_files):
        """Test that upload preserves the directory structure."""
        # Create mock S3 bucket
        s3_client = boto3.client("s3", region_name="us-east-1")
        s3_client.create_bucket(Bucket="test-bucket")

        # Upload directory
        cache_model.upload_directory_to_s3(
            s3_client,
            str(sample_model_files),
            "test-bucket",
            "models/gpt2/"
        )

        # Verify content of nested file
        response = s3_client.get_object(
            Bucket="test-bucket",
            Key="models/gpt2/tokenizer/vocab.txt"
        )
        content = response["Body"].read().decode()
        assert "word1" in content

    def test_upload_with_prefix_without_trailing_slash(self, temp_download_dir):
        """Test that upload adds trailing slash to prefix if missing."""
        # Create a test file
        test_file = Path(temp_download_dir) / "test.txt"
        test_file.write_text("content")

        # Create mock S3 bucket
        s3_client = boto3.client("s3", region_name="us-east-1")
        s3_client.create_bucket(Bucket="test-bucket")

        # Upload with prefix WITHOUT trailing slash
        cache_model.upload_directory_to_s3(
            s3_client,
            temp_download_dir,
            "test-bucket",
            "models/test"  # No trailing slash
        )

        # Verify file was uploaded with correct key
        response = s3_client.get_object(Bucket="test-bucket", Key="models/test/test.txt")
        assert response["Body"].read() == b"content"


@mock_aws
class TestCleanupLocalFiles:
    """Tests for cleanup function."""

    def test_cleanup_existing_directory(self, temp_download_dir):
        """Test cleanup removes existing directory."""
        # Create a file in temp directory
        test_file = Path(temp_download_dir) / "test.txt"
        test_file.write_text("content")

        assert os.path.exists(temp_download_dir)
        assert os.path.exists(test_file)

        # Cleanup
        cache_model.cleanup_local_files(temp_download_dir)

        # Verify directory was removed
        assert not os.path.exists(temp_download_dir)

    def test_cleanup_nonexistent_directory(self, caplog):
        """Test cleanup handles nonexistent directory gracefully."""
        import logging
        caplog.set_level(logging.INFO)

        # This should not raise an error
        cache_model.cleanup_local_files("/nonexistent/path/to/cleanup")

        # Check that appropriate log message was generated
        assert "does not exist" in caplog.text or "skipping cleanup" in caplog.text.lower()

    def test_cleanup_with_nested_directories(self, temp_download_dir):
        """Test cleanup removes nested directory structure."""
        # Create nested directories
        nested_dir = Path(temp_download_dir) / "subdir" / "deep"
        nested_dir.mkdir(parents=True)
        (nested_dir / "file.txt").write_text("content")

        assert os.path.exists(temp_download_dir)

        # Cleanup
        cache_model.cleanup_local_files(temp_download_dir)

        # Verify entire tree was removed
        assert not os.path.exists(temp_download_dir)
