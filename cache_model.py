#!/usr/bin/env python3
"""
HuggingFace Model Cache Loader for Kubernetes Init Containers

This script downloads HuggingFace models and uploads them to S3-compatible
object storage (Ceph RGW via Civo), ensuring models are available before
pod startup.
"""

import os
import sys

# Enable hf_transfer for faster downloads BEFORE importing huggingface_hub
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import logging
import re
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import boto3
from botocore.exceptions import ClientError
from huggingface_hub import snapshot_download
from huggingface_hub.utils import HfHubHTTPError
from tqdm import tqdm


# Configure logging for Kubernetes stdout capture
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)


def mask_credential(value: str) -> str:
    """
    Mask a credential for logging, showing only first 4 and last 4 characters.

    Args:
        value: The credential string to mask

    Returns:
        str: Masked version like "AKIA***WXYZ" or "***" if too short
    """
    if not value or len(value) < 8:
        return "***"
    return f"{value[:4]}***{value[-4:]}"


def validate_inputs(model_name: str, s3_bucket: str, s3_prefix: str) -> None:
    """
    Validate user inputs to prevent path traversal and injection attacks.

    Args:
        model_name: HuggingFace model repo ID
        s3_bucket: S3 bucket name
        s3_prefix: S3 prefix path

    Raises:
        ValueError: If any input fails validation
    """
    # Validate MODEL_NAME - must match HuggingFace repo pattern
    # Format: "username/repo-name" or just "repo-name"
    # Allowed chars: alphanumeric, dots, hyphens, underscores
    model_pattern = r'^[a-zA-Z0-9][a-zA-Z0-9._-]*(/[a-zA-Z0-9][a-zA-Z0-9._-]*)?$'
    if not re.match(model_pattern, model_name):
        raise ValueError(
            f"Invalid MODEL_NAME '{model_name}'. Must match pattern: "
            "alphanumeric start, followed by alphanumeric/dots/hyphens/underscores, "
            "optionally with a single '/' separator for username/repo format."
        )

    # Validate S3_BUCKET - must follow S3 naming rules
    # 3-63 chars, lowercase letters, numbers, hyphens, dots
    bucket_pattern = r'^[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$'
    if not re.match(bucket_pattern, s3_bucket):
        raise ValueError(
            f"Invalid S3_BUCKET '{s3_bucket}'. Must be 3-63 characters, "
            "lowercase letters, numbers, dots, and hyphens only."
        )

    # Validate S3_PREFIX - must not contain path traversal sequences
    if '..' in s3_prefix:
        raise ValueError(
            f"Invalid S3_PREFIX '{s3_prefix}'. Cannot contain '..' (path traversal)."
        )

    # Prevent absolute paths in S3_PREFIX
    if s3_prefix.startswith('/'):
        raise ValueError(
            f"Invalid S3_PREFIX '{s3_prefix}'. Cannot start with '/' (absolute path)."
        )

    logger.info("Input validation passed successfully")


def validate_environment() -> dict:
    """
    Validate and extract required environment variables.

    Returns:
        dict: Configuration dictionary with all environment variables

    Raises:
        SystemExit: Exits with code 1 if required variables are missing
    """
    config = {}

    # Required environment variables
    model_name = os.environ.get("MODEL_NAME")
    if not model_name:
        logger.error("MODEL_NAME environment variable is required")
        sys.exit(1)
    config["model_name"] = model_name

    s3_bucket = os.environ.get("S3_BUCKET")
    if not s3_bucket:
        logger.error("S3_BUCKET environment variable is required")
        sys.exit(1)
    config["s3_bucket"] = s3_bucket

    s3_endpoint_url = os.environ.get("S3_ENDPOINT_URL")
    if not s3_endpoint_url:
        logger.error("S3_ENDPOINT_URL environment variable is required")
        sys.exit(1)
    config["s3_endpoint_url"] = s3_endpoint_url

    # AWS credentials are required for S3 operations
    aws_access_key = os.environ.get("AWS_ACCESS_KEY_ID")
    if not aws_access_key:
        logger.error("AWS_ACCESS_KEY_ID environment variable is required")
        sys.exit(1)
    config["aws_access_key_id"] = aws_access_key

    aws_secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
    if not aws_secret_key:
        logger.error("AWS_SECRET_ACCESS_KEY environment variable is required")
        sys.exit(1)
    config["aws_secret_access_key"] = aws_secret_key

    # Optional environment variables
    config["s3_prefix"] = os.environ.get("S3_PREFIX", "models/")
    config["hf_token"] = os.environ.get("HF_TOKEN")
    config["download_dir"] = os.environ.get("DOWNLOAD_DIR", "/tmp")

    # Validate inputs to prevent security vulnerabilities
    try:
        validate_inputs(
            config["model_name"],
            config["s3_bucket"],
            config["s3_prefix"]
        )
    except ValueError as e:
        logger.error(f"Input validation failed: {e}")
        sys.exit(1)

    # Log configuration (mask sensitive credentials)
    logger.info(f"Configuration loaded:")
    logger.info(f"  MODEL_NAME: {config['model_name']}")
    logger.info(f"  S3_BUCKET: {config['s3_bucket']}")
    logger.info(f"  S3_ENDPOINT_URL: {config['s3_endpoint_url']}")
    logger.info(f"  S3_PREFIX: {config['s3_prefix']}")
    logger.info(f"  AWS_ACCESS_KEY_ID: {mask_credential(config['aws_access_key_id'])}")
    logger.info(f"  AWS_SECRET_ACCESS_KEY: {mask_credential(config['aws_secret_access_key'])}")
    logger.info(f"  HF_TOKEN: {'***' if config['hf_token'] else 'not set'}")
    logger.info(f"  DOWNLOAD_DIR: {config['download_dir']}")

    return config


def create_s3_client(endpoint_url: str, aws_access_key_id: str, aws_secret_access_key: str):
    """
    Create S3 client configured for Ceph RGW via Civo.

    Args:
        endpoint_url: S3-compatible endpoint URL (e.g., https://objectstore.lon1.civo.com)
        aws_access_key_id: AWS access key ID
        aws_secret_access_key: AWS secret access key

    Returns:
        boto3 S3 client configured with custom endpoint
    """
    logger.info(f"Creating S3 client with endpoint: {endpoint_url}")

    return boto3.client(
        's3',
        endpoint_url=endpoint_url,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key
    )


def model_exists_in_s3(s3_client, bucket: str, prefix: str) -> bool:
    """
    Check if model exists in S3 using efficient list operation.

    Args:
        s3_client: Boto3 S3 client
        bucket: S3 bucket name
        prefix: Full S3 prefix path (including model name)

    Returns:
        bool: True if at least one object exists under prefix

    Raises:
        ClientError: If S3 operation fails
    """
    logger.info(f"Checking S3 for existing model: s3://{bucket}/{prefix}")

    try:
        response = s3_client.list_objects_v2(
            Bucket=bucket,
            Prefix=prefix,
            MaxKeys=1  # Only need to know if at least one file exists
        )
        exists = 'Contents' in response

        if exists:
            logger.info(f"Model found in S3 at s3://{bucket}/{prefix}")
        else:
            logger.info(f"Model not found in S3 at s3://{bucket}/{prefix}")

        return exists
    except ClientError as e:
        logger.error(f"Error checking S3: {e}")
        raise


def download_model_from_hf(
    model_name: str,
    download_dir: str,
    hf_token: Optional[str] = None
) -> str:
    """
    Download model from HuggingFace to local directory.

    Args:
        model_name: HuggingFace repo ID (e.g., meta-llama/Llama-2-7b-hf)
        download_dir: Local directory for download
        hf_token: Optional HF token for gated models

    Returns:
        str: Path to downloaded model directory

    Raises:
        Exception: On download failure
    """
    logger.info(f"Downloading model '{model_name}' from HuggingFace...")
    logger.info(f"Download directory: {download_dir}")
    logger.info("Starting download - this may take a while for large models...")

    start_time = time.time()

    download_path = snapshot_download(
        repo_id=model_name,
        local_dir=download_dir,
        local_dir_use_symlinks=False,  # Get actual files, not symlinks
        resume_download=True,  # Allow resuming interrupted downloads
        token=hf_token  # Will be None if not provided
    )

    elapsed_time = time.time() - start_time
    logger.info(f"Download complete: {download_path}")
    logger.info(f"Download took {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")

    # Validate download - check that files were actually downloaded
    if not os.path.exists(download_path):
        logger.error(f"Download path does not exist: {download_path}")
        raise RuntimeError(
            f"Download validation failed: path {download_path} does not exist"
        )

    # Count files and calculate total size
    download_path_obj = Path(download_path)
    all_files = list(download_path_obj.rglob('*'))
    file_count = sum(1 for f in all_files if f.is_file())
    total_size_bytes = sum(f.stat().st_size for f in all_files if f.is_file())
    total_size_gb = total_size_bytes / (1024**3)

    if file_count == 0:
        logger.warning(
            "Download directory is empty - model repo may be empty or "
            "download may have failed silently"
        )
    else:
        logger.info(f"Downloaded {file_count} files")
        logger.info(f"Total download size: {total_size_gb:.2f} GB ({total_size_bytes:,} bytes)")

        # Log download speed
        if elapsed_time > 0:
            speed_mbps = (total_size_bytes / (1024**2)) / elapsed_time
            logger.info(f"Average download speed: {speed_mbps:.2f} MB/s")

        # Log largest files for debugging
        files_with_sizes = [(f, f.stat().st_size) for f in all_files if f.is_file()]
        files_with_sizes.sort(key=lambda x: x[1], reverse=True)
        logger.info(f"Largest files downloaded:")
        for file_path, size in files_with_sizes[:5]:  # Top 5 largest files
            size_mb = size / (1024**2)
            logger.info(f"  - {file_path.name}: {size_mb:.2f} MB")

    return download_path


def upload_file_to_s3(
    s3_client,
    local_file_path: str,
    bucket: str,
    s3_key: str
) -> int:
    """
    Upload a single file to S3.

    Args:
        s3_client: Boto3 S3 client
        local_file_path: Path to local file
        bucket: Target S3 bucket
        s3_key: S3 key for the file

    Returns:
        int: Size of uploaded file in bytes

    Raises:
        Exception: On upload failure
    """
    s3_client.upload_file(local_file_path, bucket, s3_key)
    return os.path.getsize(local_file_path)


def upload_directory_to_s3(
    s3_client,
    local_dir: str,
    bucket: str,
    s3_prefix: str,
    max_workers: int = 10
):
    """
    Upload directory recursively to S3 with concurrent uploads.

    Args:
        s3_client: Boto3 S3 client
        local_dir: Local directory to upload
        bucket: Target S3 bucket
        s3_prefix: S3 prefix for uploaded files
        max_workers: Maximum number of concurrent upload threads

    Raises:
        Exception: On upload failure
    """
    # Ensure s3_prefix ends with /
    if not s3_prefix.endswith('/'):
        s3_prefix += '/'

    logger.info(f"Uploading {local_dir} to s3://{bucket}/{s3_prefix}")

    # Collect all files to upload
    files_to_upload = []
    for root, dirs, files in os.walk(local_dir):
        for file in files:
            local_file_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_file_path, local_dir)
            s3_key = s3_prefix + relative_path
            files_to_upload.append((local_file_path, s3_key, relative_path))

    if not files_to_upload:
        logger.warning("No files to upload")
        return

    logger.info(f"Uploading {len(files_to_upload)} files with "
                f"{max_workers} concurrent workers")

    # Upload files concurrently with progress bar
    total_size = 0
    file_count = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all upload tasks
        future_to_file = {}
        for local_file_path, s3_key, relative_path in files_to_upload:
            future = executor.submit(
                upload_file_to_s3,
                s3_client,
                local_file_path,
                bucket,
                s3_key
            )
            future_to_file[future] = (relative_path, s3_key)

        # Process completed uploads with progress bar
        with tqdm(
            total=len(files_to_upload),
            desc="Uploading files",
            unit="file",
            disable=os.environ.get("DISABLE_PROGRESS", "").lower() == "true"
        ) as pbar:
            for future in as_completed(future_to_file):
                relative_path, s3_key = future_to_file[future]
                try:
                    file_size = future.result()
                    total_size += file_size
                    file_count += 1
                    pbar.update(1)
                    # Use tqdm.write for logging to avoid progress bar conflicts
                    tqdm.write(f"  Uploaded: {relative_path} -> {s3_key}")
                except Exception as e:
                    logger.error(
                        f"Failed to upload {relative_path}: {e}",
                        exc_info=True
                    )
                    raise

    logger.info(
        f"Upload complete: {file_count} files, "
        f"{total_size / 1024 / 1024:.2f} MB total"
    )


def cleanup_local_files(directory: str):
    """
    Remove local downloaded files.

    Args:
        directory: Directory to remove

    Note:
        Cleanup is best-effort and will not raise errors. Failures are logged
        as warnings but don't prevent the script from completing successfully.
    """
    logger.info(f"Cleaning up local files: {directory}")

    try:
        if os.path.exists(directory):
            shutil.rmtree(directory)
            logger.info("Cleanup complete")
        else:
            logger.info("Directory does not exist, skipping cleanup")
    except OSError as e:
        # Best-effort cleanup - don't fail on cleanup errors
        # OSError covers filesystem issues, permissions, etc.
        logger.warning(
            f"Cleanup failed (non-fatal): {e}",
            exc_info=True
        )


def main():
    """
    Main execution flow with proper error handling and logging.

    Exit codes:
        0: Success (model ready in S3)
        1: Failure (any error occurred)
    """
    logger.info("=" * 60)
    logger.info("HuggingFace Model Cache Loader - Starting")
    logger.info("=" * 60)

    download_dir = None  # Track for cleanup in finally block

    try:
        # Step 1: Validate environment variables
        config = validate_environment()

        # Step 2: Create S3 client with custom endpoint for Ceph RGW
        s3_client = create_s3_client(
            config["s3_endpoint_url"],
            config["aws_access_key_id"],
            config["aws_secret_access_key"]
        )

        # Step 3: Construct full S3 prefix
        s3_prefix = config["s3_prefix"]
        if s3_prefix and not s3_prefix.endswith('/'):
            s3_prefix += '/'
        full_s3_prefix = f"{s3_prefix}{config['model_name']}/"

        # Step 4: Check if model exists in S3
        if model_exists_in_s3(s3_client, config["s3_bucket"], full_s3_prefix):
            logger.info("Model found in S3, skipping download")
            logger.info("=" * 60)
            logger.info("SUCCESS: Model is ready in S3")
            logger.info("=" * 60)
            sys.exit(0)

        # Step 5: Model not found, proceed with download
        logger.info("Model not found in S3, starting download from HuggingFace")

        # Step 6: Create secure temporary download directory
        # Use tempfile.mkdtemp() for secure permissions (0700)
        download_dir = tempfile.mkdtemp(
            prefix="hf_model_",
            dir=config["download_dir"]
        )
        logger.info(f"Created secure download directory: {download_dir}")

        # Step 7: Download model from HuggingFace
        download_model_from_hf(
            config["model_name"],
            download_dir,
            config["hf_token"]
        )

        # Step 8: Upload to S3
        upload_directory_to_s3(
            s3_client,
            download_dir,
            config["s3_bucket"],
            full_s3_prefix
        )

        # Step 9: Success
        logger.info("=" * 60)
        logger.info("SUCCESS: Model downloaded and uploaded to S3")
        logger.info("=" * 60)
        sys.exit(0)

    except ClientError as e:
        # S3/boto3 specific errors
        logger.error("=" * 60)
        logger.error("FAILED: S3 operation error")
        logger.error("=" * 60)
        logger.error(f"S3 Error ({e.response['Error']['Code']}): {e}", exc_info=True)
        sys.exit(1)

    except HfHubHTTPError as e:
        # HuggingFace Hub specific errors (auth, not found, etc.)
        logger.error("=" * 60)
        logger.error("FAILED: HuggingFace Hub error")
        logger.error("=" * 60)
        logger.error(f"HF Hub Error: {str(e)}", exc_info=True)
        sys.exit(1)

    except OSError as e:
        # Filesystem errors (permissions, disk full, etc.)
        logger.error("=" * 60)
        logger.error("FAILED: Filesystem error")
        logger.error("=" * 60)
        logger.error(f"Filesystem Error: {str(e)}", exc_info=True)
        sys.exit(1)

    except Exception as e:
        # Catch-all for unexpected errors
        logger.error("=" * 60)
        logger.error("FAILED: An unexpected error occurred")
        logger.error("=" * 60)
        logger.error(
            f"Unexpected Error ({type(e).__name__}): {str(e)}",
            exc_info=True
        )
        logger.warning(
            "This is an unexpected error type. Please report this issue."
        )
        sys.exit(1)

    finally:
        # Step 10: Cleanup local files (runs even on failure)
        if download_dir and os.path.exists(download_dir):
            cleanup_local_files(download_dir)


if __name__ == "__main__":
    main()
