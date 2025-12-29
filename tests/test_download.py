"""
Tests for HuggingFace model download functionality.
"""

import pytest
import sys
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add parent directory to path to import cache_model
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cache_model


class TestDownloadModelFromHF:
    """Tests for HuggingFace model download function."""

    @patch('cache_model.snapshot_download')
    @patch('cache_model.threading.Thread')
    def test_download_model_without_token(self, mock_thread, mock_snapshot, temp_download_dir):
        """Test downloading a model without HF token."""
        # Mock the snapshot_download to return the download path
        mock_snapshot.return_value = temp_download_dir

        # Call download function
        result = cache_model.download_model_from_hf(
            "gpt2",
            temp_download_dir,
            hf_token=None
        )

        # Verify snapshot_download was called with correct base params
        call_kwargs = mock_snapshot.call_args[1]
        assert call_kwargs['repo_id'] == "gpt2"
        assert call_kwargs['local_dir'] == temp_download_dir
        assert call_kwargs['local_dir_use_symlinks'] is False
        assert call_kwargs['resume_download'] is True
        assert call_kwargs['token'] is None
        assert 'max_workers' in call_kwargs  # Should include max_workers now

        assert result == temp_download_dir

    @patch('cache_model.snapshot_download')
    @patch('cache_model.threading.Thread')
    def test_download_model_with_token(self, mock_thread, mock_snapshot, temp_download_dir, mock_hf_token):
        """Test downloading a gated model with HF token."""
        # Mock the snapshot_download to return the download path
        mock_snapshot.return_value = temp_download_dir

        # Call download function with token
        result = cache_model.download_model_from_hf(
            "meta-llama/Llama-2-7b-hf",
            temp_download_dir,
            hf_token=mock_hf_token
        )

        # Verify token was passed
        call_kwargs = mock_snapshot.call_args[1]
        assert call_kwargs['repo_id'] == "meta-llama/Llama-2-7b-hf"
        assert call_kwargs['local_dir'] == temp_download_dir
        assert call_kwargs['local_dir_use_symlinks'] is False
        assert call_kwargs['resume_download'] is True
        assert call_kwargs['token'] == mock_hf_token
        assert 'max_workers' in call_kwargs

        assert result == temp_download_dir

    @patch('cache_model.snapshot_download')
    @patch('cache_model.threading.Thread')
    def test_download_model_no_symlinks(self, mock_thread, mock_snapshot, temp_download_dir):
        """Test that symlinks are disabled in download."""
        mock_snapshot.return_value = temp_download_dir

        cache_model.download_model_from_hf("gpt2", temp_download_dir)

        # Verify local_dir_use_symlinks is False
        call_kwargs = mock_snapshot.call_args[1]
        assert call_kwargs['local_dir_use_symlinks'] is False

    @patch('cache_model.snapshot_download')
    @patch('cache_model.threading.Thread')
    def test_download_model_resume_enabled(self, mock_thread, mock_snapshot, temp_download_dir):
        """Test that resume download is enabled."""
        mock_snapshot.return_value = temp_download_dir

        cache_model.download_model_from_hf("gpt2", temp_download_dir)

        # Verify resume_download is True
        call_kwargs = mock_snapshot.call_args[1]
        assert call_kwargs['resume_download'] is True

    @patch('cache_model.snapshot_download')
    @patch('cache_model.threading.Thread')
    def test_download_model_handles_errors(self, mock_thread, mock_snapshot, temp_download_dir):
        """Test that download function propagates errors."""
        # Mock snapshot_download to raise an error
        mock_snapshot.side_effect = Exception("Download failed")

        with pytest.raises(Exception, match="Download failed"):
            cache_model.download_model_from_hf("gpt2", temp_download_dir)
