"""Tests for pipelinecore.upload module."""

import json
import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, call

import pytest


class TestUploadDicomSeg:
    """Tests for upload_dicom_seg function."""

    @pytest.fixture
    def mock_env(self, monkeypatch: Any) -> None:
        """Mock environment variables."""
        monkeypatch.setenv("PYTHON3", "/usr/bin/python3")

    @pytest.fixture
    def mock_subprocess_popen(self, mocker: Any) -> MagicMock:
        """Mock subprocess.Popen."""
        mock_popen = mocker.patch("subprocess.Popen")
        mock_process = MagicMock()
        mock_process.communicate.return_value = (b"success", b"")
        mock_popen.return_value = mock_process
        return mock_popen

    @pytest.fixture
    def setup_test_files(self, tmp_path: Path) -> dict[str, Path]:
        """Create test files for upload."""
        # Create test DICOM folder
        dicom_folder = tmp_path / "dicom_seg"
        dicom_folder.mkdir()
        
        # Create test DICOM files
        (dicom_folder / "test_volume_001.dcm").write_text("fake dicom data")
        (dicom_folder / "test_volume_002.dcm").write_text("fake dicom data")
        
        # Create test NIfTI file
        nifti_file = tmp_path / "test_volume.nii.gz"
        nifti_file.write_text("fake nifti data")
        
        return {
            "dicom_folder": dicom_folder,
            "nifti_file": nifti_file
        }

    def test_upload_dicom_seg_success(
        self, 
        mock_env: None, 
        mock_subprocess_popen: MagicMock,
        setup_test_files: dict[str, Path]
    ) -> None:
        """Test successful DICOM-SEG upload."""
        from pipelinecore.upload import upload_dicom_seg
        
        dicom_folder = str(setup_test_files["dicom_folder"])
        nifti_file = str(setup_test_files["nifti_file"])
        
        stdout, stderr = upload_dicom_seg(dicom_folder, nifti_file)
        
        # Verify subprocess was called
        assert mock_subprocess_popen.called
        assert stdout == b"success"
        assert stderr == b""

    def test_upload_dicom_seg_finds_matching_files(
        self,
        mock_env: None,
        mock_subprocess_popen: MagicMock,
        setup_test_files: dict[str, Path]
    ) -> None:
        """Test that upload_dicom_seg finds matching DICOM files."""
        from pipelinecore.upload import upload_dicom_seg
        
        dicom_folder = str(setup_test_files["dicom_folder"])
        nifti_file = str(setup_test_files["nifti_file"])
        
        upload_dicom_seg(dicom_folder, nifti_file)
        
        # Verify the command includes the DICOM files
        call_args = mock_subprocess_popen.call_args
        cmd_str = call_args[1]["args"]
        assert "test_volume" in cmd_str
        assert ".dcm" in cmd_str

    def test_upload_dicom_seg_handles_subprocess_error(
        self,
        mock_env: None,
        mock_subprocess_popen: MagicMock,
        setup_test_files: dict[str, Path]
    ) -> None:
        """Test handling of subprocess errors."""
        from pipelinecore.upload import upload_dicom_seg
        
        # Mock subprocess to return error
        mock_process = MagicMock()
        mock_process.communicate.return_value = (b"", b"error occurred")
        mock_subprocess_popen.return_value = mock_process
        
        dicom_folder = str(setup_test_files["dicom_folder"])
        nifti_file = str(setup_test_files["nifti_file"])
        
        stdout, stderr = upload_dicom_seg(dicom_folder, nifti_file)
        
        assert stdout == b""
        assert stderr == b"error occurred"


class TestUploadJson:
    """Tests for upload_json function."""

    @pytest.fixture
    def mock_env(self, monkeypatch: Any) -> None:
        """Mock environment variables."""
        monkeypatch.setenv("PYTHON3", "/usr/bin/python3")
        monkeypatch.setenv("PATH_PROCESS", "/tmp/test_process")

    @pytest.fixture
    def mock_subprocess_popen(self, mocker: Any) -> MagicMock:
        """Mock subprocess.Popen."""
        mock_popen = mocker.patch("subprocess.Popen")
        mock_process = MagicMock()
        mock_process.communicate.return_value = (b"upload success", b"")
        mock_popen.return_value = mock_process
        return mock_popen

    @pytest.fixture
    def setup_cmd_json(self, tmp_path: Path, monkeypatch: Any) -> Path:
        """Create test cmd.json file."""
        process_dir = tmp_path / "Deep_cmd_tools"
        process_dir.mkdir(parents=True)
        
        cmd_json = process_dir / "TEST001_cmd.json"
        cmd_data = [
            {
                "study_id": "TEST001",
                "name": "test_mode",
                "output_list": [
                    "/path/to/output1.nii.gz",
                    "/path/to/output2.nii.gz"
                ]
            }
        ]
        cmd_json.write_text(json.dumps(cmd_data))
        
        # Mock PATH_PROCESS environment variable
        monkeypatch.setenv("PATH_PROCESS", str(tmp_path))
        
        # Create fake platform_json files
        (tmp_path / "output1_platform_json.json").write_text("{}")
        (tmp_path / "output2_platform_json.json").write_text("{}")
        
        # Mock os.path.exists to return True for platform_json files
        import os
        original_exists = os.path.exists
        
        def mock_exists(path: str) -> bool:
            if "platform_json.json" in path:
                return True
            return original_exists(path)
        
        monkeypatch.setattr("os.path.exists", mock_exists)
        
        return cmd_json

    def test_upload_json_success(
        self,
        mock_env: None,
        mock_subprocess_popen: MagicMock,
        setup_cmd_json: Path
    ) -> None:
        """Test successful JSON upload."""
        from pipelinecore.upload import upload_json
        from pipelinecore.inference import InferenceEnum
        
        # Note: This test will need adjustment based on actual InferenceEnum values
        # For now, we'll use a string that matches the name in our test data
        result = upload_json("TEST001", "test_mode")  # type: ignore
        
        # Verify subprocess was called
        assert mock_subprocess_popen.called

    def test_upload_json_file_not_found(
        self,
        mock_env: None,
        monkeypatch: Any
    ) -> None:
        """Test FileNotFoundError when cmd.json doesn't exist."""
        from pipelinecore.upload import upload_json
        
        monkeypatch.setenv("PATH_PROCESS", "/nonexistent/path")
        
        with pytest.raises(FileNotFoundError):
            upload_json("NONEXISTENT", "test_mode")  # type: ignore

    def test_upload_json_no_matching_study(
        self,
        mock_env: None,
        setup_cmd_json: Path
    ) -> None:
        """Test ValueError when study_id doesn't match."""
        from pipelinecore.upload import upload_json
        
        # This should raise ValueError because next() will return None
        # and the code raises ValueError for None
        with pytest.raises((ValueError, StopIteration)):
            upload_json("WRONG_ID", "test_mode")  # type: ignore


class TestUploadIntegration:
    """Integration tests for upload module."""

    def test_upload_workflow(self, tmp_path: Path, monkeypatch: Any, mocker: Any) -> None:
        """Test complete upload workflow."""
        # Setup environment
        monkeypatch.setenv("PYTHON3", "/usr/bin/python3")
        monkeypatch.setenv("PATH_PROCESS", str(tmp_path))
        
        # Mock subprocess
        mock_popen = mocker.patch("subprocess.Popen")
        mock_process = MagicMock()
        mock_process.communicate.return_value = (b"success", b"")
        mock_popen.return_value = mock_process
        
        # Create test files
        dicom_folder = tmp_path / "dicom"
        dicom_folder.mkdir()
        (dicom_folder / "test.dcm").write_text("data")
        
        nifti_file = tmp_path / "test.nii.gz"
        nifti_file.write_text("data")
        
        from pipelinecore.upload import upload_dicom_seg
        
        stdout, stderr = upload_dicom_seg(str(dicom_folder), str(nifti_file))
        
        assert stdout == b"success"
        assert stderr == b""
        assert mock_popen.called
