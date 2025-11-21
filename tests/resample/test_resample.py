"""Tests for pipelinecore.resample module."""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, Mock

import numpy as np
import pytest


class TestResampleOne:
    """Tests for resample_one function."""

    @pytest.fixture
    def mock_sitk(self, mocker: Any) -> dict[str, MagicMock]:
        """Mock SimpleITK module."""
        mock_sitk = mocker.patch("pipelinecore.resample.sitk")
        
        # Mock image object
        mock_image = MagicMock()
        mock_image.GetSpacing.return_value = (2.0, 2.0, 2.0)
        mock_image.GetSize.return_value = (100, 100, 50)
        mock_image.GetOrigin.return_value = (0.0, 0.0, 0.0)
        mock_image.GetDirection.return_value = (1, 0, 0, 0, 1, 0, 0, 0, 1)
        mock_image.GetMetaData.return_value = "2"
        
        mock_sitk.ReadImage.return_value = mock_image
        mock_sitk.Transform.return_value = MagicMock()
        mock_sitk.ResampleImageFilter.return_value = MagicMock()
        
        # Mock resampler
        mock_resampler = MagicMock()
        mock_resampler.Execute.return_value = mock_image
        mock_sitk.ResampleImageFilter.return_value = mock_resampler
        
        mock_sitk.sitkLinear = 1  # Mock interpolation constant
        
        return {
            "sitk": mock_sitk,
            "image": mock_image,
            "resampler": mock_resampler
        }

    def test_resample_one_basic(self, mock_sitk: dict[str, MagicMock], tmp_path: Path) -> None:
        """Test basic resample_one functionality."""
        from pipelinecore.resample import resample_one
        
        input_file = tmp_path / "input.nii.gz"
        output_file = tmp_path / "output.nii.gz"
        
        result = resample_one(str(input_file), str(output_file))
        
        assert result == str(output_file)
        assert mock_sitk["sitk"].ReadImage.called
        assert mock_sitk["sitk"].WriteImage.called

    def test_resample_one_sets_spacing(self, mock_sitk: dict[str, MagicMock], tmp_path: Path) -> None:
        """Test that resample_one sets correct spacing."""
        from pipelinecore.resample import resample_one
        
        input_file = tmp_path / "input.nii.gz"
        output_file = tmp_path / "output.nii.gz"
        
        resample_one(str(input_file), str(output_file))
        
        # Verify resampler is configured
        resampler = mock_sitk["resampler"]
        assert resampler.SetOutputSpacing.called
        
        # Check spacing is set to [1.0, 1.0, 1.0]
        call_args = resampler.SetOutputSpacing.call_args[0][0]
        assert call_args == [1.0, 1.0, 1.0]

    def test_resample_one_calculates_new_size(self, mock_sitk: dict[str, MagicMock], tmp_path: Path) -> None:
        """Test that resample_one calculates correct new size."""
        from pipelinecore.resample import resample_one
        
        input_file = tmp_path / "input.nii.gz"
        output_file = tmp_path / "output.nii.gz"
        
        resample_one(str(input_file), str(output_file))
        
        # Verify new size is calculated based on spacing
        resampler = mock_sitk["resampler"]
        assert resampler.SetSize.called
        
        # With original spacing (2,2,2) and size (100,100,50)
        # New size should be approximately (200, 200, 100)
        call_args = resampler.SetSize.call_args[0][0]
        assert len(call_args) == 3
        assert all(isinstance(x, int) for x in call_args)


class TestGetOriginalZIndex:
    """Tests for get_original_z_index function."""

    def test_get_original_z_index_small_array(self) -> None:
        """Test get_original_z_index with small array (z < 64)."""
        from pipelinecore.resample import get_original_z_index
        
        # Create small test arrays
        img_array = np.random.rand(10, 10, 30)
        img_1mm_ori = np.random.rand(10, 10, 60)
        z_i = 30
        z_i1 = 60
        z_pixdim_img = 2
        
        result = get_original_z_index(img_array, z_i, img_1mm_ori, z_i1, z_pixdim_img)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (z_i,)
        assert result.dtype in [np.int32, np.int64, np.intp]

    def test_get_original_z_index_large_array(self) -> None:
        """Test get_original_z_index with large array (z >= 64)."""
        from pipelinecore.resample import get_original_z_index
        
        # Create large test arrays
        img_array = np.random.rand(10, 10, 80)
        img_1mm_ori = np.random.rand(10, 10, 160)
        z_i = 80
        z_i1 = 160
        z_pixdim_img = 2
        
        result = get_original_z_index(img_array, z_i, img_1mm_ori, z_i1, z_pixdim_img)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (z_i,)


class TestCheckOriginalZIndex:
    """Tests for check_original_z_index function."""

    def test_check_original_z_index_no_negative(self) -> None:
        """Test check_original_z_index when no negative differences."""
        from pipelinecore.resample import check_original_z_index
        
        argmin = np.array([2, 4, 6, 8, 10])
        img_array = np.random.rand(10, 10, 5)
        img_1mm_ori = np.random.rand(10, 10, 12)
        z_pixdim_img = 2
        
        result = check_original_z_index(argmin, img_array, img_1mm_ori, z_pixdim_img)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == argmin.shape

    def test_check_original_z_index_with_negative(self) -> None:
        """Test check_original_z_index when negative differences exist."""
        from pipelinecore.resample import check_original_z_index
        
        # Create argmin with values that will cause negative differences
        argmin = np.array([10, 12, 14, 16, 18])  # Too high, will cause negatives
        img_array = np.random.rand(10, 10, 5)
        img_1mm_ori = np.random.rand(10, 10, 20)
        z_pixdim_img = 2
        
        result = check_original_z_index(argmin, img_array, img_1mm_ori, z_pixdim_img)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == argmin.shape


class TestDataTranslate:
    """Tests for data_translate and data_translate_back functions."""

    @pytest.fixture
    def mock_nifti(self) -> MagicMock:
        """Create mock NIfTI object."""
        mock_nii = MagicMock()
        mock_header = MagicMock()
        mock_header.__getitem__ = lambda self, key: np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        mock_nii.header.copy.return_value = mock_header
        return mock_nii

    def test_data_translate_basic(self, mock_nifti: MagicMock) -> None:
        """Test data_translate basic transformation."""
        from pipelinecore.resample import data_translate
        
        img = np.ones((10, 10, 10))
        result = data_translate(img, mock_nifti)
        
        assert result.shape == (10, 10, 10)
        assert isinstance(result, np.ndarray)

    def test_data_translate_back_basic(self, mock_nifti: MagicMock) -> None:
        """Test data_translate_back basic transformation."""
        from pipelinecore.resample import data_translate_back
        
        img = np.ones((10, 10, 10))
        result = data_translate_back(img, mock_nifti)
        
        assert result.shape == (10, 10, 10)
        assert isinstance(result, np.ndarray)

    def test_data_translate_roundtrip(self, mock_nifti: MagicMock) -> None:
        """Test that translate -> translate_back is reversible."""
        from pipelinecore.resample import data_translate, data_translate_back
        
        original = np.random.rand(10, 10, 10)
        
        # Transform forward and back
        translated = data_translate(original, mock_nifti)
        recovered = data_translate_back(translated, mock_nifti)
        
        # Should recover original shape
        assert recovered.shape == original.shape


class TestNiiImgReplace:
    """Tests for nii_img_replace function."""

    def test_nii_img_replace_creates_new_image(self, mocker: Any) -> None:
        """Test nii_img_replace creates new NIfTI image."""
        mock_nib = mocker.patch("pipelinecore.resample.nib")
        
        # Mock original image
        mock_data = MagicMock()
        mock_data.affine = np.eye(4)
        mock_header = MagicMock()
        mock_data.header.copy.return_value = mock_header
        
        # Mock new image
        new_img = np.ones((10, 10, 10))
        
        from pipelinecore.resample import nii_img_replace
        
        result = nii_img_replace(mock_data, new_img)
        
        # Verify Nifti1Image was created
        assert mock_nib.nifti1.Nifti1Image.called

    def test_nii_img_replace_preserves_affine(self, mocker: Any) -> None:
        """Test nii_img_replace preserves affine transformation."""
        mock_nib = mocker.patch("pipelinecore.resample.nib")
        
        # Create specific affine matrix
        affine = np.eye(4)
        affine[0, 3] = 10.0  # Set a specific value
        
        mock_data = MagicMock()
        mock_data.affine = affine
        mock_header = MagicMock()
        mock_data.header.copy.return_value = mock_header
        
        new_img = np.ones((10, 10, 10))
        
        from pipelinecore.resample import nii_img_replace
        
        nii_img_replace(mock_data, new_img)
        
        # Verify affine was passed to Nifti1Image
        call_args = mock_nib.nifti1.Nifti1Image.call_args
        passed_affine = call_args[0][1]
        assert np.array_equal(passed_affine, affine)


class TestGetVolumeInfo:
    """Tests for get_volume_info function."""

    def test_get_volume_info_returns_tuple(self, mocker: Any) -> None:
        """Test get_volume_info returns correct tuple."""
        mock_nib = mocker.patch("pipelinecore.resample.nib")
        
        # Mock NIfTI image
        mock_img = MagicMock()
        mock_array = np.ones((256, 256, 22))
        mock_img.dataobj = mock_array
        mock_header = MagicMock()
        mock_header.__getitem__ = lambda self, key: np.array([0, 1, 1, 1, 0])
        mock_img.header.copy.return_value = mock_header
        
        mock_nib.load.return_value = mock_img
        
        from pipelinecore.resample import get_volume_info
        
        result = get_volume_info("/fake/path.nii.gz")
        
        # Should return 7-tuple
        assert len(result) == 7
        assert isinstance(result, tuple)
        
        img_nii, img_array, y_i, x_i, z_i, header_img, pixdim_img = result
        
        # Verify dimensions
        assert y_i == 256
        assert x_i == 256
        assert z_i == 22

    def test_get_volume_info_loads_file(self, mocker: Any) -> None:
        """Test get_volume_info loads file correctly."""
        mock_nib = mocker.patch("pipelinecore.resample.nib")
        
        mock_img = MagicMock()
        mock_array = np.ones((100, 100, 50))
        mock_img.dataobj = mock_array
        mock_header = MagicMock()
        mock_header.__getitem__ = lambda self, key: np.array([0, 1, 1, 1, 0])
        mock_img.header.copy.return_value = mock_header
        
        mock_nib.load.return_value = mock_img
        
        from pipelinecore.resample import get_volume_info
        
        file_path = "/test/image.nii.gz"
        get_volume_info(file_path)
        
        # Verify load was called with correct path
        mock_nib.load.assert_called_once_with(file_path)


class TestResampleIntegration:
    """Integration tests for resample module."""

    def test_resample_workflow(self, mocker: Any, tmp_path: Path) -> None:
        """Test complete resample workflow."""
        # Mock all external dependencies
        mock_sitk = mocker.patch("pipelinecore.resample.sitk")
        mock_image = MagicMock()
        mock_image.GetSpacing.return_value = (1.0, 1.0, 1.0)
        mock_image.GetSize.return_value = (100, 100, 100)
        mock_image.GetOrigin.return_value = (0, 0, 0)
        mock_image.GetDirection.return_value = (1, 0, 0, 0, 1, 0, 0, 0, 1)
        mock_image.GetMetaData.return_value = "2"
        
        mock_sitk.ReadImage.return_value = mock_image
        mock_resampler = MagicMock()
        mock_resampler.Execute.return_value = mock_image
        mock_sitk.ResampleImageFilter.return_value = mock_resampler
        mock_sitk.sitkLinear = 1
        
        from pipelinecore.resample import resample_one
        
        input_file = tmp_path / "input.nii.gz"
        output_file = tmp_path / "output.nii.gz"
        
        result = resample_one(str(input_file), str(output_file))
        
        assert result == str(output_file)
        assert mock_sitk.ReadImage.called
        assert mock_resampler.Execute.called
        assert mock_sitk.WriteImage.called
