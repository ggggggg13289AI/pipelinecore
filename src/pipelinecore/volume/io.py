"""NIfTI file I/O operations.

Pure implementations without external dependencies on SynthSeg.
"""

from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

from .volume_utils import get_dims


def save_nifti(
    data: np.ndarray,
    affine: np.ndarray,
    header: Any,
    output_path: Path | str,
) -> Path:
    """Save numpy array as NIfTI file.

    Args:
        data: Volume data to save
        affine: Affine transformation matrix
        header: NIfTI header (or None for default)
        output_path: Output file path

    Returns:
        Path to saved file
    """
    output_path = Path(output_path)
    nifti = nib.Nifti1Image(data, affine, header)
    nib.save(nifti, str(output_path))
    return output_path


def load_nifti(path: Path | str) -> tuple[np.ndarray, np.ndarray, Any]:
    """Load NIfTI file and return data, affine, header.

    Returns:
        Tuple of (data, affine, header)
    """
    nifti = nib.load(str(path))
    return (
        np.asarray(nifti.dataobj),
        nifti.affine,
        nifti.header,
    )


def get_volume_info(
    path_volume: str | Path,
    return_volume: bool = False,
    aff_ref: np.ndarray | None = None,
    max_channels: int = 10,
) -> tuple:
    """Get volume information from NIfTI file.

    Args:
        path_volume: Path to volume file
        return_volume: Whether to return the loaded volume
        aff_ref: Reference affine (for alignment)
        max_channels: Maximum number of channels

    Returns:
        Tuple of (volume, shape, aff, n_dims, n_channels, header, resolution)
        If return_volume is False, volume is None
    """
    nifti = nib.load(str(path_volume))
    data = np.asarray(nifti.dataobj)
    aff = nifti.affine
    header = nifti.header

    # Get dimensions
    n_dims, n_channels = get_dims(data.shape, max_channels)

    # Get resolution from header
    resolution = np.array(header.get_zooms()[:n_dims])

    if return_volume:
        return data, data.shape, aff, n_dims, n_channels, header, resolution
    else:
        return None, data.shape, aff, n_dims, n_channels, header, resolution


def load_volume(
    path_volume: str | Path,
    im_only: bool = True,
    squeeze: bool = True,
    dtype: np.dtype | None = None,
    aff_ref: np.ndarray | None = None,
) -> np.ndarray | tuple:
    """Load volume from NIfTI file.

    Args:
        path_volume: Path to volume file
        im_only: Whether to return only the image (not affine/header)
        squeeze: Whether to squeeze singleton dimensions
        dtype: Target data type
        aff_ref: Reference affine (for alignment)

    Returns:
        Volume array, or tuple of (volume, affine, header)
    """
    nifti = nib.load(str(path_volume))
    data = np.asarray(nifti.dataobj)

    if squeeze:
        data = np.squeeze(data)

    if dtype is not None:
        data = data.astype(dtype)

    if im_only:
        return data
    else:
        return data, nifti.affine, nifti.header


__all__ = [
    "get_volume_info",
    "load_nifti",
    "load_volume",
    "save_nifti",
]
