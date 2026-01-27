"""NIfTI file I/O operations."""

from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np


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
