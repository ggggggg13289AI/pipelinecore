"""Volume utilities for NIfTI file operations.

Consolidates volume operations into a single module with pure numpy implementations.
"""

# Alignment
from pipelinecore.volume.alignment import align_volume_to_ref, get_ras_axes

# Geometry
from pipelinecore.volume.geometry import crop_volume, pad_volume, uncrop_volume, unpad_volume

# Intensity
from pipelinecore.volume.intensity import rescale_volume

# I/O
from pipelinecore.volume.io import get_volume_info, load_nifti, load_volume, save_nifti

# Utils
from pipelinecore.volume.utils import extract_stem
from pipelinecore.volume.validation import validate_dimensions
from pipelinecore.volume.volume_utils import (
    find_closest_divisible,
    find_closest_number_divisible_by_m,
    get_dims,
    reformat_to_list,
)


def restore_volume(
    data,
    pad_idx,
    crop_idx,
    original_shape,
):
    """Restore volume to original size by unpadding and uncropping.

    Args:
        data: Processed volume data
        pad_idx: Padding indices from pad_volume
        crop_idx: Crop indices from crop_volume
        original_shape: Original volume shape

    Returns:
        Volume restored to original dimensions
    """
    # Unpad first
    result = unpad_volume(data, pad_idx)

    # Then uncrop
    result = uncrop_volume(result, original_shape, crop_idx)

    return result


__all__ = [
    # Alignment
    "align_volume_to_ref",
    "get_ras_axes",
    # Geometry
    "crop_volume",
    "pad_volume",
    "uncrop_volume",
    "unpad_volume",
    # Intensity
    "rescale_volume",
    # I/O
    "extract_stem",
    "get_volume_info",
    "load_nifti",
    "load_volume",
    "save_nifti",
    # Utils
    "find_closest_divisible",
    "find_closest_number_divisible_by_m",
    "get_dims",
    "reformat_to_list",
    "validate_dimensions",
    # Composite
    "restore_volume",
]
