"""Volume validation operations."""

import numpy as np


def validate_dimensions(
    arr: np.ndarray,
    n_dims: int,
    n_channels: int,
) -> np.ndarray:
    """Validate and fix volume dimensions.

    Args:
        arr: Input array
        n_dims: Number of spatial dimensions
        n_channels: Number of channels

    Returns:
        Array with correct dimensions
    """
    # Handle 2D volumes
    if n_dims == 2:
        arr = arr[..., np.newaxis]

    # Remove channel dimension if single channel
    if n_channels == 1 and arr.ndim == 4:
        arr = arr[..., 0]

    return arr


__all__ = [
    "validate_dimensions",
]
