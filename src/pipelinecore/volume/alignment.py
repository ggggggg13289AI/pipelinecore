"""Volume alignment operations.

Pure numpy implementations to avoid circular imports.
"""

import numpy as np


def get_dims(shape: tuple, max_channels: int = 10) -> tuple[int, int]:
    """Get the number of dimensions and channels from array shape.

    Args:
        shape: Shape of an array
        max_channels: Maximum possible number of channels

    Returns:
        Tuple of (n_dims, n_channels)
    """
    if shape[-1] <= max_channels:
        n_dims = len(shape) - 1
        n_channels = shape[-1]
    else:
        n_dims = len(shape)
        n_channels = 1
    return n_dims, n_channels


def get_ras_axes(aff: np.ndarray, n_dims: int = 3) -> np.ndarray:
    """Find RAS axes corresponding to each dimension of a volume.

    Args:
        aff: Affine matrix (n_dims x n_dims or n_dims+1 x n_dims+1)
        n_dims: Number of spatial dimensions

    Returns:
        Array of axes corresponding to RAS orientations
    """
    aff_inverted = np.linalg.inv(aff)
    img_ras_axes = np.argmax(np.absolute(aff_inverted[0:n_dims, 0:n_dims]), axis=0)

    for i in range(n_dims):
        if i not in img_ras_axes:
            unique, counts = np.unique(img_ras_axes, return_counts=True)
            incorrect_value = unique[np.argmax(counts)]
            img_ras_axes[np.where(img_ras_axes == incorrect_value)[0][-1]] = i

    return img_ras_axes


def align_volume_to_ref(
    volume: np.ndarray,
    aff: np.ndarray,
    aff_ref: np.ndarray | None = None,
    return_aff: bool = False,
    n_dims: int | None = None,
    return_copy: bool = True,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Align a volume to a reference orientation.

    Args:
        volume: Input numpy array
        aff: Affine matrix of the floating volume
        aff_ref: Affine matrix of target orientation (default: identity)
        return_aff: Whether to return the affine matrix
        n_dims: Number of dimensions (inferred if None)
        return_copy: Whether to return a copy

    Returns:
        Aligned volume, optionally with affine matrix
    """
    # Work on copy
    new_volume = volume.copy() if return_copy else volume
    aff_flo = aff.copy()

    # Default value for aff_ref
    if aff_ref is None:
        aff_ref = np.eye(4)

    # Extract ras axes
    if n_dims is None:
        n_dims, _ = get_dims(new_volume.shape)
    ras_axes_ref = get_ras_axes(aff_ref, n_dims=n_dims)
    ras_axes_flo = get_ras_axes(aff_flo, n_dims=n_dims)

    # Align axes
    aff_flo[:, ras_axes_ref] = aff_flo[:, ras_axes_flo]
    for i in range(n_dims):
        if ras_axes_flo[i] != ras_axes_ref[i]:
            new_volume = np.swapaxes(new_volume, ras_axes_flo[i], ras_axes_ref[i])
            swapped_axis_idx = np.where(ras_axes_flo == ras_axes_ref[i])
            ras_axes_flo[swapped_axis_idx], ras_axes_flo[i] = (
                ras_axes_flo[i],
                ras_axes_flo[swapped_axis_idx],
            )

    # Align directions
    dot_products = np.sum(aff_flo[:3, :3] * aff_ref[:3, :3], axis=0)
    for i in range(n_dims):
        if dot_products[i] < 0:
            new_volume = np.flip(new_volume, axis=i)
            aff_flo[:, i] = -aff_flo[:, i]
            aff_flo[:3, 3] = aff_flo[:3, 3] - aff_flo[:3, i] * (new_volume.shape[i] - 1)

    if return_aff:
        return new_volume, aff_flo
    else:
        return new_volume


__all__ = [
    "align_volume_to_ref",
    "get_dims",
    "get_ras_axes",
]
