"""Volume geometry operations (crop, pad, uncrop, unpad).

Pure numpy implementations to avoid circular imports.
"""

import numpy as np

from .volume_utils import get_dims, reformat_to_list


def crop_volume(
    volume: np.ndarray,
    cropping_margin: int | list | None = None,
    cropping_shape: int | list | None = None,
    aff: np.ndarray | None = None,
    return_crop_idx: bool = False,
    mode: str = "center",
) -> np.ndarray | tuple:
    """Crop volume by a given margin, or to a given shape.

    Args:
        volume: 2d or 3d numpy array
        cropping_margin: Margin by which to crop (applied on both sides)
        cropping_shape: Shape to which volume will be cropped
        aff: Affine matrix (updated if provided)
        return_crop_idx: Whether to return cropping indices
        mode: "center" or "random" (only for cropping_shape)

    Returns:
        Cropped volume, optionally with affine and/or crop indices
    """
    assert (cropping_margin is not None) | (cropping_shape is not None), \
        "cropping_margin or cropping_shape should be provided"
    assert not ((cropping_margin is not None) & (cropping_shape is not None)), \
        "only one of cropping_margin or cropping_shape should be provided"

    new_volume = volume.copy()
    vol_shape = new_volume.shape
    n_dims, _ = get_dims(vol_shape)

    # Find cropping indices
    if cropping_margin is not None:
        cropping_margin = reformat_to_list(cropping_margin, length=n_dims)
        do_cropping = np.array(vol_shape[:n_dims]) > 2 * np.array(cropping_margin)
        min_crop_idx = [cropping_margin[i] if do_cropping[i] else 0 for i in range(n_dims)]
        max_crop_idx = [vol_shape[i] - cropping_margin[i] if do_cropping[i] else vol_shape[i] for i in range(n_dims)]
    else:
        cropping_shape = reformat_to_list(cropping_shape, length=n_dims)
        if mode == "center":
            min_crop_idx = np.maximum([int((vol_shape[i] - cropping_shape[i]) / 2) for i in range(n_dims)], 0)
            max_crop_idx = np.minimum(
                [min_crop_idx[i] + cropping_shape[i] for i in range(n_dims)],
                np.array(vol_shape)[:n_dims],
            )
        elif mode == "random":
            crop_max_val = np.maximum(np.array([vol_shape[i] - cropping_shape[i] for i in range(n_dims)]), 0)
            min_crop_idx = np.random.randint(0, high=crop_max_val + 1)
            max_crop_idx = np.minimum(min_crop_idx + np.array(cropping_shape), np.array(vol_shape)[:n_dims])
        else:
            raise ValueError(f'mode should be "center" or "random", had {mode}')

    crop_idx = np.concatenate([np.array(min_crop_idx), np.array(max_crop_idx)])

    # Crop volume
    if n_dims == 2:
        new_volume = new_volume[crop_idx[0]:crop_idx[2], crop_idx[1]:crop_idx[3], ...]
    elif n_dims == 3:
        new_volume = new_volume[crop_idx[0]:crop_idx[3], crop_idx[1]:crop_idx[4], crop_idx[2]:crop_idx[5], ...]

    # Build output
    output = [new_volume]
    if aff is not None:
        aff = aff.copy()
        aff[0:3, -1] = aff[0:3, -1] + aff[:3, :3] @ np.array(min_crop_idx)
        output.append(aff)
    if return_crop_idx:
        output.append(crop_idx)

    return output[0] if len(output) == 1 else tuple(output)


def pad_volume(
    volume: np.ndarray,
    padding_shape: int | list,
    padding_value: float = 0,
    aff: np.ndarray | None = None,
    return_pad_idx: bool = False,
) -> np.ndarray | tuple:
    """Pad volume to specified shape.

    Args:
        volume: Input numpy array
        padding_shape: Target shape after padding
        padding_value: Value used for padding
        aff: Affine matrix (updated if provided)
        return_pad_idx: Whether to return padding indices

    Returns:
        Padded volume, optionally with affine and/or pad indices
    """
    vol_shape = volume.shape
    n_dims, n_channels = get_dims(vol_shape)
    padding_shape = reformat_to_list(padding_shape, length=n_dims)

    # Compute padding amounts
    pad_amounts = []
    pad_idx = np.zeros((n_dims, 2), dtype=np.int32)

    for i in range(n_dims):
        total_pad = max(0, padding_shape[i] - vol_shape[i])
        pad_before = total_pad // 2
        pad_after = total_pad - pad_before
        pad_amounts.append((pad_before, pad_after))
        pad_idx[i, 0] = pad_before
        pad_idx[i, 1] = pad_before + vol_shape[i]

    # Add channel dimension padding (no padding)
    if n_channels > 1:
        pad_amounts.append((0, 0))

    # Pad the volume
    new_volume = np.pad(volume, pad_amounts, mode="constant", constant_values=padding_value)

    # Build output
    output = [new_volume]
    if aff is not None:
        aff = aff.copy()
        aff[0:3, -1] = aff[0:3, -1] - aff[:3, :3] @ np.array([p[0] for p in pad_amounts[:n_dims]])
        output.append(aff)
    if return_pad_idx:
        output.append(pad_idx)

    return output[0] if len(output) == 1 else tuple(output)


def uncrop_volume(
    volume: np.ndarray,
    original_shape: tuple[int, ...],
    crop_idx: np.ndarray | None,
) -> np.ndarray:
    """Restore cropped volume to original size.

    Args:
        volume: Cropped volume
        original_shape: Original volume shape before cropping
        crop_idx: Crop indices from crop_volume

    Returns:
        Volume restored to original size
    """
    if crop_idx is None:
        return volume

    n_dims = len(original_shape)
    output = np.zeros(original_shape, dtype=volume.dtype)

    # Extract crop boundaries
    if n_dims == 2:
        output[crop_idx[0]:crop_idx[2], crop_idx[1]:crop_idx[3]] = volume
    elif n_dims == 3:
        output[crop_idx[0]:crop_idx[3], crop_idx[1]:crop_idx[4], crop_idx[2]:crop_idx[5]] = volume

    return output


def unpad_volume(
    volume: np.ndarray,
    pad_idx: np.ndarray | None,
) -> np.ndarray:
    """Remove padding from volume.

    Args:
        volume: Padded volume
        pad_idx: Pad indices from pad_volume (shape: n_dims x 2)

    Returns:
        Volume with padding removed
    """
    if pad_idx is None:
        return volume

    n_dims = pad_idx.shape[0]
    slices = tuple(slice(int(pad_idx[i, 0]), int(pad_idx[i, 1])) for i in range(n_dims))

    return volume[slices]


__all__ = [
    "crop_volume",
    "pad_volume",
    "uncrop_volume",
    "unpad_volume",
]
