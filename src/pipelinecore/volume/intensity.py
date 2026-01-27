"""Volume intensity operations.

Pure numpy implementations to avoid circular imports.
"""

import numpy as np


def rescale_volume(
    volume: np.ndarray,
    new_min: float = 0,
    new_max: float = 255,
    min_percentile: float = 2,
    max_percentile: float = 98,
    use_positive_only: bool = False,
) -> np.ndarray:
    """Linearly rescale a volume between new_min and new_max.

    Args:
        volume: Input numpy array
        new_min: Minimum value for rescaled image
        new_max: Maximum value for rescaled image
        min_percentile: Percentile for robust minimum (0 = np.min)
        max_percentile: Percentile for robust maximum (100 = np.max)
        use_positive_only: Whether to use only positive values

    Returns:
        Rescaled volume
    """
    new_volume = volume.copy()
    intensities = new_volume[new_volume > 0] if use_positive_only else new_volume.flatten()

    # Define min and max intensities for normalization
    robust_min = np.min(intensities) if min_percentile == 0 else np.percentile(intensities, min_percentile)
    robust_max = np.max(intensities) if max_percentile == 100 else np.percentile(intensities, max_percentile)

    # Trim values outside range
    new_volume = np.clip(new_volume, robust_min, robust_max)

    # Rescale image
    if robust_min != robust_max:
        return new_min + (new_volume - robust_min) / (robust_max - robust_min) * (new_max - new_min)
    else:
        return np.zeros_like(new_volume)


__all__ = [
    "rescale_volume",
]
