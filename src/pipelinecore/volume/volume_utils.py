"""Volume utility functions.

Pure numpy implementations to avoid circular imports.
"""

from typing import Any

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


def reformat_to_list(
    var: Any,
    length: int | None = None,
    load_as_numpy: bool = False,
    dtype: str | None = None,
) -> list | None:
    """Reformat variable into a list of desired length and type.

    Args:
        var: Input variable (str, int, float, list, tuple, or numpy array)
        length: Target list length (single items will be replicated)
        load_as_numpy: Whether var is a path to numpy array
        dtype: Convert items to this type ('int', 'float', 'bool', 'str')

    Returns:
        Reformatted list or None
    """
    if var is None:
        return None

    # Load numpy array if path
    if load_as_numpy and isinstance(var, str):
        var = np.load(var)

    # Convert to list
    if isinstance(var, (int, float, np.integer, np.floating)):
        var = [var]
    elif isinstance(var, tuple):
        var = list(var)
    elif isinstance(var, np.ndarray):
        if var.shape == (1,):
            var = [var[0]]
        else:
            var = np.squeeze(var).tolist()
    elif isinstance(var, str):
        var = [var]
    elif isinstance(var, bool):
        var = [var]

    if isinstance(var, list):
        if length is not None:
            if len(var) == 1:
                var = var * length
            elif len(var) != length:
                raise ValueError(f"var should be of length 1 or {length}, had {len(var)}")
    else:
        raise TypeError("var should be int, float, tuple, list, or numpy array")

    # Convert items type
    if dtype is not None:
        type_map = {"int": int, "float": float, "bool": bool, "str": str}
        if dtype not in type_map:
            raise ValueError(f"dtype should be 'str', 'float', 'int', or 'bool'; had {dtype}")
        var = [type_map[dtype](v) for v in var]

    return var


def find_closest_number_divisible_by_m(n: int, m: int, answer_type: str = "lower") -> int:
    """Find closest number divisible by m.

    Args:
        n: Input number
        m: Divisor
        answer_type: "lower", "higher", or "closer"

    Returns:
        Closest divisible number
    """
    if n % m == 0:
        return n

    lower = (n // m) * m
    higher = lower + m

    if answer_type == "lower":
        return lower
    elif answer_type == "higher":
        return higher
    elif answer_type == "closer":
        return lower if (n - lower) <= (higher - n) else higher
    else:
        raise ValueError(f"answer_type should be 'lower', 'higher', or 'closer'; had {answer_type}")


def find_closest_divisible(n: int, divisor: int, answer_type: str = "lower") -> int:
    """Find closest number divisible by divisor.

    Alias for find_closest_number_divisible_by_m with cleaner name.

    Args:
        n: Input number
        divisor: Divisor (e.g., 32 for 5-level UNet)
        answer_type: "lower", "higher", or "closer"

    Returns:
        Closest divisible number
    """
    return find_closest_number_divisible_by_m(n, divisor, answer_type)


__all__ = [
    "find_closest_divisible",
    "find_closest_number_divisible_by_m",
    "get_dims",
    "reformat_to_list",
]
