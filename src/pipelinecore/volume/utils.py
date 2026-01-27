"""Volume utility functions."""

from pathlib import Path


def extract_stem(path: Path | str) -> str:
    """Extract file stem without .nii or .nii.gz extensions.

    Examples:
        brain.nii.gz → brain
        brain.nii → brain
        brain.mgz → brain
    """
    path = Path(path)
    name = path.name

    # Handle double extensions first
    for ext in [".nii.gz"]:
        if name.endswith(ext):
            return name[: -len(ext)]

    # Then single extensions
    return path.stem
