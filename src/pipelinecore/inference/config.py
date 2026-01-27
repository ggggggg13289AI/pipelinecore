import os
import pathlib

from .base import load_config, resolve_enum_mapping_series
from .schema import InferenceEnum


def generate_output_files(
    input_paths: list[str], task_name: str, base_output_path: str, config_path: str = "config.yaml"
) -> list[str]:
    """
    Generate output file names based on input paths, task names, and configuration.

    Args:
        input_paths: List of input file paths
        task_name: Name of the inference task
        base_output_path: Base directory for output files
        config_path: Path to the configuration file

    Returns:
        List of output file paths
    """
    # Load configuration
    config = load_config(config_path)
    output_formats = config.get("output_formats", {})

    # Get output formats for the task
    task_formats = output_formats.get(task_name, [])

    # If no formats defined for the task, return empty list
    if not task_formats:
        return []

    output_files = []

    # Special handling for CMB task which needs to determine SWAN file
    if task_name == InferenceEnum.CMB and len(input_paths) >= 2:
        base_name1 = os.path.basename(input_paths[0]).split(".")[0]
        base_name2 = os.path.basename(input_paths[1]).split(".")[0]

        swan_base_name = base_name1 if base_name1.startswith("SWAN") else base_name2
        other_base_name = base_name2 if base_name1.startswith("SWAN") else base_name1

        for format_str in task_formats:
            # Replace placeholders in format string
            file_name = format_str.format(
                swan_base_name=swan_base_name,
                other_base_name=other_base_name,
                task_name=task_name,
                base_name=None,  # Not used in this case
            )
            output_files.append(os.path.join(base_output_path, file_name))

    # General case for other tasks
    else:
        for format_str in task_formats:
            # For formats that need to be applied to each input file
            if "{base_name}" in format_str:
                for input_path in input_paths:
                    base_name = os.path.basename(input_path).split(".")[0]
                    file_name = format_str.format(base_name=base_name, task_name=task_name)
                    output_files.append(os.path.join(base_output_path, file_name))
            # For formats that are applied once
            else:
                file_name = format_str.format(task_name=task_name)
                output_files.append(os.path.join(base_output_path, file_name))

    return output_files


config_file = pathlib.Path(__file__).parent.joinpath("config.yaml")
CONFIG_DICT = load_config(config_file)
MODEL_MAPPING_SERIES_CONFIG = CONFIG_DICT.get("model_mapping_series", {})
MODEL_MAPPING_SERIES_DICT = resolve_enum_mapping_series(MODEL_MAPPING_SERIES_CONFIG)
