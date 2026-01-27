import argparse
import os
import pathlib
from typing import Any

import numpy as np
import pandas as pd
import yaml
from code_ai.utils import replace_suffix

# 假設這些是原有的枚舉類型
from .schema import (
    Analysis,
    InferenceCmd,
    InferenceCmdItem,
    InferenceEnum,
    MRSeriesRenameEnum,
    T1SeriesRenameEnum,
    T2SeriesRenameEnum,
    Task,
)


def load_config(config_path: str = "config.yaml") -> dict[str, Any]:
    """
    Load configuration from YAML file.
    """
    with open(config_path) as file:
        config = yaml.safe_load(file)
    return config


def get_enum_by_name(enum_name: str) -> Any:
    """
    Get enum instance by its name string.

    Args:
        enum_name: String in format 'EnumClassName.ENUM_VALUE'

    Returns:
        Enum instance or None if not found
    """
    # 分離類名和枚舉值
    parts = enum_name.split(".")
    if len(parts) != 2:
        return None

    enum_class_name, enum_value = parts

    # 映射枚舉類名到枚舉類
    enum_classes = {
        "T1SeriesRenameEnum": T1SeriesRenameEnum,
        "T2SeriesRenameEnum": T2SeriesRenameEnum,
        "MRSeriesRenameEnum": MRSeriesRenameEnum,
        "InferenceEnum": InferenceEnum,
    }

    enum_class = enum_classes.get(enum_class_name)
    if not enum_class:
        return None

    # 獲取枚舉值
    try:
        return getattr(enum_class, enum_value)
    except AttributeError:
        return None


def resolve_enum_mapping_series(mapping_config: dict) -> dict:
    """
    Resolve enum mapping series from configuration.

    Args:
        mapping_config: Configuration dictionary with enum string names

    Returns:
        Dictionary with actual enum instances
    """
    resolved_mapping = {}

    for model_name, series_list in mapping_config.items():
        # 將模型名稱轉換為InferenceEnum
        model_enum = InferenceEnum(model_name)
        resolved_series_list = []

        for series in series_list:
            # 將每個系列中的字符串轉換為枚舉
            resolved_series = []
            for enum_name in series:
                enum_instance = get_enum_by_name(enum_name)
                if enum_instance:
                    resolved_series.append(enum_instance)

            if resolved_series:
                resolved_series_list.append(resolved_series)

        if resolved_series_list:
            resolved_mapping[model_enum] = resolved_series_list

    return resolved_mapping


def check_study_mapping_inference(
    study_path: pathlib.Path, config_path: str = "config.yaml"
) -> dict[str, dict[str, str]]:
    """
    Check study mapping inference using configuration from YAML.
    """
    config = load_config(config_path)
    model_mapping_series_config = config.get("model_mapping_series", {})

    # 解析配置中的枚舉值
    model_mapping_series_dict = resolve_enum_mapping_series(model_mapping_series_config)

    file_list = sorted(study_path.iterdir())

    if any(filter(lambda x: x.name.endswith("nii.gz") or x.name.endswith("nii"), file_list)):
        df_file = pd.DataFrame(file_list, columns=["file_path"])
        df_file["file_name"] = df_file["file_path"].map(lambda x: x.name.replace(".nii.gz", ""))
        model_mapping_dict = {}

        for model_name, model_mapping_series_list in model_mapping_series_dict.items():
            for mapping_series in model_mapping_series_list:
                # 將枚舉轉換為其值以進行比較
                mapping_series_values = [enum.value for enum in mapping_series]
                result = np.intersect1d(df_file["file_name"], mapping_series_values, return_indices=True)

                if result[0].shape[0] >= len(mapping_series_values):
                    df_result = df_file.iloc()[result[1]]
                    file_path = list(map(lambda x: str(x), df_result["file_path"].to_list()))
                    model_mapping_dict.update({model_name.value: file_path})
                    break

        return {study_path.name: model_mapping_dict}
    return None


def get_file_list(input_path: pathlib.Path, suffixes: str, filter_name=None) -> list[pathlib.Path]:
    if any(suffix in input_path.suffixes for suffix in suffixes):
        file_list = [input_path]
    else:
        file_list = sorted(list(input_path.rglob("*.nii*")))
    if filter_name:
        file_list = [f for f in file_list if filter_name in f.name]
    return file_list


def prepare_output_file_list(
    file_list: list[pathlib.Path], suffix: str, output_dir: pathlib.Path | None = None
) -> list[pathlib.Path]:
    return [
        output_dir.joinpath(replace_suffix(f"{x.name}", suffix))
        if output_dir
        else x.parent.joinpath(replace_suffix(x.name, suffix))
        for x in file_list
    ]


def check_study_mapping_inference(
    study_path: pathlib.Path, config_path: str = "config.yaml"
) -> dict[str, dict[Any, Any]] | None:
    """
    Check study mapping inference using configuration from YAML.
    """
    config = load_config(config_path)
    model_mapping_series_dict = config.get("model_mapping_series", {})

    file_list = sorted(study_path.iterdir())

    if any(filter(lambda x: x.name.endswith("nii.gz") or x.name.endswith("nii"), file_list)):
        df_file = pd.DataFrame(file_list, columns=["file_path"])
        df_file["file_name"] = df_file["file_path"].map(lambda x: x.name.replace(".nii.gz", ""))
        model_mapping_dict = {}

        for model_name, model_mapping_series_list in model_mapping_series_dict.items():
            for mapping_series in model_mapping_series_list:
                result = np.intersect1d(df_file["file_name"], mapping_series, return_indices=True)

                if result[0].shape[0] >= len(mapping_series):
                    df_result = df_file.iloc()[result[1]]
                    file_path = list(map(lambda x: str(x), df_result["file_path"].to_list()))
                    model_mapping_dict.update({model_name: file_path})
                    break

        return {study_path.name: model_mapping_dict}
    return None


def generate_output_files(
    input_paths: list[str], task_name: str, base_output_path: str, config_path: str = "config.yaml"
) -> list[str]:
    """
    Generate output file names based on input paths, task names, and configuration.
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

    for format_spec in task_formats:
        template = format_spec.get("template", "")
        apply_to = format_spec.get("apply_to", "once")
        special = format_spec.get("special", "")

        # Special handling for CMB task which needs to determine SWAN file
        if special == "swan_detection" and task_name == InferenceEnum.CMB and len(input_paths) >= 2:
            base_name1 = os.path.basename(input_paths[0]).split(".")[0]
            base_name2 = os.path.basename(input_paths[1]).split(".")[0]

            swan_base_name = base_name1 if base_name1.startswith("SWAN") else base_name2
            other_base_name = base_name2 if base_name1.startswith("SWAN") else base_name1

            file_name = template.format(
                swan_base_name=swan_base_name, other_base_name=other_base_name, task_name=task_name
            )
            output_files.append(os.path.join(base_output_path, file_name))

        # For formats that need to be applied to each input file
        elif apply_to == "each_input":
            for input_path in input_paths:
                base_name = os.path.basename(input_path).split(".")[0]
                file_name = template.format(base_name=base_name, task_name=task_name)
                output_files.append(os.path.join(base_output_path, file_name))

        # For formats that are applied once
        elif apply_to == "once":
            file_name = template.format(task_name=task_name)
            output_files.append(os.path.join(base_output_path, file_name))

    return output_files


def build_Area(mode, file_dict) -> tuple:
    """
    Build Area model based on input files.
    """
    parser = argparse.ArgumentParser(prog="build_Area")
    args = parser.parse_known_args()[0]
    args.cmb = False
    args.wmh = False
    args.dwi = False
    args.wm_file = False
    args.all = False
    args.depth_number = 5
    setattr(args, mode, True)
    output_path = pathlib.Path(file_dict["output_path"])
    args.intput_file_list = list(map(lambda x: pathlib.Path(x), file_dict["input_path_list"]))
    args.resample_file_list = prepare_output_file_list(args.intput_file_list, "_resample.nii.gz", output_path)
    args.synthseg_file_list = prepare_output_file_list(args.resample_file_list, "_synthseg.nii.gz", output_path)
    args.synthseg33_file_list = prepare_output_file_list(args.resample_file_list, "_synthseg33.nii.gz", output_path)
    args.david_file_list = prepare_output_file_list(args.resample_file_list, "_david.nii.gz", output_path)
    args.wm_file_list = prepare_output_file_list(args.resample_file_list, "_wm.nii.gz", output_path)
    return args, args.intput_file_list


def get_synthseg_args_file(inference_name, file_dict) -> tuple:
    """
    Get synthseg args file.
    """
    output_path = pathlib.Path(file_dict["output_path"])
    match inference_name:
        case InferenceEnum.Area:
            args, file_list = build_Area("wm_file", file_dict)
            return args, file_list
        case InferenceEnum.WMH_PVS:
            args, file_list = build_Area("wmh", file_dict)
            args.wmh_file_list = prepare_output_file_list(args.resample_file_list, "_WMHPVS.nii.gz", output_path)
            return args, file_list
        case InferenceEnum.DWI:
            args, file_list = build_Area("dwi", file_dict)
            args.dwi_file_list = prepare_output_file_list(args.resample_file_list, "_DWI.nii.gz", output_path)
            return args, file_list
        case InferenceEnum.CMB:
            args, file_list = build_Area("cmb", file_dict)
            args.cmb_file_list = prepare_output_file_list(args.resample_file_list, "_CMB.nii.gz", output_path)
            return args, file_list
        case InferenceEnum.Aneurysm:
            args, file_list = build_Area("wm_file", file_dict)
            return args, file_list
        case _:
            return (None, None)


def build_input_post_process(input_paths, model_name, config_path: str = "config.yaml") -> list[pathlib.Path | str]:
    """
    Build input post process using configuration.
    """
    config = load_config(config_path)
    post_process_config = config.get("input_post_process", {}).get(model_name, {})

    if not post_process_config:
        return input_paths

    input_name_list = list(map(lambda x: replace_suffix(os.path.basename(x), ""), input_paths))

    # 獲取匹配條件
    condition = post_process_config.get("condition", {})
    input_match_strs = condition.get("input_match", [])

    # 將字符串條件轉換為枚舉實例的值
    input_match = []
    for match_str in input_match_strs:
        enum_instance = get_enum_by_name(match_str)
        if enum_instance:
            input_match.append(enum_instance.value)
        else:
            input_match.append(match_str)

    if input_match:
        result = np.intersect1d(input_name_list, input_match, return_indices=True)

        # 如果條件匹配，應用指定的轉換
        if result[0].shape[0] == len(input_match):
            add_files = post_process_config.get("add_files", [])

            for file_spec in add_files:
                replace_spec = file_spec.get("replace", {})
                from_pattern = replace_spec.get("from", "")
                to_pattern = replace_spec.get("to", "")

                if from_pattern and to_pattern:
                    for input_path in input_paths:
                        if from_pattern in input_path:
                            new_path = input_path.replace(from_pattern, to_pattern)
                            input_paths.append(new_path)
                            break

    return input_paths


def build_analysis(study_path: pathlib.Path, config_path: str = "config.yaml"):
    """
    Build analysis using configuration.
    """
    mapping_inference = check_study_mapping_inference(study_path, config_path)
    study_id = study_path.name
    model_dict_values = mapping_inference.values()

    for task_dict in model_dict_values:
        tasks = {}
        for model_name, input_paths in task_dict.items():
            input_paths = build_input_post_process(input_paths, model_name, config_path)

            task_output_files = generate_output_files(input_paths, model_name, str(study_path), config_path)
            task = Task(
                intput_path_list=input_paths,
                output_path=str(study_path),
                output_path_list=task_output_files,
            )
            tasks_dump = {model_name: task.model_dump()}
            tasks[model_name] = task

    analyses = Analysis(study_id=study_id, **tasks)
    return analyses


def build_inference_cmd(
    nifti_study_path: pathlib.Path,
    dicom_study_path: pathlib.Path,
) -> InferenceCmd | None:
    """
    Build inference command.
    """
    from code_ai.pipeline import pipelines

    analysis: Analysis = build_analysis(nifti_study_path)
    inference_item_list = []

    for key, value in analysis.model_dump().items():
        if value is not None:
            if key in pipelines:
                task = getattr(analysis, key)
                if key == InferenceEnum.Infarct:
                    basename = os.path.basename(task.input_path_list[1]).split(".")[0]
                else:
                    basename = os.path.basename(task.input_path_list[0]).split(".")[0]

                if dicom_study_path.name == nifti_study_path.name:
                    intput_dicom = dicom_study_path.joinpath(basename)
                else:
                    intput_dicom = dicom_study_path.joinpath(nifti_study_path.name, basename)

                input_dicom_dir = str(intput_dicom)
                cmd_str = pipelines[key].generate_cmd(analysis.study_id, task, input_dicom_dir)
                inference_item = InferenceCmdItem(
                    study_id=analysis.study_id,
                    name=key,
                    cmd_str=cmd_str,
                    input_list=task.input_path_list,
                    output_list=task.output_path_list,
                    input_dicom_dir=str(intput_dicom),
                )
                inference_item_list.append(inference_item)

    return InferenceCmd(cmd_items=inference_item_list)
