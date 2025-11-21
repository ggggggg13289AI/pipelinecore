import argparse
import os
import pathlib
from typing import List, Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd

from code_ai.utils import replace_suffix
from .schema import InferenceCmd,InferenceCmdItem,InferenceEnum # MODEL_MAPPING_SERIES_DICT,
from .schema import Analysis,Task
from .config import MODEL_MAPPING_SERIES_DICT


def get_file_list(input_path: pathlib.Path, suffixes: str, filter_name=None) -> List[pathlib.Path]:
    if any(suffix in input_path.suffixes for suffix in suffixes):
        file_list = [input_path]
    else:
        file_list = sorted(list(input_path.rglob('*.nii*')))
    if filter_name:
        file_list = [f for f in file_list if filter_name in f.name]
    return file_list


def prepare_output_file_list(file_list: List[pathlib.Path],
                             suffix: str,
                             output_dir: Optional[pathlib.Path] = None) -> List[pathlib.Path]:
    return [
        output_dir.joinpath(replace_suffix(f'{x.name}', suffix)) if output_dir else x.parent.joinpath(
            replace_suffix(x.name, suffix)) for x in file_list]
    # return [
    #     output_dir.joinpath(x.parent.name, replace_suffix(f'{x.name}', suffix)) if output_dir else x.parent.joinpath(
    #         replace_suffix(x.name, suffix)) for x in file_list]


def check_study_mapping_inference(study_path: pathlib.Path) -> Dict[str, Dict[str, str]]:
    file_list = sorted(study_path.iterdir())

    if any(filter(lambda x: x.name.endswith('nii.gz') or x.name.endswith('nii'), file_list)):
        df_file = pd.DataFrame(file_list, columns=['file_path'])
        df_file['file_name'] = df_file['file_path'].map(lambda x: x.name.replace('.nii.gz', ''))
        model_mapping_dict = {}
        for model_name, model_mapping_series_list in MODEL_MAPPING_SERIES_DICT.items():

            for mapping_series in model_mapping_series_list:
                mapping_series_str = list(map(lambda x: x.value, mapping_series))
                result = np.intersect1d(df_file['file_name'], mapping_series_str, return_indices=True)

                if result[0].shape[0] >= len(mapping_series_str):
                    df_result = df_file.iloc()[result[1]]
                    file_path = list(map(lambda x: str(x), df_result['file_path'].to_list()))
                    model_mapping_dict.update({model_name.value: file_path})
                    break
        return {study_path.name: model_mapping_dict}



def generate_output_files(input_paths: List[str], task_name: str, base_output_path: str) -> List[str]:
    """
    Generate output file names based on input paths and task names.
    """
    output_files = []
    match task_name:
        case InferenceEnum.Aneurysm:
            output_files.append(os.path.join(base_output_path, f"Pred_Aneurysm.nii.gz"))
            output_files.append(os.path.join(base_output_path, f"Prob_Aneurysm.nii.gz"))
            output_files.append(os.path.join(base_output_path, f"Pred_Aneurysm_Vessel.nii.gz"))
            output_files.append(os.path.join(base_output_path, f"Pred_Aneurysm.json"))
            output_files.append(os.path.join(base_output_path, f"Pred_Aneurysm_Vessel16.nii.gz"))
            for input_path in input_paths:
                base_name = os.path.basename(input_path).split('.')[0]
                output_file = os.path.join(base_output_path, f"synthseg_{base_name}_original_synthseg33.nii.gz")
                output_files.append(output_file)

        case InferenceEnum.WMH_PVS:
            for input_path in input_paths:
                base_name = os.path.basename(input_path).split('.')[0]
                output_file = os.path.join(base_output_path, f"synthseg_{base_name}_original_{task_name}.nii.gz")
                output_files.append(output_file)
        # synthseg_DWI0_original_DWI.nii.gz
        case InferenceEnum.DWI:
            output_files.append(os.path.join(base_output_path, f"synthseg_DWI0_original_{task_name}.nii.gz"))
        case InferenceEnum.CMB:
            base_name1 = os.path.basename(input_paths[0]).split('.')[0]
            base_name2 = os.path.basename(input_paths[1]).split('.')[0]
            dirname = os.path.dirname(input_paths[0])
            if base_name1.startswith('SWAN'):
                # synthseg_SWAN_original_CMB_from_synthseg_T1FLAIR_AXI_original_CMB.nii.gz
                output_files.append(os.path.join(base_output_path, (f"synthseg_{base_name1}_original_{task_name}_from_"
                                                                    f"synthseg_{base_name2}_original_{task_name}.nii.gz"
                                                                    )
                                                 ))
            else:
                output_files.append(os.path.join(base_output_path, (f"synthseg_{base_name2}_original_{task_name}_from_"
                                                                    f"synthseg_{base_name1}_original_{task_name}.nii.gz"
                                                                    )
                                                 ))
            output_files.append(os.path.join(base_output_path, f"Pred_CMB.nii.gz"))
            output_files.append(os.path.join(base_output_path, f"Pred_CMB.json"))
        case InferenceEnum.Area:
            for input_path in input_paths:
                base_name = os.path.basename(input_path).split('.')[0]
                output_files.append(os.path.join(base_output_path, f"synthseg_{base_name}_original_synthseg33.nii.gz"))
                output_files.append(os.path.join(base_output_path, f"synthseg_{base_name}_original_synthseg.nii.gz"))
        case InferenceEnum.Infarct:
            output_files.append(os.path.join(base_output_path, f"Pred_Infarct.nii.gz"))
            output_files.append(os.path.join(base_output_path, f"Pred_Infarct_ADCth.nii.gz"))
            output_files.append(os.path.join(base_output_path, f"Pred_Infarct_synthseg.nii.gz"))
            output_files.append(os.path.join(base_output_path, f"Pred_Infarct.json"))
        case InferenceEnum.WMH:
            output_files.append(os.path.join(base_output_path, f"Pred_WMH.nii.gz"))
            output_files.append(os.path.join(base_output_path, f"Pred_WMH_synthseg.nii.gz"))
            output_files.append(os.path.join(base_output_path, f"Pred_WMH.json"))

        case _:
            pass
    return output_files


def build_Area(mode, file_dict) -> Tuple:
    """
    Build Area model based on input files.

    """
    parser = argparse.ArgumentParser(prog='build_Area')
    args = parser.parse_known_args()[0]
    args.cmb = False
    args.wmh = False
    args.dwi = False
    args.wm_file = False
    args.all = False
    args.depth_number = 5
    setattr(args, mode, True)
    output_path = pathlib.Path(file_dict['output_path'])
    args.intput_file_list = list(map(lambda x: pathlib.Path(x), file_dict['input_path_list']))
    args.resample_file_list = prepare_output_file_list(args.intput_file_list, '_resample.nii.gz', output_path)
    args.synthseg_file_list = prepare_output_file_list(args.resample_file_list, '_synthseg.nii.gz', output_path)
    args.synthseg33_file_list = prepare_output_file_list(args.resample_file_list, '_synthseg33.nii.gz', output_path)
    args.david_file_list = prepare_output_file_list(args.resample_file_list, '_david.nii.gz', output_path)
    args.wm_file_list = prepare_output_file_list(args.resample_file_list, '_wm.nii.gz', output_path)
    return args, args.intput_file_list


def get_synthseg_args_file(inference_name, file_dict) -> Tuple:
    """
        args
        args.cmb = False
        args.wmh = False
        args.dwi = False
        args.wm_file = False
        args.to_original_mode
        args.all = False
        args.depth_number = 5
        args.intput_file_list
        args.resample_file_list
        args.synthseg_file_list
        args.synthseg33_file_list
        args.david_file_list
        args.wm_file_list
    """
    output_path = pathlib.Path(file_dict['output_path'])
    match inference_name:
        case InferenceEnum.WMH_PVS:
            args, file_list = build_Area('wmh', file_dict)
            args.wmh_file_list = prepare_output_file_list(args.resample_file_list, '_WMHPVS.nii.gz', output_path)
            return args, file_list
        case InferenceEnum.WMH_PVS:
            args, file_list = build_Area('wmh', file_dict)
            args.wmh_file_list = prepare_output_file_list(args.resample_file_list, '_WMHPVS.nii.gz', output_path)
            return args, file_list
        case InferenceEnum.DWI:
            args, file_list = build_Area('dwi', file_dict)
            args.dwi_file_list = prepare_output_file_list(args.resample_file_list, '_DWI.nii.gz', output_path)
            return args, file_list
        case InferenceEnum.CMB:
            args, file_list = build_Area('cmb', file_dict)
            args.cmb_file_list = prepare_output_file_list(args.resample_file_list, '_CMB.nii.gz', output_path)
            return args, file_list
        case InferenceEnum.Area:
            args, file_list = build_Area('wm_file', file_dict)
            return args, file_list
        case InferenceEnum.Aneurysm:
            args, file_list = build_Area('wm_file', file_dict)
            return args, file_list
        case _:
            return (None, None)


def build_input_post_process(input_paths,model_name) -> List[Union[pathlib.Path, str]]:
    input_name_list = list(map(lambda x: replace_suffix(os.path.basename(x), ''), input_paths))
    match model_name:
        case InferenceEnum.Infarct:
            for mapping_series in MODEL_MAPPING_SERIES_DICT[InferenceEnum.Infarct]:
                mapping_series_str = list(map(lambda x: x.value, mapping_series))
                result = np.intersect1d(input_name_list, mapping_series_str, return_indices=True)
                if result[0].shape[0] == 3:
                    input_paths.append(input_paths[0].replace('ADC.nii.gz', 'synthseg_DWI0_original_DWI.nii.gz'))
        case InferenceEnum.WMH:
            for mapping_series in MODEL_MAPPING_SERIES_DICT[InferenceEnum.WMH]:
                mapping_series_str = list(map(lambda x: x.value, mapping_series))
                result = np.intersect1d(input_name_list, mapping_series_str, return_indices=True)
                if result[0].shape[0] == 1:
                    input_paths.append(input_paths[0].replace('T2FLAIR_AXI.nii.gz',
                                                              'synthseg_T2FLAIR_AXI_original_synthseg5.nii.gz'))
                    input_paths.append(input_paths[0].replace('T2FLAIR_AXI.nii.gz',
                                                              'synthseg_T2FLAIR_AXI_original_WMH_PVS.nii.gz'))
        case _:
            pass
    return input_paths


def build_analysis(study_path: pathlib.Path):
    mapping_inference = check_study_mapping_inference(study_path)
    study_id = study_path.name
    model_dict_values = mapping_inference.values()
    for task_dict in model_dict_values:
        tasks = {}
        for model_name, input_paths in task_dict.items():
            input_paths = build_input_post_process(input_paths,model_name)

            task_output_files = generate_output_files(input_paths,
                                                      model_name,
                                                      str(study_path))
            task = Task(intput_path_list=input_paths,
                        output_path=str(study_path),
                        output_path_list=task_output_files, )
            tasks_dump = {model_name: task.model_dump()}
            tasks[model_name] = task

    analyses = Analysis(study_id=study_id, **tasks)
    return analyses


def build_inference_cmd(nifti_study_path: pathlib.Path,
                        dicom_study_path: pathlib.Path,) -> Optional[InferenceCmd]:  #-> Optional[List[Tuple[str,str]]]:
    from code_ai.pipeline import pipelines
    analysis: Analysis = build_analysis(nifti_study_path)
    # 使用管道配置
    inference_item_list = []
    for key, value in analysis.model_dump().items():
        if value is not None:
            if key in pipelines:
                task = getattr(analysis, key)
                if key == InferenceEnum.Infarct:
                    basename = os.path.basename(task.input_path_list[1]).split('.')[0]
                else:
                    basename = os.path.basename(task.input_path_list[0]).split('.')[0]

                if dicom_study_path.name == nifti_study_path.name :
                    intput_dicom = dicom_study_path.joinpath(basename)
                else:
                    intput_dicom = dicom_study_path.joinpath(nifti_study_path.name,basename)
                input_dicom_dir = str(intput_dicom)
                # print('dicom_study_path',dicom_study_path)
                # print('nifti_study_path', nifti_study_path)
                # print('input_dicom_dir',input_dicom_dir)
                cmd_str = pipelines[key].generate_cmd(analysis.study_id, task,input_dicom_dir)
                inference_item = InferenceCmdItem(study_id = analysis.study_id, name=key,
                                                  cmd_str=cmd_str,
                                                  input_list=task.input_path_list,
                                                  output_list=task.output_path_list,
                                                  input_dicom_dir = str(intput_dicom)
                                                  )
                # (key, cmd_str,analysis.Infarct.input_path_list,analysis.Infarct.output_path_list)
                inference_item_list.append(inference_item)

    return InferenceCmd(cmd_items=inference_item_list)
