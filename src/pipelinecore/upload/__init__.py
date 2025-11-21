import argparse
import json
import os
import pathlib
import subprocess
from typing import Optional

from code_ai import PYTHON3
from code_ai.utils.inference import InferenceEnum, Task


def upload_dicom_seg(input_dicom_seg_folder: str, input_nifti: str):
    input_dicom_seg_folder_path = pathlib.Path(input_dicom_seg_folder)
    input_nifti_path = pathlib.Path(input_nifti)
    dicom_seg_base_name = input_nifti_path.name.split('.')[0]
    file_list = sorted(input_dicom_seg_folder_path.rglob(dicom_seg_base_name + '*.dcm'))
    file_str_list = list(map(lambda x: str(x), file_list))
    cmd_str = ('export PYTHONPATH={} && '
               '{} code_ai/pipeline/upload/orthanc_dicom.py '
               '--Input {} '.format(pathlib.Path(__file__).parent.parent.parent.absolute(),
                                    PYTHON3,
                                    ' '.join(file_str_list)
                                    )
               )
    print('upload_dicom_seg', cmd_str)

    process = subprocess.Popen(args=cmd_str, shell=True,
                               # cwd='{}'.format(pathlib.Path(__file__).parent.parent.absolute()),
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    return stdout, stderr


def upload_json(ID: str, mode: InferenceEnum) -> object:
    path_process = os.getenv("PATH_PROCESS")
    cmd_json_path_path = os.path.join(path_process, 'Deep_cmd_tools', '{}_cmd.json'.format(ID))
    if os.path.exists(cmd_json_path_path):
        with open(cmd_json_path_path, 'r') as f:
            data_json = json.load(f)
        cmd_data = next(filter(lambda x: x['study_id'] == ID and x['name'] == mode, data_json))
        if cmd_data is not None:
            file_list = cmd_data['output_list']
            nii_file_list = list(filter(lambda x: str(x).endswith('nii.gz'), file_list))
            platform_json_list = list(map(lambda x: str(x).replace('.nii.gz', '_platform_json.json'), nii_file_list))

            platform_json_list = list(filter(lambda x: os.path.exists(x), platform_json_list))
            for platform_json in platform_json_list:
                cmd_str = ('export PYTHONPATH={} && '
                           '{} code_ai/pipeline/upload/platform_json.py '
                           '--Input {} '.format(pathlib.Path(__file__).parent.parent.parent.parent.absolute(),
                                                PYTHON3,
                                                platform_json
                                                )
                           )
                print('upload_json', cmd_str)

                process = subprocess.Popen(args=cmd_str, shell=True,
                                           # cwd='{}'.format(pathlib.Path(__file__).parent.parent.absolute()),
                                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                stdout, stderr = process.communicate()
                print(stdout, stderr)
        else:
            raise ValueError('No found {} for {}'.format(mode, ID))
    else:
        raise FileNotFoundError(cmd_json_path_path)
