#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: sean Ho
"""
import traceback

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: sean Ho
"""
import asyncio
import glob

import aiofiles
import warnings
import pyorthanc

warnings.filterwarnings("ignore")  # 忽略警告输出
import os
import argparse
from pyorthanc import Orthanc,AsyncOrthanc
from dotenv import load_dotenv
load_dotenv()

sem_limit = 128
sem = asyncio.Semaphore(sem_limit)


async def upload_dicom_file(client, file_path):
    # Use the semaphore to limit concurrent operations
    async with sem:
        try:
            async with aiofiles.open(file_path, mode='rb') as f:
                content = await f.read()
                result = await client.post_instances(content)
                return result
        except Exception as e:
            print(f"Error uploading {os.path.basename(file_path)}: {str(e)}")
            traceback.print_exc()
            return {"file": file_path, "error": str(e)}


async def upload_batch(client, file_paths):
    # Process files in batches
    tasks = [upload_dicom_file(client, file_path) for file_path in file_paths]
    return await asyncio.gather(*tasks, return_exceptions=True)


async def main():
    parser = argparse.ArgumentParser(description="處理DICOM-SEG檔案至Orthanc")
    parser.add_argument('--Input', type=str, nargs='+',
                        default='/mnt/d/wsl_ubuntu/pipeline/sean/example_output/10516407_20231215_MR_21210200091/Pred_WMH_A1.dcm',
                        help='DICOM-SEG檔案')
    args = parser.parse_args()
    # upload_dicom_seg()
    UPLOAD_DATA_DICOM_SEG_URL = os.getenv("UPLOAD_DATA_DICOM_SEG_URL")
    print('args.Input',args.Input)

    if isinstance(args.Input, list) and len(args.Input)==1:
        ## Or with authentication:
        # client = Orthanc('http://localhost:8042', username='orthanc', password='orthanc')
        ## Connect to Orthanc server
        input_path = args.Input[0]
        if os.path.isfile(input_path):
            client = Orthanc(UPLOAD_DATA_DICOM_SEG_URL)
            result = pyorthanc.upload(client, input_path)
            print('result', result)
        if os.path.isdir(input_path):
            client = AsyncOrthanc(UPLOAD_DATA_DICOM_SEG_URL,timeout=300)
            dcm_list = glob.glob('{}/**/*.dcm'.format(input_path),recursive=True)
            print('dcm_list',len(dcm_list))
            total_files = len(dcm_list)
            successful_uploads = 0
            failed_uploads = 0
            # Process in batches to maintain better control
            batch_size = 500  # Adjust based on system capabilities
            for i in range(0, total_files, batch_size):
                batch = dcm_list[i:i + batch_size]
                print(
                    f"Processing batch {i // batch_size + 1}/{(total_files + batch_size - 1) // batch_size}: {len(batch)} files")
                results = await upload_batch(client, batch)
                # Count successes and failures
                for r in results:
                    if isinstance(r, dict) and 'error' in r:
                        failed_uploads += 1
                    else:
                        successful_uploads += 1
    elif isinstance(args.Input, list):
        input_list = args.Input
        client = AsyncOrthanc(UPLOAD_DATA_DICOM_SEG_URL)
        tasks = [upload_dicom_file(client, file_path) for file_path in input_list]
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks)
        print('result', results)


# 其意義是「模組名稱」。如果該檔案是被引用，其值會是模組名稱；但若該檔案是(透過命令列)直接執行，其值會是 __main__；。
if __name__ == '__main__':
    asyncio.run(main())