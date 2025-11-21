
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: sean Ho
"""
import argparse
import os
import warnings

import httpx
import orjson
from code_ai import load_dotenv

warnings.filterwarnings("ignore")  # 忽略警告输出
load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="處理 AI predict 檔案至 web server")
    parser.add_argument('--Inputs', type=str, nargs='+',
                        default=' ',
                        help='json檔案')
    args = parser.parse_args()

    UPLOAD_DATA_JSON_URL = os.getenv("UPLOAD_DATA_JSON_URL")

    client = httpx.Client()
    if isinstance(args.Inputs, str):
        with open(args.Inputs, 'rb') as f:
            data = orjson.loads(f.read())
        print('platform_json GROUP_ID', data['study']['group_id'])
        with client:
            response = client.post(UPLOAD_DATA_JSON_URL, json=data)
            print(f"Uploaded single file: Status {response.status_code}")

    if isinstance(args.Inputs, list):
        input_list = args.Inputs
        with client:
            for inputs in input_list:
                with open(inputs, 'rb') as f:
                    data = orjson.loads(f.read())
                print('platform_json GROUP_ID', data['study']['group_id'])
                response = client.post(UPLOAD_DATA_JSON_URL, json=data)
                print(f"Uploaded single file: Status {response.status_code}")
                print(response.text)


if __name__ == '__main__':
    main()
