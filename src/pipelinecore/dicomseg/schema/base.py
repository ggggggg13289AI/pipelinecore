import datetime
import os

import numpy as np
from pydantic import BaseModel, ConfigDict, PositiveInt, field_validator

from .schema.enum import ModelTypeEnum, SeriesTypeEnum

# // 目前 Orthanc 自動同步機制的 group 應為 44
GROUP_ID = os.getenv("GROUP_ID",44)


class InstanceRequest(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    # dicom id
    sop_instance_uid :str
    # 應與 ImageOrientationPatient 和 ImagePositionPatient 有關
    projection       :str

    @field_validator('projection', mode='before')
    @classmethod
    def extract_projectionr(cls, value):
        if value is None:
            return '0.0'
        if isinstance(value, (float,int)):
            return str(round(value,6))
        if isinstance(value,str):
            return str(round(float(value),6))
        return '0.0'


class SortedSeriesRequest(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    # dicom
    series_instance_uid :str
    instance            : list[InstanceRequest]


class SortedRequest(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    study_instance_uid: str
    series            : list[SortedSeriesRequest]


class MaskInstanceRequest(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    mask_index   : PositiveInt = 1
    mask_name    : str
    # diameter,type,location,sub_location,checked 在第一次上傳後寫入 db , 之後可從前端介面手動更改
    diameter     : str
    type         : str = 'saccular'
    location     : str = 'M'
    sub_location : str = '2'
    prob_max     : str
    # bool , 0 或 1
    checked      : str = '1'
    # bool , 0 或 1
    is_ai        : str = '1'

    seg_series_instance_uid : str
    seg_sop_instance_uid    : str

    # 對應到 dicom image Key porint uid
    dicom_sop_instance_uid  : str

    main_seg_slice          : int = 1
    is_main_seg             : int = 0

    @field_validator('diameter', mode='before')
    @classmethod
    def extract_diameter_number(cls, value):
        if value is None:
            return '0.0'
        if isinstance(value, np.ndarray):
            return str(value.mean().item())
        # 如果是從DICOM獲取的值
        if isinstance(value, (float, int)):
            return str(value)
        return value

    @field_validator('prob_max', mode='before')
    @classmethod
    def extract_prob_max(cls, value):
        if value is None:
            return '1.0'
        if isinstance(value, (float, int)):
            return str(round(value, 6))
        return '1.0'


class MaskSeriesRequest(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    # dicom image uid
    series_instance_uid : str
    # 目前僅 Aneu 的 TOF_MRA:1 , Pitch: 2 , Yaw: 3 , 未來有其他 Series 再加
    series_type         : str
    instances           : list[MaskInstanceRequest]
    model_type          : str


    @field_validator('series_type', mode='before')
    @classmethod
    def extract_series_type(cls, value):
        if value is None:
            return '1'
        if isinstance(value, str):
            series_type_enum_list: list[SeriesTypeEnum] = SeriesTypeEnum.to_list()
            for series_type_enum in series_type_enum_list:
                if value == series_type_enum.name:
                    return series_type_enum.value
        return value


class MaskRequest(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    study_instance_uid : str
    group_id           : PositiveInt = 44
    series             : list[MaskSeriesRequest]


class StudySeriesRequest(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    # dicom
    series_type         : str
    series_instance_uid : str
    resolution_x        : PositiveInt = 512
    resolution_y        : PositiveInt = 512


    @field_validator('series_type', mode='before')
    @classmethod
    def extract_series_type(cls, value):
        if value is None:
            return '1'
        if isinstance(value, str):
            series_type_enum_list: list[SeriesTypeEnum] = SeriesTypeEnum.to_list()
            for series_type_enum in series_type_enum_list:
                if value == series_type_enum.name:
                    return series_type_enum.value
        return value


class StudyModelRequest(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    series_type : list[str]
    model_type  : str
    lession     : str = "0"
    status      : str = "1"
    report      : str = ""

    @field_validator('lession', mode='before')
    @classmethod
    def extract_lession(cls, value):
        if value is None:
            return '0'

        return str(value)

    @field_validator('model_type', mode='before')
    @classmethod
    def extract_model_type(cls, value):
        if value is None:
            return '1'
        if isinstance(value, str):
            model_type_enum_list: list[ModelTypeEnum] = ModelTypeEnum.to_list()
            for model_type_enum in model_type_enum_list:
                if value == model_type_enum.name:
                    return model_type_enum.value
        return value


    @field_validator('series_type', mode='before')
    @classmethod
    def extract_series_type(cls, value):
        if value is None:
            return ['1']
        if isinstance(value, str):
            series_type_enum_list: list[SeriesTypeEnum] = SeriesTypeEnum.to_list()
            for series_type_enum in series_type_enum_list:
                if value == series_type_enum.name:
                    return [series_type_enum.value]
        return [value]



class StudyRequest(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    group_id           :int = GROUP_ID
    # "2017-12-25"
    study_date         :datetime.date
    # M F
    gender             :str
    # 99 56
    age                :int
    # Brain MRI
    study_name         :str
    # Mock
    patient_name       :str
    study_instance_uid :str
    patient_id         :str
    series             :list[StudySeriesRequest]
    model              :list[StudyModelRequest]

    @field_validator('patient_name', mode='before')
    @classmethod
    def extract_patient_name_number(cls, value):
        if value is None:
            return ''
        # 如果是從DICOM獲取的值
        if isinstance(value, str):

            return value
        return str(value)

    @field_validator('study_date', mode='before')
    @classmethod
    def parse_date(cls, value):
        try:
            if isinstance(value, str) and len(value) == 8:
                return datetime.date(int(value[:4]), int(value[4:6]), int(value[6:8]))
            return datetime.datetime.strptime(value,'%Y-%m-%d').date()
        except:
            return datetime.datetime.now().date()


    @field_validator('age', mode='before')
    @classmethod
    def extract_age_number(cls, value):
        if value is None:
            return None

        # 如果是從DICOM獲取的值
        if isinstance(value, str):
            # 移除所有非數字字符
            digits_only = ''.join(c for c in value if c.isdigit())
            if digits_only:
                return int(digits_only)
        return value


class AITeamRequest(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    study   : StudyRequest | None  = None
    sorted  : SortedRequest | None = None
    mask    : MaskRequest | None   = None
