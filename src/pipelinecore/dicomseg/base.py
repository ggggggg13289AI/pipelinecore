# -*- coding: utf-8 -*-
"""
DICOM-SEG Generator

@author: sean
"""
import abc
import os
from typing import Dict, List, Any, Union, Optional, TypeVar, Generic
from typing_extensions import Self
from pydantic import BaseModel
from pydicom import FileDataset
from pydicom.dicomdir import DicomDir
from .schema.enum import SeriesTypeEnum, ModelTypeEnum
from .schema.base import SortedRequest,SortedSeriesRequest
from .schema.base import StudyRequest,StudySeriesRequest,StudyModelRequest
from .schema.base import MaskRequest,MaskSeriesRequest,MaskInstanceRequest, AITeamRequest
from .schema.typing import AITeamT
from .schema.typing import StudyT,StudySeriesT,StudyModelT
from .schema.typing import SortedT,SortedSeriesT
from .schema.typing import MaskT,MaskSeriesT, MaskInstanceT

GROUP_ID = os.getenv("GROUP_ID_CMB", 44)

T = TypeVar('T', bound=BaseModel)


class PlatformJSONBuilder(Generic[T], metaclass=abc.ABCMeta):
    model_class: type[T]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._ai_team_dict: Dict[str, Any] = {}
        self.reset()

    def reset(self):
        self._study_request: Optional[StudyRequest] = None
        self._sorted_request: Optional[SortedRequest] = None
        self._mask_request: Optional[MaskRequest] = None
        self._ai_team_dict: Dict[str, Any] = {}


    @abc.abstractmethod
    def build_mask_instance(self, source_images: List[Union[FileDataset, DicomDir]],
                            dcm_seg: Union[FileDataset, DicomDir],
                            *args, **kwargs) -> Union[Dict[str,Any],Any]:
        pass

    @abc.abstractmethod
    def build_mask_series(self,
                          source_images: List[Union[FileDataset, DicomDir]],
                          reslut_list: List[Dict[str, Any]],
                          pred_json_list: List[Dict[str, Any]],
                          # dcm_seg: Union[FileDataset, DicomDir],
                          *args, **kwargs) -> Union[Dict[str,Any],Any]:
        pass

    @abc.abstractmethod
    def set_mask(self,
                 source_images: List[Union[FileDataset, DicomDir]],
                 reslut_list: List[Dict[str, Any]],
                 pred_json_list: List[Dict[str, Any]],
                 group_id: int) -> Self:
        pass

    def set_sorted(self, source_images: List[Union[FileDataset, DicomDir]]) -> Self:
        instance_list = []
        series_list = []
        series_dict = dict()
        sorted_dict = dict()

        # Process each DICOM image
        for index, dicom_ds in enumerate(source_images):
            instance_dict = dict()
            # Extract SOP Instance UID and Image Position Patient
            sop_instance_uid = dicom_ds.get((0x0008, 0x0018)).value
            image_osition_patient = str(dicom_ds.get((0x0020, 0x0032)).value[-1])
            # Format: (0020,0032) Image Position Patient -14.7431\-143.248\98.5056

            # Update instance dictionary
            instance_dict.update(dict(
                sop_instance_uid=sop_instance_uid,
                projection=str(image_osition_patient)
            ))
            instance_list.append(instance_dict)

            # If this is the first image, add study and series information
            if index == 0:
                study_instance_uid = dicom_ds.get((0x0020, 0x000D)).value
                series_instance_uid = dicom_ds.get((0x0020, 0x000E)).value
                series_dict.update(dict(
                    series_instance_uid=series_instance_uid,
                    instance=instance_list
                ))
                series_list.append(series_dict)
                sorted_dict.update({'study_instance_uid': study_instance_uid})

        # Add the series list to the sorted dictionary
        sorted_dict.update({'series': series_list})

        # Validate and return the sorted request
        self._sorted_request = SortedRequest.model_validate(sorted_dict)
        return self

    def build_study_basic_info(self, source_images: List[Union[FileDataset, DicomDir]],
                               group_id: int = GROUP_ID) -> Dict[str,Any]:
        dicom_ds = source_images[0]
        # Extract study metadata from the first DICOM image
        study_instance_uid = dicom_ds.get((0x0020, 0x000D)).value
        study_date = dicom_ds.get((0x0008, 0x0020)).value
        gender = dicom_ds.get((0x0010, 0x0040)).value
        age = dicom_ds.get((0x0010, 0x1010)).value
        study_name = dicom_ds.get((0x0008, 0x1030)).value
        patient_name = dicom_ds.get((0x0010, 0x0010)).value
        # resolution_x = dicom_ds.get((0x0028, 0x0010)).value
        # resolution_y = dicom_ds.get((0x0028, 0x0011)).value
        patient_id = dicom_ds.get((0x0010, 0x0020)).value

        # Update study dictionary with metadata
        return dict(
            group_id=group_id,
            study_instance_uid=study_instance_uid,
            study_date=study_date,
            gender=gender,
            age=age,
            study_name=study_name,
            patient_name=patient_name,
            patient_id=patient_id, )

    @abc.abstractmethod
    def build_study_series(self, source_images: List[Union[FileDataset, DicomDir]],
                           *args, **kwargs) -> Self:
        pass

    @abc.abstractmethod
    def build_study_model(self, pred_json_list: List[Dict[str, Any]],
                          *args, **kwargs) -> Self:
        pass


    @abc.abstractmethod
    def set_study(self, source_images: List[Union[FileDataset, DicomDir]],
                  result_list: List[Dict[str, Any]],
                  pred_json_list: List[Dict[str, Any]],
                  group_id: int) -> Self:
        pass

    def build(self) -> T:
        self._ai_team_dict.update({"study":self._study_request if self._study_request else None,
                                   "sorted": self._sorted_request if self._sorted_request else None,
                                   "mask": self._mask_request if self._mask_request else None,})
        instance = self.model_class.model_validate(self._ai_team_dict)
        self.reset()
        return instance



class ReviewPlatformJSONBuilder(Generic[AITeamT,
                                        StudyT,StudySeriesT,StudyModelT,
                                        SortedT,SortedSeriesT,
                                        MaskT,MaskSeriesT,MaskInstanceT],
                                metaclass=abc.ABCMeta):
    AITeamClass       : type[AITeamT]
    StudyClass        : type[StudyT]
    StudySeriesClass  : type[StudySeriesT]
    StudyModelClass   : type[StudyModelT]
    SortedClass       : type[SortedT]
    SortedSeriesClass : type[SortedSeriesT]
    MaskClass         : type[MaskT]
    MaskSeriesClass   : type[MaskSeriesT]
    MaskInstanceClass : type[MaskInstanceT]
    model_type        : ModelTypeEnum


    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._ai_team_dict   : Dict[str, Any] = {}
        self._series_dict    : Optional[Dict[Union[SeriesTypeEnum, str], List[Union[FileDataset, DicomDir]]]] = {}
        self._study_request  : Optional[StudyRequest] = None
        self._sorted_request : Optional[SortedRequest] = None
        self._mask_request   : Optional[MaskRequest] = None
        self._group_id       : Optional[int] = None
        self.reset()

    def reset(self):
        self._study_request: Optional[StudyRequest] = None
        self._sorted_request: Optional[SortedRequest] = None
        self._mask_request: Optional[MaskRequest] = None
        self._ai_team_dict: Dict[str, Any] = {}
        self._group_id: Optional[int] = None


    @property
    def group_id(self):
        return self._group_id

    @group_id.setter
    def group_id(self,group_id:int):
        self._group_id = group_id

    def set_group_id(self,group_id:int):
        self._group_id = group_id
        return self


    def set_series_type(self, series_type:Union[SeriesTypeEnum,str],
                              source_images: List[Union[FileDataset, DicomDir]],
                              ) -> Self :
        self._series_dict.update({series_type: source_images})
        return self


    def get_mask_series(self,source_images:List[Union[FileDataset, DicomDir]],
                             series_type: SeriesTypeEnum,
                             dicom_seg_result,
                             pred_json,
                            *args, **kwargs) -> "MaskSeriesClass":
        """
            dicom_seg_result : {"series_type": {}, "data":{} }
            dicom_seg_result : {"series_type": {}, "data":{} }
        """
        series_instance_uid = source_images[0].get((0x0020, 0x000E)).value
        mask_series_dict = {
            'series_instance_uid': series_instance_uid,
            'series_type': series_type,
            'model_type': self.model_type
        }
        mask_instance = self.get_mask_instance(source_images = source_images,
                                               series_type = series_type,
                                               dicom_seg_result = dicom_seg_result,
                                               pred_json = pred_json,
                                               *args, **kwargs)
        mask_series_dict.update({'instances': mask_instance})


        return self.MaskSeriesClass.model_validate(mask_series_dict)

    @abc.abstractmethod
    def get_mask_instance(self,source_images:List[Union[FileDataset, DicomDir]],
                             series_type: SeriesTypeEnum,
                             dicom_seg_result,
                             pred_json,
                            *args, **kwargs) -> List["MaskInstanceT"]:
        pass



    @abc.abstractmethod
    def get_study_model(self,series_type:SeriesTypeEnum,
                             pred_data : Dict[str,Any],
                        *args, **kwargs) -> "StudyModelClass":
        pass


    def build_study(self, *args, **kwargs) -> Self:
        study_series_list = []
        study_model_list = []
        study_dict = {}
        pred_json_list = kwargs.get('pred_json_list')
        for index, (series_name, series_source_images) in enumerate(self._series_dict.items()):
            if index==0:
                study_dict.update(self.get_study_basic_info(source_images=series_source_images,
                                                            group_id=self.group_id))
            study_series = self.get_study_series(source_images=series_source_images, series_type=series_name)
            study_series_list.append(study_series)
            study_model = self.get_study_model(series_type=series_name,pred_data=pred_json_list[index],
                                               *args, **kwargs)
            study_model_list.append(study_model)

        study_dict.update({'series': study_series_list,
                           'model': study_model_list})
        self._study_request = self.StudyClass.model_validate(study_dict)
        return self


    def get_study_basic_info(self, source_images: List[Union[FileDataset, DicomDir]],
                             group_id: int = GROUP_ID) -> Dict[str, Any]:
        dicom_ds = source_images[0]
        # Extract study metadata from the first DICOM image
        study_instance_uid = dicom_ds.get((0x0020, 0x000D)).value
        study_date = dicom_ds.get((0x0008, 0x0020)).value
        gender = dicom_ds.get((0x0010, 0x0040)).value
        age = dicom_ds.get((0x0010, 0x1010)).value
        study_name = dicom_ds.get((0x0008, 0x1030)).value
        patient_name = dicom_ds.get((0x0010, 0x0010)).value
        # resolution_x = dicom_ds.get((0x0028, 0x0010)).value
        # resolution_y = dicom_ds.get((0x0028, 0x0011)).value
        patient_id = dicom_ds.get((0x0010, 0x0020)).value

        # Update study dictionary with metadata
        return dict(
            group_id=group_id,
            study_instance_uid=study_instance_uid,
            study_date=study_date,
            gender=gender,
            age=age,
            study_name=study_name,
            patient_name=patient_name,
            patient_id=patient_id, )

    def get_study_series(self, source_images: List[Union[FileDataset, DicomDir]],
                         series_type: SeriesTypeEnum) -> "StudySeriesClass":
        series_instance_uid_dict = {}
        for index, dicom_ds in enumerate(source_images):
            series_instance_uid = dicom_ds.get((0x0020, 0x000E)).value
            if series_instance_uid_dict.get(series_instance_uid) is None:

                # 取得影像解析度
                resolution_x = dicom_ds.get((0x0028, 0x0010)).value
                resolution_y = dicom_ds.get((0x0028, 0x0011)).value

                series_instance_uid_dict.update({series_instance_uid: {
                    'series_type': series_type.name,
                    'resolution_x': resolution_x,
                    'resolution_y': resolution_y
                }})
            else:
                break
        instance_dict = {}
        for series_instance_uid, series_info in series_instance_uid_dict.items():
            # Update instance dictionary
            instance_dict.update(dict(
                series_type=series_info['series_type'],
                series_instance_uid=series_instance_uid,
                resolution_x=series_info['resolution_x'],
                resolution_y=series_info['resolution_y']
            ))
        study_series = self.StudySeriesClass.model_validate(instance_dict)
        return study_series

    def build_sorted(self,) -> Self:
        sorted_dict = dict()
        series_list = []
        for series_name, series_source_images in self._series_dict.items():
            instance_list = []
            series_dict = dict()
            # Process each DICOM image
            for index, dicom_ds in enumerate(series_source_images):
                instance_dict = dict()
                # Extract SOP Instance UID and Image Position Patient
                sop_instance_uid = dicom_ds.get((0x0008, 0x0018)).value
                image_osition_patient = str(dicom_ds.get((0x0020, 0x0032)).value[-1])
                # Format: (0020,0032) Image Position Patient -14.7431\-143.248\98.5056

                # Update instance dictionary
                instance_dict.update(dict(
                    sop_instance_uid=sop_instance_uid,
                    projection=str(image_osition_patient)
                ))
                instance_list.append(instance_dict)

                # If this is the first image, add study and series information
                if index == 0:
                    study_instance_uid = dicom_ds.get((0x0020, 0x000D)).value
                    series_instance_uid = dicom_ds.get((0x0020, 0x000E)).value
                    series_dict.update(dict(
                        series_instance_uid=series_instance_uid,
                        instance=instance_list
                    ))
                    # series_list.append(series_dict)
                    sorted_dict.update({'study_instance_uid': study_instance_uid})
            series_list.append(self.SortedSeriesClass.model_validate(series_dict))


            # Add the series list to the sorted dictionary
        sorted_dict.update({'series': series_list})

        # Validate and return the sorted request
        self._sorted_request = self.SortedClass.model_validate(sorted_dict)
        return self

    def build_mask(self,dicom_seg_result_list,pred_json_list,*args, **kwargs) -> Self:
        mask_dict = {}
        if all((self._series_dict is not None,self.group_id is not None)):
            for index, (series_name, series_source_images) in enumerate(self._series_dict.items()):
                if index == 0:
                    study_instance_uid = series_source_images[0].get((0x0020, 0x000D)).value
                    mask_dict.update( {
                        'study_instance_uid': study_instance_uid,
                        'group_id': self.group_id
                    })
                    break
        else:
            raise ValueError('self._series_dict or self.group_id is None')
        series_list = []
        for index, (series_name, series_source_images) in enumerate(self._series_dict.items()):
            mask_series = self.get_mask_series(source_images    = series_source_images,
                                               series_type      = series_name,
                                               dicom_seg_result = dicom_seg_result_list[index],
                                               pred_json        = pred_json_list[index],
                                               *args, **kwargs)
            series_list.append(mask_series)

        mask_dict.update({'series': series_list})
        self._mask_request = self.MaskClass.model_validate(mask_dict)
        return self

    def build(self) -> T:
        self._ai_team_dict.update({"study":self._study_request if self._study_request else None,
                                   "sorted": self._sorted_request if self._sorted_request else None,
                                   "mask": self._mask_request if self._mask_request else None,})
        instance = self.AITeamClass.model_validate(self._ai_team_dict)
        self.reset()
        return instance


class ReviewBasePlatformJSONBuilder(ReviewPlatformJSONBuilder[AITeamRequest,
                                                              StudyRequest,StudySeriesRequest,StudyModelRequest,
                                                              SortedRequest,SortedSeriesRequest,
                                                              MaskRequest,MaskSeriesRequest,MaskInstanceRequest],
                                    metaclass=abc.ABCMeta):
    AITeamClass       = AITeamRequest
    StudyClass        = StudyRequest
    StudyModelClass   = StudyModelRequest
    StudySeriesClass  = StudySeriesRequest
    SortedClass       = SortedRequest
    SortedSeriesClass = SortedSeriesRequest
    MaskClass         = MaskRequest
    MaskSeriesClass   = MaskSeriesRequest
    MaskInstanceClass = MaskInstanceRequest
    pass