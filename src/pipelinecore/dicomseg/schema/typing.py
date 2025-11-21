
from typing import TypeVar
from code_ai.pipeline.dicomseg.schema.base import SortedRequest,SortedSeriesRequest
from code_ai.pipeline.dicomseg.schema.base import StudyRequest,StudySeriesRequest,StudyModelRequest
from code_ai.pipeline.dicomseg.schema.base import MaskRequest,MaskSeriesRequest,MaskInstanceRequest, AITeamRequest

AITeamT      = TypeVar('AITeamT', bound=AITeamRequest)
StudyT       = TypeVar('StudyT', bound=StudyRequest)
StudySeriesT = TypeVar('StudySeriesT', bound=StudySeriesRequest)
StudyModelT  = TypeVar('StudyModelT', bound=StudyModelRequest)

SortedT       = TypeVar('SortedT', bound=SortedRequest)
SortedSeriesT = TypeVar('SortedSeriesT', bound=SortedSeriesRequest)

MaskT         = TypeVar('MaskT', bound=MaskRequest)
MaskSeriesT   = TypeVar('MaskSeriesT', bound=MaskSeriesRequest)
MaskInstanceT = TypeVar('MaskInstanceT', bound=MaskInstanceRequest)