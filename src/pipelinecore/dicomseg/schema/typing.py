from typing import TypeVar

from .base import (
    AITeamRequest,
    MaskInstanceRequest,
    MaskRequest,
    MaskSeriesRequest,
    SortedRequest,
    SortedSeriesRequest,
    StudyModelRequest,
    StudyRequest,
    StudySeriesRequest,
)

AITeamT = TypeVar("AITeamT", bound=AITeamRequest)
StudyT = TypeVar("StudyT", bound=StudyRequest)
StudySeriesT = TypeVar("StudySeriesT", bound=StudySeriesRequest)
StudyModelT = TypeVar("StudyModelT", bound=StudyModelRequest)

SortedT = TypeVar("SortedT", bound=SortedRequest)
SortedSeriesT = TypeVar("SortedSeriesT", bound=SortedSeriesRequest)

MaskT = TypeVar("MaskT", bound=MaskRequest)
MaskSeriesT = TypeVar("MaskSeriesT", bound=MaskSeriesRequest)
MaskInstanceT = TypeVar("MaskInstanceT", bound=MaskInstanceRequest)
