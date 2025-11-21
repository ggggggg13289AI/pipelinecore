"""
PipelineCore - A Python framework for building ML/inference pipelines.

This package provides a template method pattern-based pipeline framework
with GPU resource management and TensorFlow integration.
"""
from . import dicomseg
from . import inference
from . import resample
from . import resource
from . import upload

from pipelinecore.core import (
    BasePipeline,
    FileStager,
    GPUMemoryError,
    GpuResourceManager,
    LogManager,
    PipelineContext,
    PipelineError,
    PipelinePaths,
    TensorflowPipelineMixin,
    __version__,
)

__all__ = [
    "dicomseg",
    "inference",
    "resample",
    "resource",
    "upload",
    "BasePipeline",
    "FileStager",
    "GPUMemoryError",
    "GpuResourceManager",
    "LogManager",
    "PipelineContext",
    "PipelinePaths",
    "PipelineError",
    "TensorflowPipelineMixin",
    "__version__",
]
