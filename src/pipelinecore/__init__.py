"""
PipelineCore - A Python framework for building ML/inference pipelines.

This package provides a template method pattern-based pipeline framework
with GPU resource management and TensorFlow integration.
"""

from pipelinecore.core import (
    BasePipeline,
    FileStager,
    GPUMemoryError,
    GpuResourceManager,
    LogManager,
    PipelineContext,
    PipelinePaths,
    PipelineError,
    TensorflowPipelineMixin,
    __version__,
)

__all__ = [
    "__version__",
    "BasePipeline",
    "FileStager",
    "GPUMemoryError",
    "GpuResourceManager",
    "LogManager",
    "PipelineContext",
    "PipelinePaths",
    "PipelineError",
    "TensorflowPipelineMixin",
]
