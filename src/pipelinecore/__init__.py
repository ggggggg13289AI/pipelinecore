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
    PipelineError,
    PipelinePaths,
    __version__,
    TimedLogger,
    TimingCollector,
    timed_execution
)

__all__ = [
    "BasePipeline",
    "FileStager",
    "GPUMemoryError",
    "GpuResourceManager",
    "LogManager",
    "PipelineContext",
    "PipelinePaths",
    "PipelineError",
    "__version__",
    "TimedLogger",
    "TimingCollector",
    "timed_execution"
]
