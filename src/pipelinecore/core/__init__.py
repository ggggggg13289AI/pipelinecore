"""PipelineCore - ML/Inference Pipeline Framework with GPU Management."""

__version__ = "0.1.0"

from .base import (
    BasePipeline,
    FileStager,
    GPUMemoryError,
    GpuResourceManager,
    LogManager,
    PipelineContext,
    PipelineError,
    PipelinePaths,
    TensorflowPipelineMixin,
)
from .protocol import LoggerLike
from .timing import (
    TimedLogger,
    TimingCollector,
    TimingResult,
    timed_execution,
)

__all__ = [
    "__version__",
    # Base classes
    "BasePipeline",
    "FileStager",
    "GPUMemoryError",
    "GpuResourceManager",
    "LogManager",
    "PipelineContext",
    "PipelinePaths",
    "PipelineError",
    "TensorflowPipelineMixin",
    # Protocols
    "LoggerLike",
    # Timing
    "TimedLogger",
    "TimingCollector",
    "TimingResult",
    "timed_execution",
]
