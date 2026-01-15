from __future__ import annotations

import logging
import shutil
import time
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Generic, TypeVar

logger = logging.getLogger(__name__)

# Lazy imports for GPU dependencies
# tensorflow and pynvml are imported inside classes that need them

InputT = TypeVar("InputT")
PreparedT = TypeVar("PreparedT")
InferenceT = TypeVar("InferenceT")
OutputT = TypeVar("OutputT")


class PipelineError(RuntimeError):
    """Generic pipeline runtime error."""


class GPUMemoryError(PipelineError):
    """Raised when the configured GPU does not have enough free memory."""


@dataclass(frozen=True)
class PipelinePaths:
    """Convenience wrapper for commonly used pipeline directories."""

    process_dir: Path
    output_dir: Path
    log_dir: Path

    def ensure(self) -> None:
        for path in (self.process_dir, self.output_dir, self.log_dir):
            path.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class PipelineContext:
    """Shared runtime context for every pipeline execution."""

    pipeline_id: str
    paths: PipelinePaths
    logger: logging.Logger


class LogManager:
    """Factory that produces structured loggers for pipeline runs."""

    def __init__(self, log_dir: Path, logger_name: str | None = None) -> None:
        self._log_dir = log_dir
        self._name = logger_name or f"pipeline.{log_dir.name}"
        self._log_dir.mkdir(parents=True, exist_ok=True)

    def create_logger(self) -> logging.Logger:
        log_file = self._log_dir / f"{datetime.now():%Y%m%d}.log"
        logger = logging.getLogger(self._name)
        logger.setLevel(logging.INFO)
        if not any(isinstance(handler, logging.FileHandler) and handler.baseFilename == str(log_file)
                   for handler in logger.handlers):
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        return logger


class GpuResourceManager:
    """Encapsulates the GPU availability checks and TensorFlow device setup."""

    def __init__(
        self,
        gpu_index: int,
        usage_threshold: float = 0.9,
        check_interval: float = 5.0,
        max_checks: int = 3,
        usage_check_enabled: bool = False,
    ) -> None:
        self._gpu_index = gpu_index
        self._usage_threshold = usage_threshold
        self._check_interval = check_interval
        self._max_checks = max(1, max_checks)
        self._usage_check_enabled = usage_check_enabled

    def ensure_available(self) -> None:
        import tensorflow as tf  # Lazy import

        gpus = tf.config.experimental.list_physical_devices("GPU")
        if not gpus:
            raise GPUMemoryError("No GPU devices detected by TensorFlow.")
        if self._gpu_index >= len(gpus):
            raise GPUMemoryError(
                f"GPU index {self._gpu_index} is out of range for available devices {len(gpus)}."
            )

        target_device = gpus[self._gpu_index]
        self._set_visible_device(target_device)
        self._enable_memory_growth(target_device)

        if self._usage_check_enabled:
            self._wait_for_gpu_capacity()

    def _wait_for_gpu_capacity(self) -> None:
        import pynvml  # Lazy import

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(self._gpu_index)
        for attempt in range(self._max_checks):
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            usage_ratio = info.used / info.total
            if usage_ratio < self._usage_threshold:
                return
            if attempt + 1 == self._max_checks:
                raise GPUMemoryError(
                    f"GPU {self._gpu_index} usage ratio {usage_ratio:.2f} exceeds threshold "
                    f"{self._usage_threshold:.2f}"
                )
            time.sleep(self._check_interval)

    def _set_visible_device(self, device: "tf.config.PhysicalDevice") -> None:
        import tensorflow as tf  # Lazy import

        visible_devices = tf.config.experimental.get_visible_devices("GPU")
        if any(dev.name == device.name for dev in visible_devices):
            return
        try:
            tf.config.experimental.set_visible_devices([device], "GPU")
        except RuntimeError as exc:  # pragma: no cover - TF internal state
            logger.warning("Unable to set visible GPU devices: %s", exc)

    def _enable_memory_growth(self, device: "tf.config.PhysicalDevice") -> None:
        import tensorflow as tf  # Lazy import

        try:
            if not tf.config.experimental.get_memory_growth(device):
                tf.config.experimental.set_memory_growth(device, True)
        except RuntimeError as exc:  # pragma: no cover - TF internal state
            logger.warning("Unable to set memory growth for %s: %s", device.name, exc)


class TensorflowPipelineMixin:
    """Mixin that ensures GPU availability before running TensorFlow inference."""

    def __init__(self, gpu_manager: GpuResourceManager) -> None:
        self._gpu_manager = gpu_manager

    def before_run(self) -> None:  # pragma: no cover - thin coordination layer
        self._gpu_manager.ensure_available()


class FileStager:
    """Utility responsible for staging and copying files into working directories."""

    def __init__(self, base_dir: Path) -> None:
        self._base_dir = base_dir

    def copy_into(self, sources: Iterable[Path], destination: Path) -> None:
        destination.mkdir(parents=True, exist_ok=True)
        for src in sources:
            path_obj = Path(src)
            target = destination / path_obj.name
            shutil.copy2(path_obj, target)


class BasePipeline(Generic[InputT, PreparedT, InferenceT, OutputT]):
    """
    Template method based pipeline skeleton.

    The execute method coordinates the high level flow while
    subclasses implement the individual steps.
    """

    def __init__(self, context: PipelineContext) -> None:
        self.context = context

    def before_run(self) -> None:
        """Hook executed right before the template method starts."""

    def after_run(self, result: OutputT) -> OutputT:
        """Hook executed right after the template method completes."""
        return result

    def prepare(self, payload: InputT) -> PreparedT:  # pragma: no cover - abstract
        raise NotImplementedError

    def run_inference(self, prepared: PreparedT) -> InferenceT:  # pragma: no cover - abstract
        raise NotImplementedError

    def postprocess(self, inference_result: InferenceT) -> OutputT:  # pragma: no cover - abstract
        raise NotImplementedError

    def execute(self, payload: InputT) -> OutputT:
        self.context.paths.ensure()
        self.before_run()
        self.context.logger.info("Pipeline %s started.", self.context.pipeline_id)
        prepared = self.prepare(payload)
        inference = self.run_inference(prepared)
        result = self.postprocess(inference)
        final_result = self.after_run(result)
        self.context.logger.info("Pipeline %s finished.", self.context.pipeline_id)
        return final_result


