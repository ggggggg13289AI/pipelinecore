"""Tests for pipelinecore.core.base module."""

import logging
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

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
)


class TestPipelineError:
    """Tests for PipelineError exception."""

    def test_pipeline_error_is_runtime_error(self) -> None:
        """PipelineError should inherit from RuntimeError."""
        assert issubclass(PipelineError, RuntimeError)

    def test_pipeline_error_message(self) -> None:
        """PipelineError should preserve error message."""
        msg = "Pipeline failed"
        error = PipelineError(msg)
        assert str(error) == msg


class TestGPUMemoryError:
    """Tests for GPUMemoryError exception."""

    def test_gpu_memory_error_is_pipeline_error(self) -> None:
        """GPUMemoryError should inherit from PipelineError."""
        assert issubclass(GPUMemoryError, PipelineError)

    def test_gpu_memory_error_message(self) -> None:
        """GPUMemoryError should preserve error message."""
        msg = "Insufficient GPU memory"
        error = GPUMemoryError(msg)
        assert str(error) == msg


class TestPipelinePaths:
    """Tests for PipelinePaths dataclass."""

    def test_pipeline_paths_creation(self, tmp_path: Path) -> None:
        """PipelinePaths should create with valid paths."""
        paths = PipelinePaths(
            process_dir=tmp_path / "process",
            output_dir=tmp_path / "output",
            log_dir=tmp_path / "logs",
        )
        assert paths.process_dir == tmp_path / "process"
        assert paths.output_dir == tmp_path / "output"
        assert paths.log_dir == tmp_path / "logs"

    def test_pipeline_paths_frozen(self, tmp_path: Path) -> None:
        """PipelinePaths should be immutable (frozen)."""
        paths = PipelinePaths(
            process_dir=tmp_path / "process",
            output_dir=tmp_path / "output",
            log_dir=tmp_path / "logs",
        )
        with pytest.raises(AttributeError):
            paths.process_dir = tmp_path / "new_process"  # type: ignore

    def test_ensure_creates_directories(self, tmp_path: Path) -> None:
        """PipelinePaths.ensure() should create all directories."""
        paths = PipelinePaths(
            process_dir=tmp_path / "process",
            output_dir=tmp_path / "output",
            log_dir=tmp_path / "logs",
        )
        
        # Directories should not exist yet
        assert not paths.process_dir.exists()
        assert not paths.output_dir.exists()
        assert not paths.log_dir.exists()
        
        # Call ensure
        paths.ensure()
        
        # All directories should now exist
        assert paths.process_dir.exists()
        assert paths.output_dir.exists()
        assert paths.log_dir.exists()

    def test_ensure_idempotent(self, tmp_path: Path) -> None:
        """PipelinePaths.ensure() should be safe to call multiple times."""
        paths = PipelinePaths(
            process_dir=tmp_path / "process",
            output_dir=tmp_path / "output",
            log_dir=tmp_path / "logs",
        )
        
        paths.ensure()
        paths.ensure()  # Should not raise
        
        assert paths.process_dir.exists()
        assert paths.output_dir.exists()
        assert paths.log_dir.exists()


class TestPipelineContext:
    """Tests for PipelineContext dataclass."""

    def test_pipeline_context_creation(self, mock_pipeline_paths: PipelinePaths, mock_logger: logging.Logger) -> None:
        """PipelineContext should create with valid components."""
        context = PipelineContext(
            pipeline_id="test-001",
            paths=mock_pipeline_paths,
            logger=mock_logger,
        )
        assert context.pipeline_id == "test-001"
        assert context.paths == mock_pipeline_paths
        assert context.logger == mock_logger

    def test_pipeline_context_frozen(self, mock_pipeline_paths: PipelinePaths, mock_logger: logging.Logger) -> None:
        """PipelineContext should be immutable (frozen)."""
        context = PipelineContext(
            pipeline_id="test-001",
            paths=mock_pipeline_paths,
            logger=mock_logger,
        )
        with pytest.raises(AttributeError):
            context.pipeline_id = "new-id"  # type: ignore


class TestLogManager:
    """Tests for LogManager class."""

    def test_log_manager_creation(self, tmp_path: Path) -> None:
        """LogManager should create with valid log directory."""
        log_dir = tmp_path / "logs"
        manager = LogManager(log_dir, "test_logger")
        assert manager._log_dir == log_dir
        assert manager._name == "test_logger"

    def test_log_manager_creates_log_directory(self, tmp_path: Path) -> None:
        """LogManager should create log directory on initialization."""
        log_dir = tmp_path / "logs"
        assert not log_dir.exists()
        
        LogManager(log_dir)
        assert log_dir.exists()

    def test_log_manager_default_logger_name(self, tmp_path: Path) -> None:
        """LogManager should use default logger name if not provided."""
        log_dir = tmp_path / "my_pipeline"
        manager = LogManager(log_dir)
        assert manager._name == "pipeline.my_pipeline"

    def test_create_logger(self, tmp_path: Path) -> None:
        """LogManager.create_logger() should create a configured logger."""
        log_dir = tmp_path / "logs"
        manager = LogManager(log_dir, "test_logger")
        
        logger = manager.create_logger()
        
        assert logger.name == "test_logger"
        assert logger.level == logging.INFO
        assert len(logger.handlers) > 0
        assert any(isinstance(h, logging.FileHandler) for h in logger.handlers)

    def test_create_logger_no_duplicate_handlers(self, tmp_path: Path) -> None:
        """LogManager.create_logger() should not add duplicate handlers."""
        log_dir = tmp_path / "logs"
        manager = LogManager(log_dir, "test_logger")
        
        logger1 = manager.create_logger()
        handler_count1 = len(logger1.handlers)
        
        logger2 = manager.create_logger()
        handler_count2 = len(logger2.handlers)
        
        assert handler_count1 == handler_count2


class TestGpuResourceManager:
    """Tests for GpuResourceManager class."""

    def test_gpu_resource_manager_creation(self) -> None:
        """GpuResourceManager should create with valid parameters."""
        manager = GpuResourceManager(gpu_index=0, usage_threshold=0.8)
        assert manager._gpu_index == 0
        assert manager._usage_threshold == 0.8

    def test_gpu_resource_manager_default_parameters(self) -> None:
        """GpuResourceManager should use default parameters."""
        manager = GpuResourceManager(gpu_index=1)
        assert manager._gpu_index == 1
        assert manager._usage_threshold == 0.9
        assert manager._check_interval == 5.0
        assert manager._max_checks == 3
        assert manager._usage_check_enabled is False

    def test_ensure_available_no_gpus(self, mocker: Any) -> None:
        """GpuResourceManager should raise error when no GPUs detected."""
        mocker.patch("tensorflow.config.experimental.list_physical_devices", return_value=[])
        
        manager = GpuResourceManager(gpu_index=0)
        with pytest.raises(GPUMemoryError, match="No GPU devices detected"):
            manager.ensure_available()

    def test_ensure_available_invalid_gpu_index(self, mocker: Any) -> None:
        """GpuResourceManager should raise error for invalid GPU index."""
        mock_gpu = MagicMock()
        mock_gpu.name = "GPU:0"
        mocker.patch("tensorflow.config.experimental.list_physical_devices", return_value=[mock_gpu])
        
        manager = GpuResourceManager(gpu_index=5)  # Invalid index
        with pytest.raises(GPUMemoryError, match="GPU index 5 is out of range"):
            manager.ensure_available()

    def test_ensure_available_success(self, mocker: Any) -> None:
        """GpuResourceManager should successfully configure GPU."""
        mock_gpu = MagicMock()
        mock_gpu.name = "GPU:0"
        mocker.patch("tensorflow.config.experimental.list_physical_devices", return_value=[mock_gpu])
        mocker.patch("tensorflow.config.experimental.get_visible_devices", return_value=[])
        mock_set_visible = mocker.patch("tensorflow.config.experimental.set_visible_devices")
        mock_set_growth = mocker.patch("tensorflow.config.experimental.set_memory_growth")
        mocker.patch("tensorflow.config.experimental.get_memory_growth", return_value=False)
        
        manager = GpuResourceManager(gpu_index=0, usage_check_enabled=False)
        manager.ensure_available()
        
        mock_set_visible.assert_called_once()
        mock_set_growth.assert_called_once()

    def test_wait_for_gpu_capacity_success(self, mocker: Any) -> None:
        """GpuResourceManager should pass when GPU usage is below threshold."""
        mock_handle = MagicMock()
        mock_info = MagicMock()
        mock_info.used = 4 * 1024**3  # 4 GB
        mock_info.total = 8 * 1024**3  # 8 GB (usage = 0.5)
        
        mocker.patch("pynvml.nvmlInit")
        mocker.patch("pynvml.nvmlDeviceGetHandleByIndex", return_value=mock_handle)
        mocker.patch("pynvml.nvmlDeviceGetMemoryInfo", return_value=mock_info)
        
        manager = GpuResourceManager(gpu_index=0, usage_threshold=0.9, usage_check_enabled=True)
        manager._wait_for_gpu_capacity()  # Should not raise

    def test_wait_for_gpu_capacity_exceeds_threshold(self, mocker: Any) -> None:
        """GpuResourceManager should raise error when GPU usage exceeds threshold."""
        mock_handle = MagicMock()
        mock_info = MagicMock()
        mock_info.used = 7.5 * 1024**3  # 7.5 GB
        mock_info.total = 8 * 1024**3    # 8 GB (usage = 0.9375)
        
        mocker.patch("pynvml.nvmlInit")
        mocker.patch("pynvml.nvmlDeviceGetHandleByIndex", return_value=mock_handle)
        mocker.patch("pynvml.nvmlDeviceGetMemoryInfo", return_value=mock_info)
        
        manager = GpuResourceManager(gpu_index=0, usage_threshold=0.9, max_checks=1, usage_check_enabled=True)
        with pytest.raises(GPUMemoryError, match="usage ratio.*exceeds threshold"):
            manager._wait_for_gpu_capacity()


class TestTensorflowPipelineMixin:
    """Tests for TensorflowPipelineMixin class."""

    def test_mixin_initialization(self) -> None:
        """TensorflowPipelineMixin should store GPU manager."""
        gpu_manager = GpuResourceManager(gpu_index=0)
        mixin = TensorflowPipelineMixin(gpu_manager)
        assert mixin._gpu_manager is gpu_manager

    def test_before_run_calls_ensure_available(self, mocker: Any) -> None:
        """TensorflowPipelineMixin.before_run() should call GPU manager."""
        gpu_manager = GpuResourceManager(gpu_index=0)
        mock_ensure = mocker.patch.object(gpu_manager, "ensure_available")
        
        mixin = TensorflowPipelineMixin(gpu_manager)
        mixin.before_run()
        
        mock_ensure.assert_called_once()


class TestFileStager:
    """Tests for FileStager class."""

    def test_file_stager_creation(self, tmp_path: Path) -> None:
        """FileStager should create with valid base directory."""
        stager = FileStager(tmp_path)
        assert stager._base_dir == tmp_path

    def test_copy_into_single_file(self, tmp_path: Path) -> None:
        """FileStager.copy_into() should copy a single file."""
        # Create source file
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        source_file = source_dir / "test.txt"
        source_file.write_text("test content")
        
        # Setup destination
        dest_dir = tmp_path / "dest"
        
        # Copy file
        stager = FileStager(tmp_path)
        stager.copy_into([source_file], dest_dir)
        
        # Verify
        assert dest_dir.exists()
        copied_file = dest_dir / "test.txt"
        assert copied_file.exists()
        assert copied_file.read_text() == "test content"

    def test_copy_into_multiple_files(self, tmp_path: Path) -> None:
        """FileStager.copy_into() should copy multiple files."""
        # Create source files
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        file1 = source_dir / "file1.txt"
        file2 = source_dir / "file2.txt"
        file1.write_text("content 1")
        file2.write_text("content 2")
        
        # Setup destination
        dest_dir = tmp_path / "dest"
        
        # Copy files
        stager = FileStager(tmp_path)
        stager.copy_into([file1, file2], dest_dir)
        
        # Verify
        assert (dest_dir / "file1.txt").exists()
        assert (dest_dir / "file2.txt").exists()
        assert (dest_dir / "file1.txt").read_text() == "content 1"
        assert (dest_dir / "file2.txt").read_text() == "content 2"

    def test_copy_into_creates_destination(self, tmp_path: Path) -> None:
        """FileStager.copy_into() should create destination directory."""
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        source_file = source_dir / "test.txt"
        source_file.write_text("test")
        
        dest_dir = tmp_path / "nested" / "dest"
        assert not dest_dir.exists()
        
        stager = FileStager(tmp_path)
        stager.copy_into([source_file], dest_dir)
        
        assert dest_dir.exists()
        assert (dest_dir / "test.txt").exists()


class TestBasePipeline:
    """Tests for BasePipeline abstract class."""

    class ConcretePipeline(BasePipeline[str, str, str, str]):
        """Concrete implementation for testing."""

        def prepare(self, payload: str) -> str:
            return f"prepared_{payload}"

        def run_inference(self, prepared: str) -> str:
            return f"inferred_{prepared}"

        def postprocess(self, inference_result: str) -> str:
            return f"result_{inference_result}"

    def test_pipeline_initialization(self, mock_pipeline_context: PipelineContext) -> None:
        """BasePipeline should initialize with context."""
        pipeline = self.ConcretePipeline(mock_pipeline_context)
        assert pipeline.context is mock_pipeline_context

    def test_execute_flow(self, mock_pipeline_context: PipelineContext) -> None:
        """BasePipeline.execute() should run complete pipeline flow."""
        pipeline = self.ConcretePipeline(mock_pipeline_context)
        result = pipeline.execute("test_input")
        
        assert result == "result_inferred_prepared_test_input"

    def test_execute_creates_directories(self, mock_pipeline_context: PipelineContext) -> None:
        """BasePipeline.execute() should create required directories."""
        pipeline = self.ConcretePipeline(mock_pipeline_context)
        
        # Remove directories
        for path in [mock_pipeline_context.paths.process_dir,
                     mock_pipeline_context.paths.output_dir,
                     mock_pipeline_context.paths.log_dir]:
            path.rmdir()
        
        pipeline.execute("test")
        
        # Directories should be recreated
        assert mock_pipeline_context.paths.process_dir.exists()
        assert mock_pipeline_context.paths.output_dir.exists()
        assert mock_pipeline_context.paths.log_dir.exists()

    def test_execute_calls_hooks(self, mock_pipeline_context: PipelineContext, mocker: Any) -> None:
        """BasePipeline.execute() should call before_run and after_run hooks."""
        pipeline = self.ConcretePipeline(mock_pipeline_context)
        
        mock_before = mocker.patch.object(pipeline, "before_run")
        mock_after = mocker.patch.object(pipeline, "after_run", return_value="hooked_result")
        
        result = pipeline.execute("test")
        
        mock_before.assert_called_once()
        mock_after.assert_called_once()
        assert result == "hooked_result"

    def test_execute_logs_pipeline_lifecycle(self, mock_pipeline_context: PipelineContext, mocker: Any) -> None:
        """BasePipeline.execute() should log pipeline start and finish."""
        pipeline = self.ConcretePipeline(mock_pipeline_context)
        mock_info = mocker.patch.object(mock_pipeline_context.logger, "info")
        
        pipeline.execute("test")
        
        assert mock_info.call_count == 2
        start_call = mock_info.call_args_list[0]
        finish_call = mock_info.call_args_list[1]
        
        assert "started" in start_call[0][0]
        assert "finished" in finish_call[0][0]

    def test_abstract_methods_raise_not_implemented(self, mock_pipeline_context: PipelineContext) -> None:
        """BasePipeline abstract methods should raise NotImplementedError."""
        pipeline = BasePipeline(mock_pipeline_context)
        
        with pytest.raises(NotImplementedError):
            pipeline.prepare("test")  # type: ignore
        
        with pytest.raises(NotImplementedError):
            pipeline.run_inference("test")  # type: ignore
        
        with pytest.raises(NotImplementedError):
            pipeline.postprocess("test")  # type: ignore

    def test_before_run_default_implementation(self, mock_pipeline_context: PipelineContext) -> None:
        """BasePipeline.before_run() should have empty default implementation."""
        pipeline = self.ConcretePipeline(mock_pipeline_context)
        pipeline.before_run()  # Should not raise

    def test_after_run_default_returns_result(self, mock_pipeline_context: PipelineContext) -> None:
        """BasePipeline.after_run() should return result unchanged by default."""
        pipeline = self.ConcretePipeline(mock_pipeline_context)
        result = pipeline.after_run("test_result")
        assert result == "test_result"
