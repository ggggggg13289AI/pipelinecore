"""Pytest configuration and shared fixtures for pipelinecore tests."""

import logging
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture
def tmp_pipeline_dir(tmp_path: Path) -> Path:
    """Create a temporary directory structure for pipeline testing."""
    process_dir = tmp_path / "process"
    output_dir = tmp_path / "output"
    log_dir = tmp_path / "logs"
    
    process_dir.mkdir()
    output_dir.mkdir()
    log_dir.mkdir()
    
    return tmp_path


@pytest.fixture
def mock_pipeline_paths(tmp_pipeline_dir: Path) -> Any:
    """Create PipelinePaths instance with temporary directories."""
    from pipelinecore.core import PipelinePaths
    
    return PipelinePaths(
        process_dir=tmp_pipeline_dir / "process",
        output_dir=tmp_pipeline_dir / "output",
        log_dir=tmp_pipeline_dir / "logs",
    )


@pytest.fixture
def mock_logger() -> logging.Logger:
    """Create a test logger."""
    logger = logging.getLogger("test_pipeline")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    return logger


@pytest.fixture
def mock_pipeline_context(mock_pipeline_paths: Any, mock_logger: logging.Logger) -> Any:
    """Create PipelineContext for testing."""
    from pipelinecore.core import PipelineContext
    
    return PipelineContext(
        pipeline_id="test-pipeline-001",
        paths=mock_pipeline_paths,
        logger=mock_logger,
    )


@pytest.fixture(autouse=True)
def reset_tf_config(mocker: Any) -> None:
    """Reset TensorFlow configuration before each test."""
    # Mock TensorFlow to avoid actual GPU access in tests
    mocker.patch("tensorflow.config.experimental.list_physical_devices", return_value=[])
    mocker.patch("tensorflow.config.experimental.get_visible_devices", return_value=[])
    mocker.patch("tensorflow.config.experimental.set_visible_devices")
    mocker.patch("tensorflow.config.experimental.get_memory_growth", return_value=False)
    mocker.patch("tensorflow.config.experimental.set_memory_growth")


@pytest.fixture(autouse=True)
def reset_pynvml(mocker: Any) -> None:
    """Mock pynvml to avoid actual GPU access in tests."""
    mocker.patch("pynvml.nvmlInit")
    mocker.patch("pynvml.nvmlDeviceGetHandleByIndex")
    mocker.patch("pynvml.nvmlDeviceGetMemoryInfo")
