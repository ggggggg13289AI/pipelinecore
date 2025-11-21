# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**pipelinecore** is a Python framework for building machine learning/inference pipelines with GPU resource management and TensorFlow integration. The architecture follows a template method pattern with lifecycle hooks for extensibility.

## Development Commands

### Package Management
```bash
# Install dependencies (using uv package manager)
uv sync

# Activate virtual environment
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows
```

### Python Environment
- **Required Python**: >=3.10
- **Package Manager**: `uv` (version 0.8.4+)
- **Build Backend**: `uv_build`

### Running the Application
```bash
# Execute the main pipeline
pipelinecore

# Or via Python module
python -m pipelinecore
```

## Architecture

### Core Pipeline Pattern

The framework uses a **Template Method Pattern** with type-safe generics:

```python
BasePipeline[InputT, PreparedT, InferenceT, OutputT]
```

**Pipeline Execution Flow** (src/pipelinecore/core/base.py:189-198):
1. `execute()` - Orchestrates the entire pipeline
2. `before_run()` - Pre-execution hook
3. `prepare()` - Transform input → prepared data
4. `run_inference()` - Execute model inference
5. `postprocess()` - Transform inference → output
6. `after_run()` - Post-execution hook with result

### Key Components

**PipelineContext** (base.py:42-48):
- Shared runtime context for pipeline execution
- Contains: `pipeline_id`, `paths` (PipelinePaths), `logger`
- Immutable dataclass ensuring consistent state

**GpuResourceManager** (base.py:72-134):
- Encapsulates GPU availability checks and TensorFlow device configuration
- Features:
  - GPU memory usage monitoring via `pynvml`
  - Automatic device visibility configuration
  - Memory growth enablement
  - Configurable retry logic with `max_checks` and `check_interval`
- Critical for multi-GPU environments to prevent OOM errors

**TensorflowPipelineMixin** (base.py:137-144):
- Mixin class ensuring GPU availability before TensorFlow inference
- Integrates `GpuResourceManager` into pipeline lifecycle
- Use with `BasePipeline` for GPU-dependent pipelines

**LogManager** (base.py:51-70):
- Factory for structured pipeline loggers
- Creates daily log files: `{log_dir}/YYYYMMDD.log`
- Prevents duplicate handlers when creating multiple loggers

**FileStager** (base.py:147-159):
- Utility for staging/copying files into working directories
- Handles batch file operations with `copy_into()`

### Type System

Four generic type parameters provide end-to-end type safety:
- `InputT` - Raw input payload type
- `PreparedT` - Preprocessed data type
- `InferenceT` - Model inference output type
- `OutputT` - Final processed output type

### Error Handling

**Custom Exceptions**:
- `PipelineError` (base.py:22) - Generic pipeline runtime errors
- `GPUMemoryError` (base.py:26) - GPU capacity/availability issues

## Implementation Patterns

### Creating a Pipeline

```python
from pipelinecore.core import BasePipeline, PipelineContext, TensorflowPipelineMixin, GpuResourceManager

class MyPipeline(TensorflowPipelineMixin, BasePipeline[Input, Prepared, Inference, Output]):
    def __init__(self, context: PipelineContext):
        gpu_manager = GpuResourceManager(gpu_index=0, usage_check_enabled=True)
        TensorflowPipelineMixin.__init__(self, gpu_manager)
        BasePipeline.__init__(self, context)

    def prepare(self, payload: Input) -> Prepared:
        # Implement preprocessing logic
        pass

    def run_inference(self, prepared: Prepared) -> Inference:
        # Implement model inference
        pass

    def postprocess(self, inference_result: Inference) -> Output:
        # Implement output formatting
        pass
```

### GPU Resource Management

**For TensorFlow pipelines**:
- Always use `TensorflowPipelineMixin` when GPU resources are required
- Configure `GpuResourceManager` with appropriate thresholds:
  - `gpu_index`: Target GPU device
  - `usage_threshold`: Max memory usage ratio (default 0.9)
  - `usage_check_enabled`: Enable/disable memory checks

**Best Practices**:
- Enable `usage_check_enabled=True` in multi-pipeline environments
- Set `max_checks` and `check_interval` based on expected GPU contention
- Handle `GPUMemoryError` appropriately in orchestration code

### Directory Structure

```
pipelinecore/
├── src/pipelinecore/
│   └── core/
│       ├── __init__.py       # Public API exports
│       └── base.py           # Core pipeline classes
├── pyproject.toml            # Project configuration
└── README.md                 # Project documentation
```

## Dependencies

**Core Dependencies**:
- `tensorflow` - Deep learning framework
- `pynvml` - NVIDIA GPU monitoring

**No Optional Dependencies**: This is a minimal framework requiring only essential packages.

## Notes

- **No Tests Yet**: Test suite not yet implemented
- **Minimal Framework**: Focus on core pipeline abstractions without additional plugins
- **GPU-Centric**: Designed primarily for GPU-accelerated inference workflows
- **Type-Safe**: Leverages Python generics for compile-time type checking
