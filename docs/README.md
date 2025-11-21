# PipelineCore

[繁體中文](./README.zh-TW.md) | **English**

A Python framework for building machine learning and inference pipelines with GPU resource management and TensorFlow integration.

## Features

- **Template Method Pattern**: Extensible pipeline architecture with lifecycle hooks
- **GPU Resource Management**: Automatic GPU availability checking and memory management
- **Type-Safe Generics**: End-to-end type safety with generic type parameters
- **TensorFlow Integration**: Seamless integration with TensorFlow for ML workloads
- **Structured Logging**: Built-in logging infrastructure for pipeline executions

## Installation

### Using uv (Recommended)

```bash
# Clone the repository
git clone https://github.com/ggggggg13289AI/pipelinecore
cd pipelinecore

# Install with uv
uv sync

# Activate virtual environment
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows
```

### Using pip

```bash
pip install pipelinecore
```

## Quick Start

### Basic Pipeline Example

```python
from pipelinecore.core import BasePipeline, PipelineContext, PipelinePaths, LogManager
from pathlib import Path

# Define your pipeline with type parameters
class MyPipeline(BasePipeline[str, dict, list, str]):
    def prepare(self, payload: str) -> dict:
        """Preprocess the input data"""
        return {"data": payload.split()}

    def run_inference(self, prepared: dict) -> list:
        """Run model inference"""
        return [word.upper() for word in prepared["data"]]

    def postprocess(self, inference_result: list) -> str:
        """Format the output"""
        return " ".join(inference_result)

# Create pipeline context
paths = PipelinePaths(
    process_dir=Path("./data/process"),
    output_dir=Path("./data/output"),
    log_dir=Path("./logs")
)
paths.ensure()

log_manager = LogManager(paths.log_dir)
context = PipelineContext(
    pipeline_id="my_pipeline_001",
    paths=paths,
    logger=log_manager.create_logger()
)

# Execute pipeline
pipeline = MyPipeline(context)
result = pipeline.execute("hello world")
print(result)  # Output: HELLO WORLD
```

### GPU-Enabled Pipeline

```python
from pipelinecore.core import (
    BasePipeline,
    TensorflowPipelineMixin,
    GpuResourceManager,
    PipelineContext
)

class GpuPipeline(TensorflowPipelineMixin, BasePipeline[Input, Prepared, Inference, Output]):
    def __init__(self, context: PipelineContext):
        # Configure GPU resource manager
        gpu_manager = GpuResourceManager(
            gpu_index=0,
            usage_threshold=0.9,
            usage_check_enabled=True
        )

        TensorflowPipelineMixin.__init__(self, gpu_manager)
        BasePipeline.__init__(self, context)

    def prepare(self, payload: Input) -> Prepared:
        # Your preprocessing logic
        pass

    def run_inference(self, prepared: Prepared) -> Inference:
        # GPU-accelerated TensorFlow inference
        # GPU availability is automatically ensured before this runs
        pass

    def postprocess(self, inference_result: Inference) -> Output:
        # Your postprocessing logic
        pass
```

## Architecture

### Core Components

#### BasePipeline

The foundation of all pipelines, implementing the Template Method pattern:

```python
BasePipeline[InputT, PreparedT, InferenceT, OutputT]
```

**Type Parameters:**
- `InputT`: Raw input payload type
- `PreparedT`: Preprocessed data type
- `InferenceT`: Model inference output type
- `OutputT`: Final processed output type

**Lifecycle Methods:**
1. `before_run()` - Pre-execution hook
2. `prepare(payload)` - Transform input → prepared data
3. `run_inference(prepared)` - Execute model inference
4. `postprocess(inference)` - Transform inference → output
5. `after_run(result)` - Post-execution hook

#### GpuResourceManager

Manages GPU availability and TensorFlow device configuration:

- GPU memory usage monitoring via NVIDIA Management Library (pynvml)
- Automatic device visibility configuration
- Memory growth enablement
- Configurable retry logic for GPU availability

#### TensorflowPipelineMixin

Mixin class for GPU-dependent pipelines:

- Integrates GpuResourceManager into pipeline lifecycle
- Ensures GPU availability before inference
- Handles TensorFlow device configuration

#### PipelineContext

Shared runtime context for pipeline execution:

```python
@dataclass(frozen=True)
class PipelineContext:
    pipeline_id: str
    paths: PipelinePaths
    logger: logging.Logger
```

## Development

### Requirements

- Python >=3.10
- TensorFlow >=2.10.0
- pynvml >=11.0.0

### Setup Development Environment

```bash
# Install development dependencies
uv sync

# Run tests
pytest

# Run linting
ruff check src/

# Type checking
mypy src/
```

### Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/pipelinecore --cov-report=html

# Run specific test file
pytest tests/test_base.py
```

### Code Quality

```bash
# Format code
ruff format src/

# Lint code
ruff check src/ --fix

# Type check
mypy src/
```

## Project Structure

```
pipelinecore/
├── src/pipelinecore/
│   └── core/
│       ├── __init__.py       # Public API exports
│       └── base.py           # Core pipeline classes
├── tests/                    # Test suite
├── docs/                     # Documentation
│   ├── README.md            # English documentation
│   └── README.zh-TW.md      # Traditional Chinese documentation
├── pyproject.toml           # Project configuration
└── CLAUDE.md                # Claude Code guidance
```

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Links

- **Repository**: https://github.com/user/pipelinecore
- **Issue Tracker**: https://github.com/user/pipelinecore/issues
- **Documentation**: https://github.com/user/pipelinecore/blob/main/docs/README.md
