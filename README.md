# PipelineCore

[繁體中文](./docs/README.zh-TW.md) | [English](./docs/README.md)

A Python framework for building machine learning and inference pipelines with GPU resource management and TensorFlow integration.

## Quick Links

- 📖 **[Full Documentation (English)](./docs/README.md)**
- 📖 **[完整文件（繁體中文）](./docs/README.zh-TW.md)**
- 💻 **[Repository](https://github.com/ggggggg13289AI/pipelinecore)**
- 🐛 **[Issue Tracker](https://github.com/ggggggg13289AI/pipelinecore/issues)**

## Features

- **Template Method Pattern**: Extensible pipeline architecture with lifecycle hooks
- **GPU Resource Management**: Automatic GPU availability checking and memory management
- **Type-Safe Generics**: End-to-end type safety with generic type parameters
- **TensorFlow Integration**: Seamless integration with TensorFlow for ML workloads
- **Structured Logging**: Built-in logging infrastructure for pipeline executions

## Installation

### For Users (Installing as a Package)

```bash
# Method 1: Install from Git repository
uv pip install git+https://github.com/ggggggg13289AI/pipelinecore.git

# Method 2: Install from local directory
uv pip install /path/to/pipelinecore

# Method 3: Install in editable mode (for development)
uv pip install -e /path/to/pipelinecore
```

📖 **[Complete Installation Guide](./docs/INSTALLATION_GUIDE.md)** - Detailed instructions for all installation methods

### For Developers (Setting up the Project)

```bash
# Clone the repository
git clone https://github.com/ggggggg13289AI/pipelinecore.git
cd pipelinecore

# Install dependencies with uv
uv sync

# Activate virtual environment
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows
```

## Quick Start

```python
from pipelinecore.core import BasePipeline, PipelineContext, PipelinePaths, LogManager
from pathlib import Path

# Define your pipeline
class MyPipeline(BasePipeline[str, dict, list, str]):
    def prepare(self, payload: str) -> dict:
        return {"data": payload.split()}

    def run_inference(self, prepared: dict) -> list:
        return [word.upper() for word in prepared["data"]]

    def postprocess(self, inference_result: list) -> str:
        return " ".join(inference_result)

# Execute pipeline
pipeline = MyPipeline(context)
result = pipeline.execute("hello world")
```

## Documentation

For comprehensive documentation including architecture details, GPU management, and advanced usage, please refer to:

- **English**: [docs/README.md](./docs/README.md)
- **繁體中文**: [docs/README.zh-TW.md](./docs/README.zh-TW.md)

## Development

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

## Requirements

- Python >=3.10
- TensorFlow >=2.10.0
- pynvml >=11.0.0

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
