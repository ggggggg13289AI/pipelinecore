# PipelineCore 安裝指南

[English Version](#english-version) | [繁體中文版本](#繁體中文版本)

---

## 繁體中文版本

### 將 PipelineCore 安裝為套件

PipelineCore 可以作為 Python 套件安裝，供其他專案使用。以下是不同的安裝方式：

#### 方法 1: 從本地開發目錄安裝（開發模式）

適用於正在開發 pipelinecore 並希望在其他專案中即時測試變更的情況。

```bash
# 進入您的目標專案目錄
cd /path/to/your/project

# 使用 uv 以可編輯模式安裝 pipelinecore
uv pip install -e /path/to/pipelinecore

# 或使用標準 pip
pip install -e /path/to/pipelinecore
```

**優點**：
- ✅ 在 pipelinecore 中的變更會立即反映在其他專案中
- ✅ 無需重新安裝即可測試修改
- ✅ 適合同時開發多個相關專案

**缺點**：
- ⚠️ 需要保持 pipelinecore 原始碼在本地
- ⚠️ 不適合生產環境部署

#### 方法 2: 從本地目錄安裝（正式安裝）

適用於穩定版本的安裝。

```bash
# 進入您的目標專案目錄
cd /path/to/your/project

# 使用 uv 安裝
uv pip install /path/to/pipelinecore

# 或使用標準 pip
pip install /path/to/pipelinecore
```

**優點**：
- ✅ 標準的套件安裝方式
- ✅ 安裝後不依賴原始碼目錄
- ✅ 適合正式環境使用

#### 方法 3: 從 Git 儲存庫安裝

適用於從遠端 Git 儲存庫安裝。

```bash
# 從 GitHub 安裝最新版本
uv pip install git+https://github.com/ggggggg13289AI/pipelinecore.git

# 或使用標準 pip
pip install git+https://github.com/ggggggg13289AI/pipelinecore.git

# 安裝特定分支
uv pip install git+https://github.com/ggggggg13289AI/pipelinecore.git@feature-branch

# 安裝特定標籤/版本
uv pip install git+https://github.com/ggggggg13289AI/pipelinecore.git@v0.1.0
```

**優點**：
- ✅ 可以指定特定版本或分支
- ✅ 適合 CI/CD 流程
- ✅ 不需要本地原始碼

#### 方法 4: 建置並分發 Wheel 套件

適用於需要分發給多個使用者或部署到多個環境的情況。

```bash
# 1. 在 pipelinecore 目錄中建置套件
cd /path/to/pipelinecore

# 安裝建置工具
uv pip install build

# 建置 wheel 和 source distribution
python -m build

# 這會在 dist/ 目錄中產生：
# - pipelinecore-0.1.0-py3-none-any.whl
# - pipelinecore-0.1.0.tar.gz

# 2. 在目標專案中安裝建置好的 wheel
cd /path/to/your/project
uv pip install /path/to/pipelinecore/dist/pipelinecore-0.1.0-py3-none-any.whl

# 或使用標準 pip
pip install /path/to/pipelinecore/dist/pipelinecore-0.1.0-py3-none-any.whl
```

**優點**：
- ✅ 專業的套件分發方式
- ✅ 可以上傳到私有或公開 PyPI
- ✅ 安裝速度快
- ✅ 適合生產環境

### 在專案中使用 PipelineCore

安裝完成後，您可以在專案中這樣使用：

```python
# 匯入核心元件
from pipelinecore.core import (
    BasePipeline,
    PipelineContext,
    PipelinePaths,
    LogManager,
    GpuResourceManager,
    TensorflowPipelineMixin,
)

# 建立您的管線
class MyCustomPipeline(BasePipeline[InputType, PreparedType, InferenceType, OutputType]):
    def prepare(self, payload: InputType) -> PreparedType:
        # 您的實作
        pass

    def run_inference(self, prepared: PreparedType) -> InferenceType:
        # 您的實作
        pass

    def postprocess(self, inference_result: InferenceType) -> OutputType:
        # 您的實作
        pass
```

### 將 PipelineCore 加入專案依賴

#### 使用 pyproject.toml

```toml
[project]
dependencies = [
    "pipelinecore @ git+https://github.com/ggggggg13289AI/pipelinecore.git@v0.1.0",
    # 其他依賴...
]

# 或如果從本地路徑安裝
[tool.uv.sources]
pipelinecore = { path = "../pipelinecore", editable = true }
```

#### 使用 requirements.txt

```txt
# 從 Git 安裝
pipelinecore @ git+https://github.com/ggggggg13289AI/pipelinecore.git@v0.1.0

# 或從本地路徑安裝（開發模式）
-e /path/to/pipelinecore

# 或從本地路徑安裝（正式安裝）
/path/to/pipelinecore
```

### 驗證安裝

```bash
# 檢查是否成功安裝
python -c "import pipelinecore; print(pipelinecore.__version__)"

# 或使用 uv/pip 檢查
uv pip list | grep pipelinecore
# 或
pip list | grep pipelinecore
```

### 疑難排解

#### 問題 1: 找不到模組

```bash
# 確認 pipelinecore 已安裝
uv pip list | grep pipelinecore

# 重新安裝
uv pip uninstall pipelinecore
uv pip install -e /path/to/pipelinecore
```

#### 問題 2: 依賴衝突

```bash
# 查看依賴樹
uv pip tree

# 或升級相關套件
uv pip install --upgrade tensorflow pynvml
```

#### 問題 3: 權限問題

```bash
# 使用虛擬環境（推薦）
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# 或
.venv\Scripts\activate  # Windows

# 然後再安裝
uv pip install -e /path/to/pipelinecore
```

---

## English Version

### Installing PipelineCore as a Package

PipelineCore can be installed as a Python package for use in other projects. Here are different installation methods:

#### Method 1: Install from Local Development Directory (Editable Mode)

Suitable when you're actively developing pipelinecore and want to test changes in other projects immediately.

```bash
# Navigate to your target project directory
cd /path/to/your/project

# Install pipelinecore in editable mode using uv
uv pip install -e /path/to/pipelinecore

# Or using standard pip
pip install -e /path/to/pipelinecore
```

**Pros**:
- ✅ Changes in pipelinecore are immediately reflected in other projects
- ✅ No need to reinstall after modifications
- ✅ Perfect for simultaneous multi-project development

**Cons**:
- ⚠️ Requires keeping pipelinecore source code locally
- ⚠️ Not suitable for production deployment

#### Method 2: Install from Local Directory (Standard Installation)

Suitable for stable version installation.

```bash
# Navigate to your target project directory
cd /path/to/your/project

# Install using uv
uv pip install /path/to/pipelinecore

# Or using standard pip
pip install /path/to/pipelinecore
```

**Pros**:
- ✅ Standard package installation method
- ✅ Independent of source code directory after installation
- ✅ Suitable for production environments

#### Method 3: Install from Git Repository

Suitable for installing from remote Git repository.

```bash
# Install latest version from GitHub
uv pip install git+https://github.com/ggggggg13289AI/pipelinecore.git

# Or using standard pip
pip install git+https://github.com/ggggggg13289AI/pipelinecore.git

# Install specific branch
uv pip install git+https://github.com/ggggggg13289AI/pipelinecore.git@feature-branch

# Install specific tag/version
uv pip install git+https://github.com/ggggggg13289AI/pipelinecore.git@v0.1.0
```

**Pros**:
- ✅ Can specify specific versions or branches
- ✅ Suitable for CI/CD pipelines
- ✅ No local source code required

#### Method 4: Build and Distribute Wheel Package

Suitable for distribution to multiple users or deployment to multiple environments.

```bash
# 1. Build package in pipelinecore directory
cd /path/to/pipelinecore

# Install build tools
uv pip install build

# Build wheel and source distribution
python -m build

# This generates in dist/ directory:
# - pipelinecore-0.1.0-py3-none-any.whl
# - pipelinecore-0.1.0.tar.gz

# 2. Install built wheel in target project
cd /path/to/your/project
uv pip install /path/to/pipelinecore/dist/pipelinecore-0.1.0-py3-none-any.whl

# Or using standard pip
pip install /path/to/pipelinecore/dist/pipelinecore-0.1.0-py3-none-any.whl
```

**Pros**:
- ✅ Professional package distribution method
- ✅ Can be uploaded to private or public PyPI
- ✅ Fast installation
- ✅ Suitable for production environments

### Using PipelineCore in Your Project

After installation, you can use it in your project like this:

```python
# Import core components
from pipelinecore.core import (
    BasePipeline,
    PipelineContext,
    PipelinePaths,
    LogManager,
    GpuResourceManager,
    TensorflowPipelineMixin,
)

# Create your pipeline
class MyCustomPipeline(BasePipeline[InputType, PreparedType, InferenceType, OutputType]):
    def prepare(self, payload: InputType) -> PreparedType:
        # Your implementation
        pass

    def run_inference(self, prepared: PreparedType) -> InferenceType:
        # Your implementation
        pass

    def postprocess(self, inference_result: InferenceType) -> OutputType:
        # Your implementation
        pass
```

### Adding PipelineCore to Project Dependencies

#### Using pyproject.toml

```toml
[project]
dependencies = [
    "pipelinecore @ git+https://github.com/ggggggg13289AI/pipelinecore.git@v0.1.0",
    # Other dependencies...
]

# Or if installing from local path
[tool.uv.sources]
pipelinecore = { path = "../pipelinecore", editable = true }
```

#### Using requirements.txt

```txt
# Install from Git
pipelinecore @ git+https://github.com/ggggggg13289AI/pipelinecore.git@v0.1.0

# Or install from local path (editable mode)
-e /path/to/pipelinecore

# Or install from local path (standard installation)
/path/to/pipelinecore
```

### Verify Installation

```bash
# Check if successfully installed
python -c "import pipelinecore; print(pipelinecore.__version__)"

# Or check using uv/pip
uv pip list | grep pipelinecore
# or
pip list | grep pipelinecore
```

### Troubleshooting

#### Issue 1: Module Not Found

```bash
# Confirm pipelinecore is installed
uv pip list | grep pipelinecore

# Reinstall
uv pip uninstall pipelinecore
uv pip install -e /path/to/pipelinecore
```

#### Issue 2: Dependency Conflicts

```bash
# View dependency tree
uv pip tree

# Or upgrade related packages
uv pip install --upgrade tensorflow pynvml
```

#### Issue 3: Permission Issues

```bash
# Use virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate  # Windows

# Then install
uv pip install -e /path/to/pipelinecore
```
