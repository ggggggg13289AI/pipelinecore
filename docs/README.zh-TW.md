# PipelineCore

**繁體中文** | [English](./README.md)

一個用於建構機器學習與推論管線（Pipeline）的 Python 框架，具備 GPU 資源管理與 TensorFlow 整合功能。

## 功能特色

- **模板方法模式（Template Method Pattern）**：可擴展的管線架構，配備生命週期鉤子（Lifecycle Hooks）
- **GPU 資源管理**：自動檢查 GPU 可用性與記憶體管理
- **型別安全泛型**：使用泛型型別參數實現端到端的型別安全
- **TensorFlow 整合**：與 TensorFlow 無縫整合，支援機器學習工作負載
- **結構化日誌**：內建管線執行的日誌基礎設施

## 安裝

### 使用 uv（推薦）

```bash
# 複製儲存庫
git clone https://github.com/ggggggg13289AI/pipelinecore
cd pipelinecore

# 使用 uv 安裝
uv sync

# 啟動虛擬環境
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows
```

### 使用 pip

```bash
pip install pipelinecore
```

## 快速入門

### 基本管線範例

```python
from pipelinecore.core import BasePipeline, PipelineContext, PipelinePaths, LogManager
from pathlib import Path

# 使用型別參數定義您的管線
class MyPipeline(BasePipeline[str, dict, list, str]):
    def prepare(self, payload: str) -> dict:
        """前處理輸入資料"""
        return {"data": payload.split()}

    def run_inference(self, prepared: dict) -> list:
        """執行模型推論"""
        return [word.upper() for word in prepared["data"]]

    def postprocess(self, inference_result: list) -> str:
        """格式化輸出結果"""
        return " ".join(inference_result)

# 建立管線上下文
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

# 執行管線
pipeline = MyPipeline(context)
result = pipeline.execute("hello world")
print(result)  # 輸出：HELLO WORLD
```

### 啟用 GPU 的管線

```python
from pipelinecore.core import (
    BasePipeline,
    TensorflowPipelineMixin,
    GpuResourceManager,
    PipelineContext
)

class GpuPipeline(TensorflowPipelineMixin, BasePipeline[Input, Prepared, Inference, Output]):
    def __init__(self, context: PipelineContext):
        # 配置 GPU 資源管理器
        gpu_manager = GpuResourceManager(
            gpu_index=0,
            usage_threshold=0.9,
            usage_check_enabled=True
        )

        TensorflowPipelineMixin.__init__(self, gpu_manager)
        BasePipeline.__init__(self, context)

    def prepare(self, payload: Input) -> Prepared:
        # 您的前處理邏輯
        pass

    def run_inference(self, prepared: Prepared) -> Inference:
        # GPU 加速的 TensorFlow 推論
        # 此方法執行前會自動確保 GPU 可用性
        pass

    def postprocess(self, inference_result: Inference) -> Output:
        # 您的後處理邏輯
        pass
```

## 架構

### 核心元件

#### BasePipeline（基礎管線）

所有管線的基礎，實作模板方法模式（Template Method Pattern）：

```python
BasePipeline[InputT, PreparedT, InferenceT, OutputT]
```

**型別參數：**
- `InputT`：原始輸入資料型別
- `PreparedT`：前處理後的資料型別
- `InferenceT`：模型推論輸出型別
- `OutputT`：最終處理輸出型別

**生命週期方法：**
1. `before_run()` - 執行前鉤子
2. `prepare(payload)` - 轉換輸入 → 準備好的資料
3. `run_inference(prepared)` - 執行模型推論
4. `postprocess(inference)` - 轉換推論結果 → 輸出
5. `after_run(result)` - 執行後鉤子

#### GpuResourceManager（GPU 資源管理器）

管理 GPU 可用性與 TensorFlow 裝置配置：

- 透過 NVIDIA 管理函式庫（pynvml）監控 GPU 記憶體使用量
- 自動配置裝置可見性
- 啟用記憶體增長功能
- 可配置的 GPU 可用性重試邏輯

#### TensorflowPipelineMixin（TensorFlow 管線混入類別）

適用於依賴 GPU 的管線的混入類別（Mixin）：

- 將 GpuResourceManager 整合到管線生命週期中
- 在推論前確保 GPU 可用性
- 處理 TensorFlow 裝置配置

#### PipelineContext（管線上下文）

管線執行的共享執行時上下文：

```python
@dataclass(frozen=True)
class PipelineContext:
    pipeline_id: str
    paths: PipelinePaths
    logger: logging.Logger
```

## 開發

### 需求

- Python >=3.10
- TensorFlow >=2.10.0
- pynvml >=11.0.0

### 設置開發環境

```bash
# 安裝開發依賴項
uv sync

# 執行測試
pytest

# 執行程式碼檢查
ruff check src/

# 型別檢查
mypy src/
```

### 測試

```bash
# 執行所有測試
pytest

# 執行測試並產生覆蓋率報告
pytest --cov=src/pipelinecore --cov-report=html

# 執行特定測試檔案
pytest tests/test_base.py
```

### 程式碼品質

```bash
# 格式化程式碼
ruff format src/

# 檢查程式碼
ruff check src/ --fix

# 型別檢查
mypy src/
```

## 專案結構

```
pipelinecore/
├── src/pipelinecore/
│   └── core/
│       ├── __init__.py       # 公開 API 匯出
│       └── base.py           # 核心管線類別
├── tests/                    # 測試套件
├── docs/                     # 文件
│   ├── README.md            # 英文文件
│   └── README.zh-TW.md      # 繁體中文文件
├── pyproject.toml           # 專案配置
└── CLAUDE.md                # Claude Code 指引
```

## 授權條款

MIT License

## 貢獻

歡迎貢獻！請隨時提交 Pull Request。

## 連結

- **儲存庫**：https://github.com/user/pipelinecore
- **問題追蹤**：https://github.com/user/pipelinecore/issues
- **文件**：https://github.com/user/pipelinecore/blob/main/docs/README.md
