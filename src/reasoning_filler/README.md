# 轨迹推理填充系统

为标注的轨迹数据补充推理过程，通过调用 Qwen 模型生成从观察到动作的推理内容。

## 🎯 功能

- ✅ 读取 `data/annotate/trajectories` 目录下的标注轨迹
- ✅ 使用 Qwen VL 模型生成推理过程
- ✅ 填充到 `Action["raw_prediction"]` 字段
- ✅ 保存到 `data/annotate_with_reasoning` 目录
- ✅ 支持断点重续
- ✅ 完善的错误处理和进度跟踪

## 📦 模块结构

```
src/reasoning_filler/
├── __init__.py                 # 包初始化
├── main.py                     # 主程序入口
├── trajectory_loader.py        # 轨迹读取模块
├── prompt_builder.py           # Prompt 构造模块
├── qwen_caller.py             # Qwen API 调用模块
├── trajectory_filler.py       # 轨迹填充与保存模块
├── progress_tracker.py        # 断点重续管理模块
├── config.yaml                # 配置文件
└── README.md                  # 本文档
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install dashscope pillow pyyaml
```

### 2. 配置 API Key

方式一：设置环境变量
```bash
export DASHSCOPE_API_KEY="your-api-key"
```

方式二：修改 `config.yaml`
```yaml
qwen:
  api_key: "your-api-key"
```

### 3. 运行

处理所有轨迹：
```bash
python -m src.reasoning_filler.main
```

只处理 classifieds 环境：
```bash
python -m src.reasoning_filler.main --env classifieds
```

处理前 10 个轨迹（测试用）：
```bash
python -m src.reasoning_filler.main --max_count 10
```

重置进度并重新处理：
```bash
python -m src.reasoning_filler.main --reset_progress
```

## 📋 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--api_key` | DashScope API Key | 从环境变量读取 |
| `--model` | Qwen 模型名称 | qwen-vl-plus |
| `--input_dir` | 输入目录 | data/annotate/trajectories |
| `--output_dir` | 输出目录 | data/annotate_with_reasoning |
| `--env` | 环境过滤 | None（所有环境） |
| `--max_count` | 最大处理数量 | None（全部） |
| `--prompt_style` | Prompt 风格 | LIFT |
| `--reset_progress` | 重置进度 | False |

## 📊 输出结构

```
data/annotate_with_reasoning/
├── classifieds/
│   ├── classifieds_116_*.pkl.xz          # 填充后的轨迹
│   └── classifieds_116_*_metadata.json   # 元数据
├── reddit/
├── shopping/
└── progress.json                          # 处理进度
```

## 🔄 断点重续

系统自动保存处理进度到 `progress.json`，支持：
- ✅ 记录已处理的文件
- ✅ 记录失败的文件和错误原因
- ✅ 统计信息（总处理数、失败数、填充动作数）
- ✅ 从中断处继续处理

## 🧪 测试单个模块

测试轨迹加载：
```bash
python src/reasoning_filler/trajectory_loader.py
```

测试 Prompt 构建：
```bash
python src/reasoning_filler/prompt_builder.py
```

测试进度跟踪：
```bash
python src/reasoning_filler/progress_tracker.py
```

## 📝 工作流程

1. **扫描轨迹** - 扫描输入目录，获取所有 `.pkl.xz` 文件
2. **加载轨迹** - 读取轨迹数据和元数据
3. **遍历动作** - 对每个 Action：
   - 提取当前状态截图
   - 构建 prompt（包含任务、历史、ground truth 动作）
   - 调用 Qwen API 生成推理
   - 填充到 `raw_prediction` 字段
4. **保存结果** - 保存填充后的轨迹到输出目录
5. **更新进度** - 记录处理状态

## ⚙️ 配置说明

编辑 `config.yaml` 修改配置：

```yaml
# API 配置
qwen:
  api_key: "your-key"
  model_name: "qwen-vl-plus"  # 或 qwen-vl-max

# 路径配置
paths:
  input_dir: "data/annotate/trajectories"
  output_dir: "data/annotate_with_reasoning"

# Prompt 配置
prompt:
  style: "LIFT"  # 或 ORIGINAL

# 处理配置
processing:
  save_frequency: 5  # 每 5 个文件保存一次进度
  skip_if_processed: true
```

## 🐛 故障排除

**问题：API 调用失败**
- 检查 API key 是否正确
- 检查网络连接
- 检查 DashScope 服务状态

**问题：内存不足**
- 减小 `max_count` 参数
- 分批处理不同环境

**问题：进度丢失**
- 检查 `progress.json` 文件是否存在
- 确保有写入权限

## 📖 代码示例

```python
from src.reasoning_filler.trajectory_loader import TrajectoryLoader
from src.reasoning_filler.qwen_caller import QwenCaller
from src.reasoning_filler.prompt_builder import PromptBuilder

# 初始化
loader = TrajectoryLoader()
qwen = QwenCaller(api_key="your-key")
builder = PromptBuilder(prompt_style="LIFT")

# 加载轨迹
trajectories = loader.scan_trajectories(env_filter="classifieds")
trajectory, metadata = loader.load_trajectory(trajectories[0])

# 处理...
```

## 📄 License

MIT License

## 👥 Authors

LIFT Team
