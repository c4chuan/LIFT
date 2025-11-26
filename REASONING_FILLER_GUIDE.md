# 轨迹推理填充系统 - 使用指南

## 📖 系统概述

该系统用于为已标注的轨迹数据补充推理过程。给定任务目标、历史信息和 ground truth 动作，通过调用 Qwen-VL 模型生成从观察到动作的详细推理过程，并填充到 `Action["raw_prediction"]` 字段中。

## 🎯 主要功能

- ✅ **自动读取轨迹**: 扫描 `data/annotate/trajectories` 目录下的所有轨迹文件
- ✅ **智能 Prompt 构建**: 基于任务目标、当前状态、历史动作和 ground truth 动作构建 prompt
- ✅ **API 调用**: 使用 DashScope API 调用 Qwen-VL 模型生成推理
- ✅ **自动填充**: 将生成的推理填充到 Action 的 raw_prediction 字段
- ✅ **结果保存**: 保存到 `data/annotate_with_reasoning` 目录，按环境分类
- ✅ **断点重续**: 支持从中断处继续，避免重复处理
- ✅ **错误处理**: 完善的重试机制和错误记录
- ✅ **Action 验证**: 自动验证生成的 action 是否与标注一致
- ✅ **自动纠正**: 验证失败时自动重试，直接告知正确答案要求重新生成
- ✅ **质量保证**: 只保存验证通过的推理，确保数据集质量

## 🔍 Action 验证与纠正机制

### 验证流程

系统会自动验证每个生成的 reasoning 中的 action 是否与标注的 ground truth action 一致：

1. **第1次尝试**: 模型根据提示词生成详细推理过程
2. **验证**: 从生成的 reasoning 中提取 `<action>` 标签内容，与 ground truth 精确匹配
3. **纠正（如需要）**:
   - 如果验证失败，直接告知模型正确的 action
   - 要求模型重新生成推理过程，自然地引导到正确的 action
   - 最多重试 2 次（总共 3 次机会）
4. **最终处理**:
   - ✅ **验证通过**: 保存 reasoning 到 `raw_prediction` 字段，加入历史上下文
   - ❌ **验证失败**: 清空 `raw_prediction` 字段，不保存，不加入历史

### 验证统计

系统会自动跟踪验证统计信息：
- `validation_passed`: 验证通过的 action 数量
- `validation_failed`: 验证失败的 action 数量
- `passed_first_attempt`: 第 1 次就成功的数量
- `corrected_first_retry`: 第 2 次成功（第 1 次重试）的数量
- `corrected_second_retry`: 第 3 次成功（第 2 次重试）的数量
- `total_api_calls`: 总 API 调用次数

### 配置验证功能

在 `config.yaml` 中配置验证参数：

```yaml
# 验证配置
validation:
  # 是否启用 action 验证
  enabled: true

  # 最大重试次数（总共 1+max_retry_attempts 次尝试）
  max_retry_attempts: 2

  # 是否使用严格匹配（完全精确匹配）
  strict_matching: true
```

或通过命令行参数控制：

```bash
# 启用验证（默认）
python -m src.reasoning_filler.main

# 禁用验证
python -m src.reasoning_filler.main --disable_validation

# 设置最大重试次数
python -m src.reasoning_filler.main --max_retry_attempts 3
```

## 🚀 快速开始

### 方式一：使用启动脚本（推荐）

Windows PowerShell:
```powershell
.\run_reasoning_filler.ps1
```

启动脚本提供交互式菜单，包含：
1. 运行测试（不调用 API）
2. 运行测试（调用 API）
3. 处理单个环境
4. 处理前 N 个轨迹（测试用）
5. 批量处理所有轨迹
6. 查看进度
7. 重置进度

### 方式二：直接运行 Python 脚本

```bash
# 设置 API Key
export DASHSCOPE_API_KEY="your-api-key"

# 运行测试
python -m src.reasoning_filler.test_single

# 处理所有轨迹
python -m src.reasoning_filler.main

# 只处理 classifieds 环境
python -m src.reasoning_filler.main --env classifieds

# 处理前 10 个轨迹（测试）
python -m src.reasoning_filler.main --max_count 10
```

## 📋 详细步骤

### 1. 环境准备

#### 安装依赖
```bash
pip install dashscope pillow pyyaml
```

#### 配置 API Key

方式一：环境变量（推荐）
```bash
# Linux/Mac
export DASHSCOPE_API_KEY="your-api-key"

# Windows PowerShell
$env:DASHSCOPE_API_KEY="your-api-key"
```

方式二：修改配置文件
编辑 `src/reasoning_filler/config.yaml`:
```yaml
qwen:
  api_key: "your-api-key"
```

### 2. 运行测试

**测试系统功能（不调用 API）**
```bash
python -m src.reasoning_filler.test_single
```

**完整测试（包含 API 调用）**
```bash
python -m src.reasoning_filler.test_single --no-dry-run
```

测试会验证：
- ✅ 轨迹加载功能
- ✅ Prompt 构建功能
- ✅ Qwen API 调用功能
- ✅ 完整处理流程

### 3. 处理轨迹

**处理所有轨迹**
```bash
python -m src.reasoning_filler.main
```

**指定环境**
```bash
# 只处理 classifieds
python -m src.reasoning_filler.main --env classifieds

# 只处理 reddit
python -m src.reasoning_filler.main --env reddit

# 只处理 shopping
python -m src.reasoning_filler.main --env shopping
```

**限制处理数量**
```bash
# 只处理前 5 个轨迹（用于测试）
python -m src.reasoning_filler.main --max_count 5

# 处理 classifieds 的前 10 个
python -m src.reasoning_filler.main --env classifieds --max_count 10
```

**重置进度**
```bash
python -m src.reasoning_filler.main --reset_progress
```

### 4. 查看进度

进度文件位于 `data/annotate_with_reasoning/progress.json`

```bash
# Linux/Mac
cat data/annotate_with_reasoning/progress.json | jq

# Windows PowerShell
Get-Content data/annotate_with_reasoning/progress.json | ConvertFrom-Json
```

## 📊 输出结构

```
data/
├── annotate/
│   └── trajectories/              # 输入：原始标注轨迹
│       ├── classifieds/
│       │   ├── *.pkl.xz
│       │   └── *_metadata.json
│       ├── reddit/
│       └── shopping/
│
└── annotate_with_reasoning/       # 输出：填充推理后的轨迹
    ├── classifieds/
    │   ├── *.pkl.xz              # 填充后的轨迹（含 raw_prediction）
    │   └── *_metadata.json       # 元数据（复制自原始）
    ├── reddit/
    ├── shopping/
    └── progress.json              # 处理进度
```

## 🔄 处理流程详解

### 整体流程

```
1. 扫描轨迹文件
   ↓
2. 检查进度（跳过已处理）
   ↓
3. 加载轨迹和元数据
   ↓
4. 遍历 Trajectory（StateInfo 和 Action 交替）
   ↓
5. 对每个 Action:
   a. 提取当前状态截图
   b. 构建 Prompt
   c. 调用 Qwen API
   d. 填充 raw_prediction
   ↓
6. 保存填充后的轨迹
   ↓
7. 更新进度
```

### Prompt 构建

对于每个 Action，构建的 Prompt 包含：

```
System Prompt:
- 说明任务是生成推理过程
- 定义输出格式（LIFT 或 ORIGINAL 风格）

User Prompt:
- 任务目标 (intent)
- 当前页面 URL
- 当前页面截图
- 历史动作列表
- Ground truth 动作
- 任务相关的输入图片（如果有）
```

### 推理生成示例

**输入:**
- Intent: "搜索 'laptop' 并找到价格最便宜的商品"
- 当前页面: 搜索结果页
- 历史: `type [search_box] [laptop] [1]`
- Ground truth: `click [product_123]`

**输出 (LIFT 风格):**
```
Let's observe step by step.
<zoom in>
当前页面显示了搜索结果列表，包含多个 laptop 产品...
</zoom in>

<shift>
观察价格信息，发现产品 123 的价格是 $299，
是列表中最便宜的...
</shift>

<summary>
根据任务目标，需要找到最便宜的商品。
通过观察发现产品 123 价格最低。
因此，下一步动作是 ```click [123]```
</summary>
```

## ⚙️ 配置说明

编辑 `src/reasoning_filler/config.yaml`:

```yaml
# API 配置
qwen:
  api_key: "your-key"          # API Key
  model_name: "qwen-vl-plus"   # 模型: qwen-vl-plus 或 qwen-vl-max
  max_retries: 3               # 最大重试次数
  retry_delay: 2.0             # 重试延迟（秒）
  timeout: 60                  # 请求超时（秒）

# 路径配置
paths:
  input_dir: "data/annotate/trajectories"
  output_dir: "data/annotate_with_reasoning"

# Prompt 配置
prompt:
  style: "LIFT"  # LIFT 或 ORIGINAL

# 处理配置
processing:
  save_frequency: 5  # 每处理 5 个文件保存一次进度
  skip_if_processed: true

# 验证配置
validation:
  enabled: true              # 是否启用 action 验证
  max_retry_attempts: 2      # 最大重试次数（总共3次机会）
  strict_matching: true      # 严格匹配模式

# 提示词增强配置
prompt_enhancement:
  require_comprehensive_observation: true  # 要求全面观察
  min_observation_steps: 5                 # 最少观察步骤数
  require_phase_structure: true            # 要求5阶段结构
```

## 📝 提示词增强：5阶段观察结构

系统现在要求模型按照 5 个阶段进行全面的页面观察，确保生成的推理过程覆盖更多页面内容：

### 5 个观察阶段

**Phase 1 - Global Layout（全局布局）**
- 使用 `<zoom in>` 观察整体页面布局
- 识别所有主要功能区域（导航栏、侧边栏、主内容区、底部等）

**Phase 2 - Area Exploration（区域探索）**
- 使用 `<shift>` 探索页面上的不同区域
- 包括与任务相关和看似无关的信息
- 检查页面元素的分布和组织方式

**Phase 3 - Element Comparison（元素对比）**
- 观察多个可能的候选元素
- 对比分析它们的特征
- 解释为什么某些元素不合适

**Phase 4 - Target Focus（目标聚焦）**
- 使用 `<zoom in>` 聚焦到目标元素
- 仔细观察其边框颜色以确定 som_id
- 检查其属性和状态

**Phase 5 - Decision Justification（决策论证）**
- 总结为什么这个特定元素是最佳选择
- 解释为什么不选择页面上的其他元素

### 输出示例

```
Let's observe step by step. First, I will zoom in to observe the overall page structure.

<zoom in>
Phase 1 - Global Layout: The page is divided into three main areas:
a top navigation bar, a left sidebar with filters, and a main content
area displaying product listings...
</zoom in>

<shift>
Phase 2 - Area Exploration: Examining the sidebar, I see multiple
filter options including category, price range, and location. The
navigation bar contains search, cart, and account links...
</shift>

<shift>
Phase 3 - Element Comparison: There are several clickable elements
visible. Element [5] is a category filter, element [7] is a price
filter, and element [12] is a search button. For our task of filtering
by location, elements [5] and [12] are not suitable...
</shift>

<zoom in>
Phase 4 - Target Focus: Focusing on the location filter field, I
observe its border has a specific color. By examining the border color,
I can identify this as element [7]...
</zoom in>

<summary>
Phase 5 - Decision Justification:
Based on the comprehensive observation:
1. The page layout shows clear filter options in the left sidebar
2. Multiple filter types are available but only location filter matches our need
3. Element [7] is the location input field, confirmed by border color
4. This element is the correct choice as it directly supports our goal

So the next action I will perform is type [7] [Washington] [0]
</summary>

<action>
type [7] [Washington] [0]
</action>
```

这种结构化的观察方式确保：
- ✅ 模型全面理解页面内容
- ✅ 推理过程展示完整的分析思路
- ✅ 最终action有充分的论证支持
- ✅ 提高推理质量和可解释性

## 🐛 故障排除

### API 调用失败

**问题**: `API 调用失败 (status: 400)`

**解决方案**:
1. 检查 API key 是否正确
2. 检查账户余额
3. 检查模型名称是否正确

**问题**: `API 调用超时`

**解决方案**:
1. 检查网络连接
2. 增加 `timeout` 配置
3. 减小图片尺寸

### 内存问题

**问题**: 内存不足

**解决方案**:
1. 使用 `--max_count` 参数分批处理
2. 分别处理不同环境
3. 减小批处理大小

### 进度丢失

**问题**: 进度文件损坏或丢失

**解决方案**:
1. 检查 `data/annotate_with_reasoning/progress.json`
2. 如果损坏，删除并重新开始
3. 使用 `--reset_progress` 重置

## 📝 高级用法

### 命令行参数完整列表

```bash
python -m src.reasoning_filler.main \
  --api_key "your-key" \           # API Key
  --model "qwen-vl-plus" \         # 模型名称
  --input_dir "path/to/input" \    # 输入目录
  --output_dir "path/to/output" \  # 输出目录
  --env "classifieds" \            # 环境过滤
  --max_count 10 \                 # 最大处理数量
  --prompt_style "LIFT" \          # Prompt 风格
  --reset_progress                 # 重置进度
```

### 编程方式使用

```python
from src.reasoning_filler.trajectory_loader import TrajectoryLoader
from src.reasoning_filler.qwen_caller import QwenCaller
from src.reasoning_filler.prompt_builder import PromptBuilder
from src.reasoning_filler.trajectory_filler import TrajectoryFiller

# 初始化
loader = TrajectoryLoader()
qwen = QwenCaller(api_key="your-key")
builder = PromptBuilder(prompt_style="LIFT")
filler = TrajectoryFiller(qwen, builder)

# 加载轨迹
trajectories = loader.scan_trajectories(env_filter="classifieds")
trajectory, metadata = loader.load_trajectory(trajectories[0])

# 填充推理
filled_trajectory, num_filled = filler.fill_trajectory(trajectory, metadata)

# 保存结果
filler.save_filled_trajectory(
    filled_trajectory,
    metadata,
    "classifieds",
    "task_1.pkl.xz"
)
```

## 📞 支持

如有问题，请：
1. 查看 `src/reasoning_filler/README.md`
2. 运行测试脚本诊断问题
3. 检查进度文件中的错误信息
4. 联系开发团队

## 📄 相关文档

- [README.md](src/reasoning_filler/README.md) - 模块文档
- [config.yaml](src/reasoning_filler/config.yaml) - 配置文件
- [test_single.py](src/reasoning_filler/test_single.py) - 测试脚本

## 🎉 完成

处理完成后：
1. 检查输出目录 `data/annotate_with_reasoning`
2. 查看进度摘要
3. 验证填充的轨迹数据

祝使用愉快！
