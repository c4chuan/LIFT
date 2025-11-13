# Reward Top Analyzer 使用说明

## 功能介绍

`reward_top_analyzer.py` 是一个用于分析rollout训练数据的工具，可以找出各种奖励指标的top5样本并生成HTML可视化报告。

## 主要功能

1. **数据收集与过滤**：支持按训练步骤范围过滤数据
2. **Top样本分析**：找出5组不同指标的top5样本
   - Zoom Reward 最高 top5
   - Zoom Reward 最低 top5（排除值为0的记录）
   - Shift Reward 最高 top5
   - Shift Reward 最低 top5
   - Total Reward (Shift + Zoom) 最高 top5
3. **数据导出**：将top样本导出为JSONL文件
4. **HTML可视化**：生成精美的HTML页面展示每个样本的详细信息

## 使用方法

### 基本用法

```bash
# 分析所有步骤的数据
python3 src/reward_top_analyzer.py \
    --input-dir sup_rollout_data_dir_1012 \
    --output-dir top_rewards_analysis
```

### 指定步骤范围

```bash
# 仅分析步骤100-200的数据
python3 src/reward_top_analyzer.py \
    --input-dir sup_rollout_data_dir_1012 \
    --output-dir top_rewards_analysis \
    --start-step 100 \
    --end-step 200
```

### 参数说明

- `--input-dir`: 输入目录（包含JSONL文件），默认：`../sup_rollout_data_dir_1012`
- `--output-dir`: 输出目录（HTML可视化结果），默认：`../top_rewards_analysis`
- `--start-step`: 起始训练步骤（包含），默认：无限制
- `--end-step`: 结束训练步骤（包含），默认：无限制
- `--html-dir`: 原始HTML文件目录（可选），默认：`../sup_rollout_data_html_dir_1012`

### 查看帮助

```bash
python3 src/reward_top_analyzer.py --help
```

## 输出结果

工具会生成以下目录结构：

```
top_rewards_analysis/
├── index.html              # 主索引页面
├── original_steps/         # 原始完整HTML文件（可选）
│   ├── 100.html           # 对应训练步骤的完整HTML
│   ├── 101.html
│   └── ...
├── zoom_highest/           # Zoom Reward 最高样本
│   ├── rank1_stepXX_recYY.html
│   ├── rank1_stepXX_recYY.jsonl
│   └── ...
├── zoom_lowest/            # Zoom Reward 最低样本（非0）
├── shift_highest/          # Shift Reward 最高样本
├── shift_lowest/           # Shift Reward 最低样本
└── total_highest/          # Total Reward 最高样本
```

## 查看结果

用浏览器打开 `top_rewards_analysis/index.html` 即可查看可视化报告。

报告包含：
- 所有top样本的汇总表格（带原始HTML链接）
- 每个样本的详细页面，包括：
  - 奖励指标（高亮显示相关指标）
  - 原始完整HTML按钮（如果提供了--html-dir）
  - 截图图片（支持点击放大）
  - 输入提示
  - 模型输出
  - 训练步骤和记录索引
- 原始HTML文件（包含对应步骤的所有rollout记录）

## 示例

### 分析最近10个步骤的数据

```bash
python3 src/reward_top_analyzer.py \
    --input-dir sup_rollout_data_dir_1012 \
    --output-dir recent_analysis \
    --start-step 460 \
    --end-step 470
```

### 分析所有数据（默认）

```bash
python3 src/reward_top_analyzer.py
```

### 包含原始HTML文件

```bash
python3 src/reward_top_analyzer.py \
    --input-dir sup_rollout_data_dir_1012 \
    --output-dir top_rewards_analysis \
    --html-dir sup_rollout_data_html_dir_1012
```

这将复制对应步骤的完整HTML文件到 `original_steps/` 目录，方便查看top样本所在步骤的完整上下文。

## 技术细节

- 使用 `heapq` 模块高效查找top样本
- 复用 `jsonl_to_html_converter.py` 的可视化模板
- 支持base64编码的图片内嵌显示
- 响应式HTML设计，支持移动端访问

## 注意事项

1. 输入目录中的JSONL文件应以数字命名（如 `100.jsonl`, `101.jsonl`），数字代表训练步骤
2. 每个JSONL文件可包含多条记录（每行一个JSON对象）
3. 每条记录应包含以下字段：
   - `input`: 输入提示
   - `output`: 模型输出
   - `score`: 总分
   - `shift_reward`: shift奖励
   - `zoom_reward`: zoom奖励
   - `images`: 截图（base64编码）

## 故障排查

### 找不到数据

确保输入目录路径正确，且包含以数字命名的JSONL文件。

### JSON解析错误

检查JSONL文件格式是否正确，每行应为独立的JSON对象。

### 图片不显示

确保 `images` 字段包含有效的base64编码图片数据。
