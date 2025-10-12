# JSONL 数据读取工具使用说明

## 功能介绍

`read_jsonl.py` 是一个用于读取和分析 `sup_rollout_data_dir/1.jsonl` 训练数据文件的工具脚本。

### 主要功能

1. **读取 JSONL 文件**: 解析每一行的 JSON 数据
2. **数据分析**: 统计数据集的基本信息(总数、评分、步骤等)
3. **解析条目**: 提取每条数据的关键信息(URL、目标任务、动作等)
4. **保存图片**: 将 base64 编码的截图保存为 PNG 文件

## 使用方法

### 1. 显示帮助信息

```bash
python3 read_jsonl.py --help
```

### 2. 分析数据集统计信息

```bash
python3 read_jsonl.py -a
```

输出示例:
```
=== Dataset Statistics ===
Total entries: 16
Step range: 1 - 1
Score range: 0.00 - 2.00
Average score: 0.56
Total images: 32
Average images per entry: 2.00
```

### 3. 解析并显示数据详情

```bash
# 解析所有数据
python3 read_jsonl.py -p

# 只解析前 5 条数据
python3 read_jsonl.py -p -l 5
```

### 4. 保存图片

```bash
# 保存所有图片到 images 目录
python3 read_jsonl.py -s images

# 只保存前 3 条数据的图片
python3 read_jsonl.py -s images -l 3
```

### 5. 组合使用

```bash
# 同时分析和解析数据
python3 read_jsonl.py -a -p -l 10

# 分析、解析并保存图片
python3 read_jsonl.py -a -p -s output_images
```

### 6. 指定不同的文件

```bash
python3 read_jsonl.py -f path/to/other.jsonl -a
```

## 数据结构说明

每条 JSON 数据包含以下字段:

- **input**: 系统提示词和用户任务描述
  - 包含 URL、OBJECTIVE、PREVIOUS ACTION 等信息
- **output**: 模型的响应和动作
  - 包含观察、推理和最终的 action
- **score**: 该动作的评分 (0.0-2.0)
- **step**: 当前步骤编号
- **images**: base64 编码的截图列表

## 代码示例

### 在自己的脚本中使用

```python
from read_jsonl import read_jsonl, parse_entry, analyze_data

# 读取数据
data = read_jsonl('sup_rollout_data_dir/1.jsonl')

# 分析统计信息
analyze_data(data)

# 解析单条数据
for entry in data:
    parsed = parse_entry(entry)
    print(f"Objective: {parsed.get('objective')}")
    print(f"Action: {parsed.get('action')}")
    print(f"Score: {parsed['score']}")
```

## 注意事项

1. 确保文件路径正确
2. 图片保存需要足够的磁盘空间
3. 使用 `-l` 参数可以限制处理的数据量,适合快速预览
