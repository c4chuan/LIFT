# 轨迹查看器使用说明

## 概述

轨迹查看器 (`trajectory_viewer.py`) 是一个用于查看、分析和管理已保存的标注轨迹的工具。

## 功能特性

- 📄 **轨迹列表查看** - 显示所有保存的轨迹文件
- 🔍 **轨迹详情查看** - 查看单个轨迹的详细信息
- 📊 **统计分析** - 显示轨迹统计信息和成功率
- 🔎 **搜索过滤** - 按环境、分数、关键词等条件搜索轨迹
- 📤 **数据导出** - 导出轨迹数据为JSON、CSV或统计报告

## 使用方法

### 1. 交互式模式（推荐）

```bash
python trajectory_viewer.py
```

启动后会显示主菜单，根据提示选择功能：

```
📋 主菜单:
  1. 列出所有轨迹
  2. 查看轨迹详情
  3. 显示统计信息
  4. 搜索轨迹
  5. 导出轨迹数据
  q. 退出
```

### 2. 命令行模式

#### 快速列出轨迹
```bash
# 列出最新的10个轨迹
python trajectory_viewer.py --list

# 列出classifieds环境的20个轨迹
python trajectory_viewer.py --list --env classifieds --limit 20
```

#### 显示统计信息
```bash
python trajectory_viewer.py --stats
```

#### 指定数据目录
```bash
python trajectory_viewer.py --annotate-dir /path/to/your/annotate/dir --list
```

## 轨迹文件结构

标注系统为每个完成的任务生成3个文件：

```
data/annotate/trajectories/
├── classifieds/
│   ├── classifieds_4_20250926_163653.pkl.xz      # 二进制轨迹数据
│   ├── classifieds_4_20250926_163653_metadata.json  # 元数据(JSON格式)
│   └── classifieds_4_20250926_163653_script.py      # Playwright回放脚本
├── reddit/
└── shopping/
```

### 文件说明

1. **`.pkl.xz` 文件** - 压缩的pickle格式，包含完整的轨迹数据
2. **`_metadata.json` 文件** - JSON格式元数据，包含：
   - 任务ID和环境信息
   - 任务描述和起始URL
   - 评估分数和轨迹长度
   - 标注时间戳
3. **`_script.py` 文件** - Playwright回放脚本（人类可读）

## 功能详解

### 轨迹列表查看

显示轨迹的基本信息：
- 环境类型（CLASSIFIEDS/REDDIT/SHOPPING）
- 任务ID
- 评估分数（带emoji指示器）
- 轨迹步数
- 标注时间
- 任务描述

### 轨迹详情查看

显示单个轨迹的完整信息：
- 任务详细描述
- 起始URL和登录需求
- 评估分数和轨迹长度
- 相关图片信息
- 文件路径信息

### 统计分析

提供以下统计信息：
- 总轨迹数量
- 平均轨迹长度
- 平均评估分数
- 各环境的详细统计

### 搜索功能

支持多条件搜索：
- **环境过滤** - 按classifieds/reddit/shopping筛选
- **分数过滤** - 设置最低分数阈值
- **关键词搜索** - 在任务描述中搜索关键词

### 数据导出

支持多种导出格式：

1. **JSON格式** - 完整的轨迹元数据
2. **CSV格式** - 轨迹基本信息表格
3. **统计报告** - 详细的分析报告（TXT格式）

## 示例用法

### 查看所有高分轨迹
```bash
# 交互模式下选择"搜索轨迹"，然后输入最低分数 0.8
python trajectory_viewer.py
```

### 查找特定任务
```bash
# 搜索包含"price"关键词的任务
# 在搜索界面输入关键词: price
```

### 导出分析数据
```bash
# 在交互模式下选择"导出轨迹数据"
# 选择JSON格式导出所有轨迹元数据
```

## 注意事项

1. 确保 `data/annotate` 目录存在且包含轨迹文件
2. 轨迹查看器只读取数据，不会修改任何文件
3. 大量轨迹文件可能需要更长的加载时间
4. 导出功能会在当前目录创建文件

## 故障排除

### 常见问题

**Q: 提示"没有找到轨迹文件"**
A: 检查 `data/annotate/trajectories/` 目录是否存在且包含 `*_metadata.json` 文件

**Q: 无法加载轨迹数据**
A: 确保对应的 `.pkl.xz` 文件存在且未损坏

**Q: 时间显示异常**
A: 可能是时间戳格式问题，轨迹查看器会尝试自动处理

## 扩展功能

轨迹查看器的代码结构便于扩展，可以轻松添加：
- 轨迹回放功能
- 可视化图表
- 更多导出格式
- 轨迹比较分析