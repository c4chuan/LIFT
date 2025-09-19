# Rewarder 多卡加速使用指南

## 概述

原有的 `Rewarder` 类现在支持使用 `accelerate` 库进行多卡张量并行，无需创建新类，完全向后兼容。

## 修改内容

1. **在 `Rewarder.__init__()` 中添加了 `use_accelerate=False` 参数**
2. **所有子类（`ChunkRewarder`, `ContainDisRewarder`, `ContainRewarder`）自动继承此功能**
3. **保持完全向后兼容，默认行为不变**

## 使用方法

### 1. 单卡模式（默认，无变化）

```python
from src.reward.rewarder import Rewarder, ChunkRewarder, ContainDisRewarder

# 原有用法完全不变
rewarder = Rewarder()
rewarder = ChunkRewarder(chunk_size=7200)
rewarder = ContainDisRewarder()
```

### 2. 多卡模式（新增功能）

```python
# 启用 accelerate 多卡模式
rewarder = Rewarder(use_accelerate=True)
rewarder = ChunkRewarder(chunk_size=7200, use_accelerate=True)
rewarder = ContainDisRewarder(use_accelerate=True)
```

### 3. 命令行启动

#### 单卡模式
```bash
python src/reward/rewarder.py
```

#### 多卡模式
```bash
accelerate launch --multi_gpu --num_processes=4 src/reward/rewarder.py --accelerate
```

## 安装依赖

```bash
pip install accelerate
```

首次使用需要配置：
```bash
accelerate config
```

## 技术实现

- **自动设备管理**：使用 accelerate 的 device_map="auto"
- **混合精度**：多卡模式自动启用 fp16 减少显存占用
- **分布式 attention 收集**：自动收集各 GPU 的注意力权重
- **主进程模式**：只在主进程计算最终 reward 和可视化，避免重复

## 性能优势

- **显存优化**：模型权重分布到多张卡，单卡显存占用降低
- **计算加速**：并行 attention 计算，提升推理速度
- **向后兼容**：原有代码无需修改即可享受性能提升

## 注意事项

1. 确保所有进程都能访问相同的模型文件和数据文件
2. 多卡模式下会自动使用 fp16，可能略微影响精度
3. 只有主进程会输出结果和保存可视化文件