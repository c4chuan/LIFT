# 测试数据生成器

这个目录包含用于替换测试配置文件中占位符的脚本。

## 文件说明

- **`generate_test_data.py`** - 主脚本，支持三种数据集类型
- **`config_example.json`** - 示例配置文件
- **`README.md`** - 此说明文件

## 快速使用

### 对于 data/annotate 目录的文件

```bash
# 进入脚本目录
cd src/scripts

# 运行脚本替换占位符
python generate_test_data.py --config config_example.json --dataset annotate --input-dir ../../data --verbose
```

这将处理以下文件：
- `data/annotate/shopping_tasks.json` - 替换 `__SHOPPING__` 占位符
- `data/annotate/reddit_tasks.json` - 替换 `__REDDIT__` 和 `__SHOPPING__` 占位符
- `data/annotate/classifieds_tasks.json` - 替换 `__CLASSIFIEDS__` 和 `__SHOPPING__` 占位符

### 其他数据集

```bash
# WebArena 数据集
python generate_test_data.py --config config_example.json --dataset webarena --input-dir config_files

# VisualWebArena 数据集
python generate_test_data.py --config config_example.json --dataset visualwebarena --input-dir config_files
```

## 配置文件格式

配置文件应包含各数据集的网站映射：

```json
{
  "annotate": {
    "REDDIT": "http://127.0.0.1:9999",
    "SHOPPING": "http://127.0.0.1:7770",
    "CLASSIFIEDS": "http://127.0.0.1:9980"
  }
}
```

## 参数说明

- `--config/-c`: 配置文件路径（必需）
- `--dataset/-d`: 数据集类型（webarena/visualwebarena/annotate）
- `--input-dir/-i`: 输入目录（默认：data）
- `--verbose/-v`: 详细输出模式