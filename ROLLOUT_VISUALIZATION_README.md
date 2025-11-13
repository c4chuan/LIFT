# GRPO Rollout 数据可视化工具

## 简介

这个工具可以将 `sup_rollout_data_dir/` 中的 JSONL 格式训练数据转换为美观的 HTML 可视化页面，方便查看和分析每个训练步骤的详细信息。

## 功能特性

✨ **支持的数据展示**：
- 📸 **截图图片**：显示每个步骤的网页截图（base64 编码）
- 📥 **输入提示**：显示模型的输入 prompt
- 📤 **模型输出**：显示模型生成的推理和动作
- 🏆 **奖励分数**：显示各项奖励指标

🎨 **用户体验**：
- 响应式布局，自适应不同屏幕尺寸
- 图片点击放大功能（支持 ESC 键关闭）
- 清晰的分区和配色方案
- 索引页面方便导航

## 使用方法

### 方法 1: 使用交互式脚本（推荐）

```bash
./convert_sup_rollout.sh
```

然后根据提示选择处理模式：
- **选项 1**: 处理所有文件（时间较长，适合完整转换）
- **选项 2**: 处理前 10 个文件（快速预览）
- **选项 3**: 处理指定范围（如 1-50）
- **选项 4**: 处理单个文件

### 方法 2: 直接使用 Python 工具

#### 处理所有文件
```bash
python3 src/jsonl_to_html_converter.py \
    --input-dir sup_rollout_data_dir \
    --output-dir sup_rollout_data_html_dir
```

#### 处理单个文件
```python
from pathlib import Path
from src.jsonl_to_html_converter import convert_jsonl_to_html

input_file = Path('sup_rollout_data_dir/1.jsonl')
output_dir = Path('sup_rollout_data_html_dir')
output_dir.mkdir(exist_ok=True)

convert_jsonl_to_html(input_file, output_dir)
```

## 查看结果

转换完成后，在浏览器中打开：
```
sup_rollout_data_html_dir/index.html
```

或直接打开单个文件：
```
sup_rollout_data_html_dir/1.html
```

## 数据格式

工具支持以下 JSONL 数据格式：

```json
{
    "input": "系统提示和用户输入...",
    "output": "模型输出和推理...",
    "score": 0.1557,
    "step": 1,
    "images": [
        "iVBORw0KGgoAAAANSUhEUgAA...",  // base64 编码的图片
        "iVBORw0KGgoAAAANSUhEUgAA..."
    ]
}
```

## 性能说明

- **单文件处理时间**：约 2-5 秒（取决于图片数量和大小）
- **生成的 HTML 文件大小**：约 20-30 MB（包含嵌入的 base64 图片）
- **推荐处理方式**：
  - 测试时先处理少量文件（如前 10 个）
  - 确认效果后再处理全部文件
  - 可以分批处理，避免一次性占用太多时间

## 技术细节

- **图片格式支持**：PNG、JPEG（自动检测 base64 前缀）
- **图片显示**：使用 data URI 直接嵌入 HTML
- **点击放大**：JavaScript 实现的模态框
- **兼容性**：支持所有现代浏览器

## 示例输出

每个 HTML 文件包含：
1. **页面头部**：显示步骤编号、平均分数、生成时间
2. **每个 Record**：
   - 截图图片（可点击放大）
   - 输入 prompt（高亮关键词）
   - 模型输出（格式化显示）
   - 奖励分数（分类显示）
3. **导航**：返回索引页面的链接

## 故障排除

### Q: 处理速度很慢？
A: 这是正常的。每个文件包含大量 base64 编码的图片数据，处理需要时间。建议先处理少量文件测试。

### Q: HTML 文件很大？
A: 这是正常的。因为图片数据直接嵌入 HTML，每个文件约 20-30 MB。优点是不需要额外的图片文件，方便分享和查看。

### Q: 浏览器打开很慢？
A: 第一次加载可能较慢，浏览器需要解码 base64 图片。建议使用 Chrome 或 Firefox 的最新版本。

### Q: 图片显示不正确？
A: 确保 JSONL 文件中的 images 字段包含有效的 base64 编码数据。

## 联系与支持

如有问题或建议，请参考 `src/jsonl_to_html_converter.py` 的源代码。
