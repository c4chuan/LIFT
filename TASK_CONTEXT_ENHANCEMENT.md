# 交互式标注程序任务信息持续显示功能

## 功能概述
本次改进为交互式Web任务标注工具添加了任务信息的持续显示功能，让用户在整个标注过程中始终了解当前任务的关键信息。

## 主要改进

### 1. 任务上下文跟踪
- **环境名称**: 当前任务所属环境（如 REDDIT、SHOPPING 等）
- **任务ID**: 任务的唯一标识符
- **任务描述**: 任务目标的简洁描述（长文本会自动截断）
- **URL信息**: 起始URL和当前页面URL的跟踪
- **步骤计数**: 实时记录用户执行的操作步数

### 2. 显示位置和时机

#### 2.1 用户输入提示时
每次提示用户输入操作时，都会先显示当前任务的上下文信息：
```
[当前任务] 环境: REDDIT | ID: 42 | 目标: Find blue kayak post... | 页面: reddit.com | 步骤: 3
[INPUT] 请输入操作命令 (输入 'help' 查看帮助):
```

#### 2.2 动作执行结果显示时
当用户操作成功执行后，会显示更新的进度信息：
```
[OK] 动作执行成功: 点击搜索按钮
[进度更新] 环境: REDDIT | ID: 42 | 目标: Find blue kayak post... | 页面: reddit.com | 步骤: 4
```

#### 2.3 截图显示时
当显示新的截图时，会包含任务状态提醒：
```
[任务状态] 环境: REDDIT | ID: 42 | 目标: Find blue kayak post... | 页面: reddit.com | 步骤: 3
[SCREENSHOT] 环境屏幕截图已准备就绪
```

### 3. URL状态跟踪
- **起始页面**: 当当前URL与起始URL相同时显示 "起始页面: domain.com"
- **页面变化**: 当URL发生变化时显示 "页面: domain.com"
- **域名简化**: 自动提取域名，避免显示过长的URL

### 4. 智能文本处理
- **长文本截断**: 任务描述超过50字符时自动截断并添加省略号
- **步骤计数**: 只在步骤数大于0时显示
- **空值处理**: 没有任务信息时不显示任何上下文

## 技术实现

### 修改的文件
1. **src/annotation/annotation_ui.py**: 添加任务上下文管理功能
2. **src/interactive_annotator.py**: 集成任务上下文到主程序流程

### 新增的核心方法
- `set_task_context()`: 设置/更新任务上下文信息
- `get_task_context_summary()`: 获取格式化的任务摘要
- `display_current_task_context()`: 显示当前任务上下文
- `clear_task_context()`: 清除任务上下文信息

## 使用效果示例

### 典型交互流程
```
============================================================
第1步: 初始页面加载完毕，等待用户操作
============================================================
[任务状态] 环境: REDDIT | ID: 42 | 目标: Find blue kayak post and get the price informati... | 起始页面: reddit.com
[SCREENSHOT] 环境屏幕截图已准备就绪

[当前任务] 环境: REDDIT | ID: 42 | 目标: Find blue kayak post and get the price informati... | 起始页面: reddit.com
[INPUT] 请输入操作命令 (输入 'help' 查看帮助):
>>> click [search_box]

[OK] 动作执行成功: 点击搜索框
[进度更新] 环境: REDDIT | ID: 42 | 目标: Find blue kayak post and get the price informati... | 起始页面: reddit.com | 步骤: 1
```

## 测试和验证

### 测试文件
- `test_task_context_display.py`: 基础功能单元测试
- `test_interactive_flow_demo.py`: 完整交互流程演示

### 运行测试
```powershell
# 基础功能测试
python test_task_context_display.py

# 完整流程演示
python test_interactive_flow_demo.py
```

## 配置和自定义

### 任务描述截断长度
可以修改 `get_task_context_summary()` 方法中的截断长度（当前为50字符）：
```python
intent = context['task_intent'][:50] + "..." if len(context['task_intent']) > 50 else context['task_intent']
```

### URL显示格式
可以自定义URL的显示格式，修改域名提取逻辑：
```python
domain = parsed_url.netloc
```

## 兼容性
- ✅ 与现有交互式标注程序完全兼容
- ✅ 不影响现有的UI显示功能
- ✅ 支持所有现有的图片显示模式
- ✅ 向后兼容原有的输入处理逻辑

## 重要更新: URL实时跟踪 (2024-12-19)

### 问题解决
原始实现中URL只在任务开始时设置，不会随页面跳转更新。现已修复：

### 改进内容
1. **修改 `_handle_browser_action` 方法**：
   - 从环境控制器的 `new_state` 中提取当前页面URL
   - 每次动作执行成功后自动更新任务上下文中的current_url

2. **关键修复: URL显示逻辑**：
   ```python
   # 修复前: 只显示域名，看不出页面变化
   domain = parsed_url.netloc
   summary_parts.append(f"页面: {domain}")  # 总是显示同样的域名

   # 修复后: 显示完整URL，不截断
   full_url = context['current_url']
   summary_parts.append(f"页面: {full_url}")  # 显示完整URL变化
   ```

3. **修复后显示效果**：
   - 初始状态：`起始页面: https://shop.example.com/`
   - 搜索页面：`页面: https://shop.example.com/search`
   - 搜索结果：`页面: https://shop.example.com/search?query=wireless+headphones&category=electronics`
   - 商品详情：`页面: https://shop.example.com/products/wireless-headphones-xyz123`
   - 评论标签：`页面: https://shop.example.com/products/wireless-headphones-xyz123#reviews`
   - 跨域跳转：`页面: https://competitor.com/similar-product`

### 测试验证
- `test_url_tracking.py`: 基础URL跟踪功能测试
- `demo_url_tracking_flow.py`: 完整交互流程中的URL跟踪演示

现在用户可以清楚看到每次页面操作的URL变化，包括路径和参数变化！

## 总结
此功能增强显著改善了用户体验，让标注人员在复杂的Web任务操作过程中始终清楚当前的任务目标、进展情况和所在页面位置。**特别是URL实时跟踪功能，让用户随时了解当前所在的页面位置，在多页面跳转的复杂任务中提供了清晰的导航指引。**这些改进大幅提高了标注效率和准确性。