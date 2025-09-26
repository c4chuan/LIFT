# 🖼️ 增强图片显示功能使用指南

## 🎯 新功能概览

LIFT项目的图片显示系统已全面升级，现在支持：

1. **原比例显示** - 保持图片原始长宽比，避免变形
2. **持续窗口模式** - 图片窗口保持打开状态，边看边操作
3. **智能缩放** - 自适应屏幕尺寸，确保最佳显示效果
4. **灵活配置** - 丰富的命令行参数控制显示行为

## 🚀 快速开始

### 基本用法（推荐设置）
```bash
python src/interactive_annotator.py --image-display-method tkinter
```

### 使用原比例显示（默认开启）
```bash
# 图片以原始比例显示，不会变形
python src/interactive_annotator.py --image-display-method tkinter
```

### 使用持续窗口模式（默认开启）
```bash
# 图片窗口保持打开，每次新截图会更新窗口内容
python src/interactive_annotator.py --image-display-method tkinter
```

## ⚙️ 配置选项

### 显示方法选择
```bash
# Tkinter GUI显示（推荐）
--image-display-method tkinter

# ASCII艺术显示
--image-display-method ascii --ascii-width 100

# 系统默认查看器
--image-display-method system

# 仅显示路径
--image-display-method off
```

### 比例控制
```bash
# 保持原始比例（默认）
python src/interactive_annotator.py

# 不保持比例，按指定尺寸缩放
python src/interactive_annotator.py --no-keep-aspect-ratio --image-window-size 1024x768
```

### 窗口行为控制
```bash
# 持续窗口模式（默认）- 窗口保持打开
python src/interactive_annotator.py

# 传统模式 - 每次显示需手动关闭
python src/interactive_annotator.py --no-persistent-window
```

## 🎮 实际使用体验

### 场景1：标准Web任务标注
```bash
python src/interactive_annotator.py \
  --image-display-method tkinter \
  --render
```

**体验效果：**
- 📸 每次SOM操作后，截图以原始比例显示
- 🖼️ 图片窗口保持打开，可以边看截图边输入下一个操作
- ✨ 窗口内容自动更新，无需重复开关窗口

### 场景2：无GUI环境
```bash
python src/interactive_annotator.py \
  --image-display-method ascii \
  --ascii-width 120 \
  --no-render
```

**体验效果：**
- 🎨 截图转换为ASCII艺术在终端显示
- 📏 可调整显示宽度适应终端尺寸
- 🔄 每次操作后显示新的ASCII图片

### 场景3：高分辨率显示
```bash
python src/interactive_annotator.py \
  --image-display-method tkinter \
  --image-window-size 1920x1080
```

**体验效果：**
- 🖥️ 充分利用大屏幕空间显示图片细节
- 📐 保持原始比例，避免图片变形
- 🎯 适合处理复杂的网页布局

## 💡 使用技巧

### 1. 根据任务选择显示模式
- **复杂界面任务**: 使用 `tkinter` 模式，保持原比例
- **简单任务**: 使用 `ascii` 模式，快速预览
- **批量处理**: 使用 `off` 模式，专注操作

### 2. 优化显示设置
```bash
# 大图片任务 - 确保能看清细节
--keep-aspect-ratio --image-display-method tkinter

# 快速调试 - 减少窗口干扰
--no-persistent-window --image-display-method ascii
```

### 3. 多屏幕环境
- 持续窗口模式下，可以将图片窗口移到副屏
- 主屏专注输入操作，副屏查看图片内容

## 🔧 高级配置

### 自定义窗口尺寸
```bash
# 固定窗口大小（不保持比例时生效）
--image-window-size 800x600 --no-keep-aspect-ratio
```

### ASCII显示优化
```bash
# 适配不同终端宽度
--ascii-width 80    # 标准终端
--ascii-width 120   # 宽屏终端
--ascii-width 160   # 超宽终端
```

## 🚨 注意事项

1. **持续窗口模式注意**：
   - 窗口会保持打开状态直到程序结束
   - 可以手动关闭窗口，下次显示时会创建新窗口

2. **性能考虑**：
   - 大图片在原比例显示时可能占用较多内存
   - 可以通过 `--no-keep-aspect-ratio` 强制缩小图片

3. **系统兼容性**：
   - Tkinter模式需要GUI环境支持
   - ASCII模式适用于所有环境
   - 系统查看器模式依赖系统默认应用

## 📊 性能对比

| 模式 | 显示速度 | 内存占用 | 显示质量 | 适用场景 |
|------|----------|----------|----------|----------|
| Tkinter | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | 复杂任务，细节重要 |
| ASCII | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | 快速预览，远程环境 |
| System | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | 熟悉的查看器界面 |
| Off | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | - | 专注操作，批量处理 |

## 🎉 升级优势

与旧版本相比，新的显示系统提供：

✅ **更好的用户体验** - 持续窗口，边看边操作
✅ **更真实的显示** - 保持原始比例，避免误判
✅ **更灵活的配置** - 丰富的参数控制显示行为
✅ **更好的兼容性** - 支持各种环境和使用场景
✅ **更稳定的性能** - 优化的窗口管理和资源清理

现在就开始体验全新的图片显示功能吧！🚀