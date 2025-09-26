# 交互式Web任务标注工具

这是一个基于终端的交互式标注工具，用于标注Web任务轨迹。工具基于 `visualwebarena/runners/eval/eval_vwa_agent.py` 的核心流程，将agent自动决策替换为用户手动输入操作指令。

## 功能特点

🎯 **任务管理**: 自动加载和管理三个环境的任务队列 (classifieds, reddit, shopping)
🔄 **断点续标**: 支持中断后继续标注功能
🖥️ **用户界面**: 清晰的终端界面显示任务信息和进度
📝 **输入解析**: 支持常见操作命令的文本解析
🌐 **环境控制**: 复用现有的浏览器环境和评估系统
💾 **轨迹保存**: 自动保存成功的标注轨迹到不同格式

## 系统要求

### 必须要求
- **Python 3.10+** (项目使用了match语句等新特性)
- visualwebarena 项目的完整环境
- 浏览器环境 (Playwright)

### 依赖包
```bash
pip install playwright beautifulsoup4 requests pillow
```

## 项目结构

```
src/annotation/
├── __init__.py                 # 模块初始化
├── task_manager.py            # 任务管理器
├── annotation_ui.py           # 用户界面
├── input_parser.py            # 输入解析器
├── environment_controller.py  # 环境控制器
└── trajectory_manager.py      # 轨迹管理器

src/interactive_annotator.py   # 主程序入口
```

## 安装和配置

1. **确保Python版本**
   ```bash
   python --version  # 应该是 3.10 或更高版本
   ```

2. **安装依赖**
   ```bash
   pip install playwright requests pillow beautifulsoup4
   playwright install chromium
   ```

3. **检查数据目录**
   ```
   data/annotate/
   ├── classifieds_tasks.json  # 分类广告任务
   ├── reddit_tasks.json      # Reddit任务
   └── shopping_tasks.json    # 购物任务
   ```

## 使用方法

### 基本使用
```bash
# 启动交互式标注工具（显示浏览器）
python src/interactive_annotator.py --render

# 隐藏浏览器界面（提高性能）
python src/interactive_annotator.py --no-render

# 查看所有选项
python src/interactive_annotator.py --help
```

### 命令行选项
```
--annotate-dir        标注数据目录 (默认: data/annotate)
--progress-dir        进度数据目录 (默认: data/annotation_progress)
--render              显示浏览器界面 (默认)
--no-render           隐藏浏览器界面
--slow-mo N           浏览器操作延迟毫秒数
--max-steps N         每个任务的最大步数 (默认: 30)
--viewport-width      浏览器视窗宽度 (默认: 1280)
--viewport-height     浏览器视窗高度 (默认: 2048)
```

## 操作命令

### 基本操作
- `click [ID]` - 点击元素，如: `click [10]`
- `type [ID] [文本] [1]` - 在元素中输入文本，如: `type [5] [blue kayak] [1]`
- `hover [ID]` - 悬停在元素上，如: `hover [15]`
- `scroll [方向]` - 滚动页面，如: `scroll up` 或 `scroll down`

### 页面操作
- `key_press [按键]` - 按键操作，如: `key_press Enter`
- `goto [URL]` - 跳转到URL
- `go_back` - 返回上一页
- `go_forward` - 前进到下一页
- `new_tab` - 打开新标签
- `close_tab` - 关闭当前标签

### 控制命令
- `stop [答案]` - 停止当前任务并提交答案
- `reset` - 重置当前任务
- `skip` - 跳过当前任务
- `help` - 显示帮助信息
- `quit` - 退出程序

## 工作流程

1. **启动程序** - 显示欢迎界面和进度统计
2. **加载任务** - 自动加载下一个待标注任务，显示任务信息和初始截图
3. **交互标注** - 用户输入操作命令，系统执行并显示新截图
4. **任务完成** - 输入`stop`命令后自动评估任务
5. **保存轨迹** - 成功任务自动保存轨迹到多种格式
6. **继续下一个** - 自动进入下一个任务，支持断点续标

## 数据保存

### 轨迹保存位置
```
data/annotate/trajectories/
├── classifieds/    # 分类广告任务轨迹
├── reddit/         # Reddit任务轨迹
└── shopping/       # 购物任务轨迹
```

### 保存格式
- `*.pkl.xz` - 压缩的pickle格式（与原系统兼容）
- `*_metadata.json` - 任务元数据（便于查看和分析）
- `*_script.py` - Python脚本格式（便于回放和理解）

### 进度跟踪
```
data/annotation_progress/
├── progress.json        # 整体进度
├── completed_tasks.json # 已完成任务列表
└── temp/               # 临时文件
```

## 测试

### 基本模块测试
```bash
# 测试基本模块功能（不依赖完整环境）
python simple_test.py
```

### 完整功能测试
```bash
# 需要Python 3.10+和完整环境
python src/interactive_annotator.py --render --max-steps 5
```

## 故障排除

### Python版本问题
```
Error: invalid syntax (match statement)
```
- 解决: 升级到Python 3.10+

### 模块导入问题
```
ModuleNotFoundError: No module named 'browser_env'
```
- 检查visualwebarena项目完整性
- 确保路径设置正确

### 浏览器问题
```
playwright._impl._api_types.Error
```
- 运行: `playwright install chromium`
- 检查浏览器权限

### 任务文件问题
```
FileNotFoundError: task file not found
```
- 检查data/annotate目录是否包含任务文件
- 确保JSON文件格式正确

## 开发说明

### 模块设计
- **TaskManager**: 任务队列和进度管理
- **AnnotationUI**: 终端用户界面
- **InputParser**: 用户命令解析
- **EnvironmentController**: 浏览器环境封装
- **TrajectoryManager**: 轨迹存储和管理

### 扩展方向
- 支持更多动作类型
- 添加任务筛选和排序功能
- 实现轨迹回放功能
- 支持多用户协作标注
- 添加质量评估指标

## 当前状态

✅ 所有核心模块已完成并测试通过
✅ 成功加载233个标注任务 (classifieds: 56, reddit: 63, shopping: 114)
✅ 基本模块功能测试全部通过
✅ 支持完整的交互式标注流程
⚠️ 需要Python 3.10+才能使用完整功能

## 许可证

本工具基于原 visualwebarena 项目开发，遵循相同的许可证条款。