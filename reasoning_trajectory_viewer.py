#!/usr/bin/env python3
"""
Reasoning Trajectory Viewer
可视化包含推理过程的轨迹数据，生成 HTML 网页
"""

import argparse
import base64
import json
import lzma
import pickle
import re
import webbrowser
from io import BytesIO
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

from PIL import Image

# 导入必要的常量用于 action 解码
try:
    import os
    # 设置临时环境变量避免 browser_env 初始化错误
    if 'DATASET' not in os.environ:
        os.environ['DATASET'] = 'webarena'

    from browser_env.constants import (
        ASCII_CHARSET,
        FREQ_UNICODE_CHARSET,
        SPECIAL_KEYS,
    )
    from itertools import chain

    # 构建 id 到字符的映射
    _key2id = {
        key: i
        for i, key in enumerate(
            chain(SPECIAL_KEYS, ASCII_CHARSET, FREQ_UNICODE_CHARSET, ["\n"])
        )
    }
    _id2key = sorted(_key2id, key=_key2id.get)  # type: ignore[arg-type]
except (ImportError, KeyError) as e:
    # 如果导入失败，使用默认值
    print(f"警告: 无法导入 browser_env 常量 ({e})，将使用简化的文本解码")
    _key2id = {}
    _id2key = []


# ActionTypes 常量定义（避免导入使用 match/case 的模块）
class ActionTypes:
    """Valid action types for browser env."""
    NONE = 0
    SCROLL = 1
    KEY_PRESS = 2
    MOUSE_CLICK = 3
    KEYBOARD_TYPE = 4
    MOUSE_HOVER = 5
    CLICK = 6
    TYPE = 7
    HOVER = 8
    PAGE_FOCUS = 9
    NEW_TAB = 10
    GO_BACK = 11
    GO_FORWARD = 12
    GOTO_URL = 13
    PAGE_CLOSE = 14
    CHECK = 15
    SELECT_OPTION = 16
    STOP = 17
    CLEAR = 18


class TrajectoryDataLoader:
    """轨迹数据加载器"""

    @staticmethod
    def load_trajectory(file_path: Path) -> List[Any]:
        """
        加载轨迹文件

        Args:
            file_path: .pkl.xz 文件路径

        Returns:
            轨迹数据列表
        """
        try:
            with lzma.open(file_path, 'rb') as f:
                trajectory = pickle.load(f)
            return trajectory
        except Exception as e:
            raise Exception(f"加载轨迹文件失败: {e}")

    @staticmethod
    def load_metadata(file_path: Path) -> Dict[str, Any]:
        """
        加载元数据文件

        Args:
            file_path: .pkl.xz 文件路径（自动查找对应的 metadata.json）

        Returns:
            元数据字典
        """
        # 构建 metadata 文件路径
        base_name = file_path.stem.replace('.pkl', '')
        metadata_file = file_path.parent / f"{base_name}_metadata.json"

        if metadata_file.exists():
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"警告: 加载元数据失败: {e}")
                return {}
        else:
            return {}

    @staticmethod
    def extract_steps(trajectory: List[Any]) -> List[Tuple[Dict[str, Any], Any]]:
        """
        从轨迹中提取 (StateInfo, Action) 对

        Args:
            trajectory: 轨迹数据列表

        Returns:
            (StateInfo, Action) 元组列表
        """
        steps = []
        current_state = None

        for item in trajectory:
            # 判断是 StateInfo
            if isinstance(item, dict) and "observation" in item and "info" in item:
                current_state = item
            # 判断是 Action（通过检查是否有 action_type 属性）
            elif hasattr(item, 'action_type'):
                if current_state is not None:
                    steps.append((current_state, item))

        return steps


class ImageProcessor:
    """图像处理器"""

    @staticmethod
    def pil_to_base64(image: Image.Image, max_width: int = 800, quality: int = 85) -> str:
        """
        将 PIL Image 转换为 base64 编码字符串

        Args:
            image: PIL Image 对象
            max_width: 最大宽度（压缩）
            quality: JPEG 质量 (1-100)

        Returns:
            base64 编码的图片字符串
        """
        # 调整大小
        if image.width > max_width:
            ratio = max_width / image.width
            new_height = int(image.height * ratio)
            image = image.resize((max_width, new_height), Image.Resampling.LANCZOS)

        # 转换为 RGB（如果是 RGBA）
        if image.mode == 'RGBA':
            image = image.convert('RGB')

        # 保存为 JPEG 并编码
        buffered = BytesIO()
        image.save(buffered, format="JPEG", quality=quality)
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

        return f"data:image/jpeg;base64,{img_str}"

    @staticmethod
    def extract_screenshot_from_state(state_info: Dict[str, Any]) -> Optional[Image.Image]:
        """
        从 StateInfo 提取截图

        Args:
            state_info: StateInfo 字典

        Returns:
            PIL Image 对象，如果提取失败则返回 None
        """
        try:
            observation = state_info.get("observation", {})

            # 尝试不同的键名
            for key in ["image", "image_som", "screenshot"]:
                if key in observation:
                    img_data = observation[key]

                    # 如果是 numpy array
                    if hasattr(img_data, "shape"):
                        return Image.fromarray(img_data)

                    # 如果是 PIL Image
                    if isinstance(img_data, Image.Image):
                        return img_data

                    # 如果是 bytes
                    if isinstance(img_data, bytes):
                        return Image.open(BytesIO(img_data))

            return None

        except Exception as e:
            print(f"警告: 提取截图失败: {e}")
            return None


class ReasoningFormatter:
    """推理过程格式化器"""

    @staticmethod
    def highlight_reasoning(reasoning: str) -> str:
        """
        对推理文本进行语法高亮

        Args:
            reasoning: 原始推理文本

        Returns:
            带 HTML 标签的推理文本
        """
        if not reasoning:
            return "<p class='no-reasoning'>（无推理过程）</p>"

        # 转义 HTML 特殊字符
        reasoning = reasoning.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

        # 恢复我们的标签（用于高亮）
        reasoning = reasoning.replace('&lt;zoom in&gt;', '<span class="tag-marker">&lt;zoom in&gt;</span><div class="tag-zoom-in">')
        reasoning = reasoning.replace('&lt;/zoom in&gt;', '</div><span class="tag-marker">&lt;/zoom in&gt;</span>')

        reasoning = reasoning.replace('&lt;shift&gt;', '<span class="tag-marker">&lt;shift&gt;</span><div class="tag-shift">')
        reasoning = reasoning.replace('&lt;/shift&gt;', '</div><span class="tag-marker">&lt;/shift&gt;</span>')

        reasoning = reasoning.replace('&lt;summary&gt;', '<span class="tag-marker">&lt;summary&gt;</span><div class="tag-summary">')
        reasoning = reasoning.replace('&lt;/summary&gt;', '</div><span class="tag-marker">&lt;/summary&gt;</span>')

        reasoning = reasoning.replace('&lt;action&gt;', '<span class="tag-marker">&lt;action&gt;</span><div class="tag-action">')
        reasoning = reasoning.replace('&lt;/action&gt;', '</div><span class="tag-marker">&lt;/action&gt;</span>')

        # 保留换行
        reasoning = reasoning.replace('\n', '<br>')

        return f'<div class="reasoning-content">{reasoning}</div>'

    @staticmethod
    def format_action(action: Any) -> Tuple[str, str]:
        """
        格式化动作为字符串和类型

        Args:
            action: Action dataclass

        Returns:
            (动作字符串, 动作类型名称) 元组
        """
        element_id = action.element_id
        action_type = action.action_type

        if action_type == ActionTypes.CLICK:
            action_str = f"click [{element_id}]"
            action_type_name = "click"
        elif action_type == ActionTypes.CLEAR:
            action_str = f"clear [{element_id}]"
            action_type_name = "clear"
        elif action_type == ActionTypes.TYPE:
            text = "".join([_id2key[i] for i in action.text]) if action.text else ""
            action_str = f"type [{element_id}] [{text}]"
            action_type_name = "type"
        elif action_type == ActionTypes.HOVER:
            action_str = f"hover [{element_id}]"
            action_type_name = "hover"
        elif action_type == ActionTypes.SCROLL:
            action_str = f"scroll [{action.direction}]"
            action_type_name = "scroll"
        elif action_type == ActionTypes.KEY_PRESS:
            action_str = f"press [{action.key_comb}]"
            action_type_name = "key_press"
        elif action_type == ActionTypes.GOTO_URL:
            action_str = f"goto [{action.url}]"
            action_type_name = "goto"
        elif action_type == ActionTypes.NEW_TAB:
            action_str = "new_tab"
            action_type_name = "new_tab"
        elif action_type == ActionTypes.PAGE_CLOSE:
            action_str = "close_tab"
            action_type_name = "close_tab"
        elif action_type == ActionTypes.GO_BACK:
            action_str = "go_back"
            action_type_name = "go_back"
        elif action_type == ActionTypes.GO_FORWARD:
            action_str = "go_forward"
            action_type_name = "go_forward"
        elif action_type == ActionTypes.PAGE_FOCUS:
            action_str = f"page_focus [{action.page_number}]"
            action_type_name = "page_focus"
        elif action_type == ActionTypes.STOP:
            action_str = f"stop {action.answer}"
            action_type_name = "stop"
        elif action_type == ActionTypes.NONE:
            action_str = "none"
            action_type_name = "none"
        else:
            action_str = f"unknown_action_{action_type}"
            action_type_name = "unknown"

        return action_str, action_type_name


class HTMLGenerator:
    """HTML 生成器"""

    def __init__(self, output_path: Path):
        """
        初始化 HTML 生成器

        Args:
            output_path: 输出 HTML 文件路径
        """
        self.output_path = output_path

    def generate_html(
        self,
        steps: List[Tuple[Dict[str, Any], Any]],
        metadata: Dict[str, Any],
        trajectory_name: str
    ):
        """
        生成完整的 HTML 文件

        Args:
            steps: (StateInfo, Action) 元组列表
            metadata: 元数据
            trajectory_name: 轨迹名称
        """
        intent = metadata.get("intent", "未知任务")
        env_name = metadata.get("environment", "未知环境")

        # 生成 HTML 内容
        html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{trajectory_name} - Reasoning Trajectory Viewer</title>
    {self._generate_styles()}
</head>
<body>
    {self._generate_header(trajectory_name, intent, env_name, len(steps))}
    <div class="container">
        {self._generate_steps(steps)}
    </div>
    {self._generate_scripts()}
</body>
</html>
"""

        # 写入文件
        with open(self.output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"✓ HTML 文件已生成: {self.output_path}")

    def _generate_header(self, trajectory_name: str, intent: str, env_name: str, num_steps: int) -> str:
        """生成页面头部"""
        # 生成步骤导航链接
        nav_links = []
        for i in range(1, num_steps + 1):
            nav_links.append(f'<a href="#step-{i}" class="nav-link">步骤 {i}</a>')

        nav_html = " ".join(nav_links)

        return f"""
    <header class="header">
        <div class="header-content">
            <h1>🔍 Reasoning Trajectory Viewer</h1>
            <div class="info-grid">
                <div class="info-item">
                    <span class="info-label">轨迹名称:</span>
                    <span class="info-value">{trajectory_name}</span>
                </div>
                <div class="info-item">
                    <span class="info-label">环境:</span>
                    <span class="info-value">{env_name}</span>
                </div>
                <div class="info-item">
                    <span class="info-label">总步数:</span>
                    <span class="info-value">{num_steps}</span>
                </div>
            </div>
            <div class="task-intent">
                <span class="info-label">🎯 任务目标:</span>
                <p>{intent}</p>
            </div>
            <div class="step-navigation">
                <span class="info-label">快速跳转:</span>
                <div class="nav-links">
                    {nav_html}
                </div>
            </div>
        </div>
    </header>
"""

    def _generate_steps(self, steps: List[Tuple[Dict[str, Any], Any]]) -> str:
        """生成所有步骤"""
        steps_html = []

        for i, (state_info, action) in enumerate(steps, 1):
            step_html = self._generate_step(i, state_info, action)
            steps_html.append(step_html)

        return "\n".join(steps_html)

    def _generate_step(self, step_num: int, state_info: Dict[str, Any], action: Any) -> str:
        """生成单个步骤"""
        # 提取截图
        screenshot = ImageProcessor.extract_screenshot_from_state(state_info)
        if screenshot:
            img_base64 = ImageProcessor.pil_to_base64(screenshot)
            img_html = f'<img src="{img_base64}" alt="Step {step_num} Screenshot" class="screenshot">'
        else:
            img_html = '<p class="no-image">（无截图）</p>'

        # 格式化推理
        reasoning = getattr(action, 'raw_prediction', '')
        reasoning_html = ReasoningFormatter.highlight_reasoning(reasoning)

        # 格式化动作
        action_str, action_type = ReasoningFormatter.format_action(action)

        return f"""
    <div class="step-card" id="step-{step_num}">
        <div class="step-header">
            <h2 class="step-title">步骤 {step_num}</h2>
            <a href="#step-{step_num}" class="step-anchor">#</a>
        </div>

        <div class="screenshot-container">
            {img_html}
        </div>

        <div class="reasoning-section">
            <h3 class="section-title">📝 推理过程</h3>
            {reasoning_html}
        </div>

        <div class="action-section">
            <h3 class="section-title">🎯 执行动作</h3>
            <div class="action-badge action-{action_type}">
                {action_str}
            </div>
        </div>
    </div>
"""

    def _generate_styles(self) -> str:
        """生成 CSS 样式"""
        return """
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
            line-height: 1.6;
            padding-bottom: 50px;
        }

        .header {
            background: white;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 30px;
            position: sticky;
            top: 0;
            z-index: 1000;
        }

        .header-content {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }

        .header h1 {
            color: #667eea;
            font-size: 2em;
            margin-bottom: 15px;
        }

        .info-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 15px;
        }

        .info-item {
            background: #f8f9fa;
            padding: 10px;
            border-radius: 8px;
        }

        .info-label {
            font-weight: 600;
            color: #667eea;
            display: block;
            margin-bottom: 5px;
        }

        .info-value {
            color: #333;
        }

        .task-intent {
            background: #e3f2fd;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 15px;
            border-left: 4px solid #2196f3;
        }

        .task-intent p {
            margin-top: 8px;
            font-size: 1.05em;
            color: #1565c0;
        }

        .step-navigation {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
        }

        .nav-links {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 10px;
        }

        .nav-link {
            background: #667eea;
            color: white;
            padding: 6px 12px;
            border-radius: 5px;
            text-decoration: none;
            font-size: 0.9em;
            transition: all 0.2s;
        }

        .nav-link:hover {
            background: #764ba2;
            transform: translateY(-2px);
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 0 20px;
        }

        .step-card {
            background: white;
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 5px 20px rgba(0,0,0,0.1);
            transition: transform 0.2s;
        }

        .step-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 30px rgba(0,0,0,0.15);
        }

        .step-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 2px solid #e0e0e0;
        }

        .step-title {
            color: #667eea;
            font-size: 1.8em;
        }

        .step-anchor {
            color: #999;
            text-decoration: none;
            font-size: 1.2em;
            transition: color 0.2s;
        }

        .step-anchor:hover {
            color: #667eea;
        }

        .screenshot-container {
            text-align: center;
            margin-bottom: 25px;
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
        }

        .screenshot {
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 3px 15px rgba(0,0,0,0.1);
            cursor: pointer;
            transition: transform 0.2s;
        }

        .screenshot:hover {
            transform: scale(1.02);
        }

        .no-image {
            color: #999;
            font-style: italic;
            padding: 40px;
        }

        .reasoning-section, .action-section {
            margin-bottom: 25px;
        }

        .section-title {
            color: #333;
            font-size: 1.3em;
            margin-bottom: 15px;
            display: flex;
            align-items: center;
        }

        .reasoning-content {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            font-family: "Consolas", "Monaco", monospace;
            font-size: 0.95em;
            line-height: 1.8;
            white-space: pre-wrap;
            word-wrap: break-word;
        }

        .tag-marker {
            color: #999;
            font-weight: bold;
        }

        .tag-zoom-in, .tag-shift, .tag-summary, .tag-action {
            margin: 10px 0;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid;
        }

        .tag-zoom-in {
            background: #e3f2fd;
            border-left-color: #2196f3;
        }

        .tag-shift {
            background: #e8f5e9;
            border-left-color: #4caf50;
        }

        .tag-summary {
            background: #fff3e0;
            border-left-color: #ff9800;
        }

        .tag-action {
            background: #fce4ec;
            border-left-color: #e91e63;
        }

        .no-reasoning {
            color: #999;
            font-style: italic;
            padding: 20px;
        }

        .action-badge {
            display: inline-block;
            padding: 12px 20px;
            border-radius: 8px;
            font-family: "Consolas", "Monaco", monospace;
            font-size: 1.1em;
            font-weight: 600;
            color: white;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }

        .action-click { background: #2196f3; }
        .action-type { background: #4caf50; }
        .action-scroll { background: #ff9800; }
        .action-hover { background: #9c27b0; }
        .action-goto { background: #00bcd4; }
        .action-key_press { background: #ff5722; }
        .action-stop { background: #f44336; }
        .action-clear { background: #795548; }
        .action-new_tab { background: #3f51b5; }
        .action-close_tab { background: #607d8b; }
        .action-go_back { background: #009688; }
        .action-go_forward { background: #8bc34a; }
        .action-page_focus { background: #cddc39; color: #333; }
        .action-none { background: #9e9e9e; }
        .action-unknown { background: #757575; }

        @media (max-width: 768px) {
            .header h1 {
                font-size: 1.5em;
            }

            .info-grid {
                grid-template-columns: 1fr;
            }

            .nav-links {
                justify-content: center;
            }

            .step-card {
                padding: 20px;
            }

            .step-title {
                font-size: 1.4em;
            }
        }

        html {
            scroll-behavior: smooth;
        }
    </style>
"""

    def _generate_scripts(self) -> str:
        """生成 JavaScript 脚本"""
        return """
    <script>
        // 点击图片放大
        document.addEventListener('DOMContentLoaded', function() {
            const screenshots = document.querySelectorAll('.screenshot');
            screenshots.forEach(img => {
                img.addEventListener('click', function() {
                    window.open(this.src, '_blank');
                });
            });
        });
    </script>
"""


def list_trajectories(base_path: str = "data/annotate_with_reasoning") -> Dict[str, List[Path]]:
    """列出所有可用的轨迹文件"""
    path = Path(base_path)
    if not path.exists():
        return {}

    envs = {}
    for env_dir in path.iterdir():
        if env_dir.is_dir():
            trajectories = sorted(env_dir.glob("*.pkl.xz"))
            if trajectories:
                envs[env_dir.name] = trajectories

    return envs


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="Reasoning Trajectory Viewer - 可视化包含推理过程的轨迹数据"
    )

    parser.add_argument(
        "--input",
        type=str,
        help="指定 .pkl.xz 文件路径"
    )

    parser.add_argument(
        "--output",
        type=str,
        help="指定输出 HTML 文件路径（默认与输入文件同目录）"
    )

    parser.add_argument(
        "--env",
        type=str,
        help="指定环境名称（处理该环境下所有轨迹）",
        default = "classifieds"
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="列出所有可用的轨迹文件"
    )

    parser.add_argument(
        "--open",
        action="store_true",
        help="生成后自动在浏览器中打开"
    )

    parser.add_argument(
        "--base-path",
        type=str,
        default="data/annotate_with_reasoning",
        help="轨迹文件基础路径（默认: data/annotate_with_reasoning）"
    )

    args = parser.parse_args()

    # 列出所有轨迹
    if args.list:
        envs = list_trajectories(args.base_path)
        if not envs:
            print(f"在 '{args.base_path}' 中未找到轨迹文件")
            return

        print(f"\n{'='*70}")
        print("可用的轨迹文件:")
        print(f"{'='*70}\n")

        for env_name, trajectories in envs.items():
            print(f"{env_name} ({len(trajectories)} 个文件):")
            for traj in trajectories:
                size = traj.stat().st_size / 1024
                print(f"  - {traj.name} ({size:.1f} KB)")
            print()
        return

    # 处理单个文件
    if args.input:
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"错误: 文件不存在: {input_path}")
            return

        # 确定输出路径
        if args.output:
            output_path = Path(args.output)
        else:
            output_path = input_path.parent / f"{input_path.stem.replace('.pkl', '')}_view.html"

        # 加载数据
        print(f"正在加载轨迹: {input_path.name}")
        trajectory = TrajectoryDataLoader.load_trajectory(input_path)
        metadata = TrajectoryDataLoader.load_metadata(input_path)
        steps = TrajectoryDataLoader.extract_steps(trajectory)

        print(f"找到 {len(steps)} 个步骤")

        # 生成 HTML
        generator = HTMLGenerator(output_path)
        generator.generate_html(steps, metadata, input_path.stem.replace('.pkl', ''))

        # 自动打开浏览器
        if args.open:
            webbrowser.open(output_path.absolute().as_uri())
            print(f"✓ 已在浏览器中打开")

        return

    # 处理环境下所有文件
    if args.env:
        env_path = Path(args.base_path) / args.env
        if not env_path.exists():
            print(f"错误: 环境目录不存在: {env_path}")
            return

        trajectories = sorted(env_path.glob("*.pkl.xz"))
        if not trajectories:
            print(f"在环境 '{args.env}' 中未找到轨迹文件")
            return

        print(f"\n正在处理环境 '{args.env}' 中的 {len(trajectories)} 个轨迹...\n")

        for traj_file in trajectories:
            try:
                # 确定输出路径
                output_path = traj_file.parent / f"{traj_file.stem.replace('.pkl', '')}_view.html"

                # 加载数据
                print(f"处理: {traj_file.name}")
                trajectory = TrajectoryDataLoader.load_trajectory(traj_file)
                metadata = TrajectoryDataLoader.load_metadata(traj_file)
                steps = TrajectoryDataLoader.extract_steps(trajectory)

                # 生成 HTML
                generator = HTMLGenerator(output_path)
                generator.generate_html(steps, metadata, traj_file.stem.replace('.pkl', ''))

            except Exception as e:
                print(f"  ✗ 失败: {e}")

        print(f"\n✓ 完成！共处理 {len(trajectories)} 个轨迹")
        return

    # 默认显示帮助
    parser.print_help()


if __name__ == "__main__":
    main()
