#!/usr/bin/env python3
"""
交互式Web任务标注工具
主程序入口
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional

# 添加路径以便导入模块
sys.path.append(str(Path(__file__).parent.parent))

from src.annotation.task_manager import TaskManager
from src.annotation.annotation_ui import AnnotationUI
from src.annotation.input_parser import InputParser, InputParseError
from src.annotation.environment_controller import EnvironmentController
from src.annotation.trajectory_manager import TrajectoryManager

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class InteractiveAnnotator:
    """交互式标注器主类"""

    def __init__(self, args):
        """
        初始化标注器

        Args:
            args: 命令行参数
        """
        self.args = args
        self.task_manager = TaskManager(
            annotate_dir=args.annotate_dir,
            progress_dir=args.progress_dir,
            difficulty_filter=getattr(args, 'difficulty', None)
        )
        self.ui = AnnotationUI(
            image_display_method=getattr(args, 'image_display_method', 'auto'),
            image_window_size=getattr(args, 'image_window_size', '800x600'),
            ascii_width=getattr(args, 'ascii_width', 80),
            keep_aspect_ratio=getattr(args, 'keep_aspect_ratio', True),
            persistent_window=getattr(args, 'persistent_window', True),
            target_screen=getattr(args, 'display_screen', 0)
        )
        self.input_parser = InputParser()
        self.env_controller = None
        self.trajectory_manager = TrajectoryManager(annotate_dir=args.annotate_dir)

        self.current_env_name = None
        self.current_task = None
        self.current_task_info = None

    async def run(self):
        """运行主程序循环"""
        try:
            self.ui.display_welcome()
            self._show_progress()

            while True:
                # 获取下一个任务
                next_task = self.task_manager.get_next_task()
                if not next_task:
                    self.ui.display_completion_message()
                    break

                env_name, task = next_task
                await self._process_task(env_name, task)

        except KeyboardInterrupt:
            self.ui.display_info("检测到 Ctrl+C，正在清理并退出...")
        except Exception as e:
            logger.exception(f"主程序发生异常: {e}")
            self.ui.display_error(f"程序异常: {e}")
        finally:
            await self._cleanup()

    def _show_progress(self):
        """显示进度信息"""
        progress_info = self.task_manager.get_progress_summary()
        self.ui.display_progress(progress_info)

    async def _process_task(self, env_name: str, task: Dict[str, Any]):
        """
        处理单个任务

        Args:
            env_name: 环境名称
            task: 任务数据
        """
        self.current_env_name = env_name
        self.current_task = task

        try:
            # 创建临时配置文件
            config_file = await self._create_temp_config_file(env_name, task)

            # 初始化环境控制器
            self.env_controller = EnvironmentController(
                render=self.args.render,
                slow_mo=self.args.slow_mo,
                observation_type=self.args.observation_type,
                viewport_width=self.args.viewport_width,
                viewport_height=self.args.viewport_height,
                max_steps=self.args.max_steps,
                result_dir=self.args.result_dir
            )

            # 初始化环境
            initial_state, task_info = await self.env_controller.initialize_environment(config_file)
            self.current_task_info = task_info

            # 设置任务上下文信息
            self.ui.set_task_context(
                env_name=env_name,
                task_id=task.get('task_id', 'unknown'),
                task_intent=task_info.get('intent', task.get('intent', '')),
                start_url=task_info.get('start_url', task.get('start_url', '')),
                current_url=task_info.get('start_url', task.get('start_url', '')),
                step_count=0
            )

            # 显示任务信息
            self.ui.display_task_info(env_name, task_info)

            # 显示初始截图
            screenshot_path = self.env_controller.get_latest_screenshot_path()
            self.ui.display_screenshot_ready(screenshot_path)

            # 进入交互循环
            await self._interactive_loop()

        except Exception as e:
            logger.exception(f"处理任务时发生异常: {e}")
            self.ui.display_error(f"任务处理失败: {e}")

    async def _create_temp_config_file(self, env_name: str, task: Dict[str, Any]) -> str:
        """
        创建临时配置文件

        Args:
            env_name: 环境名称
            task: 任务数据

        Returns:
            配置文件路径
        """
        # 创建临时目录
        temp_dir = tempfile.mkdtemp()

        # 配置文件内容
        config = task.copy()

        # 确保存储状态路径正确
        if config.get("storage_state"):
            # 转换相对路径为绝对路径
            storage_state = config["storage_state"]
            if not os.path.isabs(storage_state):
                storage_state = os.path.abspath(storage_state)
            config["storage_state"] = storage_state

        # 保存配置文件
        config_filename = f"{env_name}_{task['task_id']}.json"
        config_file_path = os.path.join(temp_dir, config_filename)

        with open(config_file_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        return config_file_path

    async def _interactive_loop(self):
        """交互式操作循环"""
        while True:
            try:
                # 处理GUI事件（如果有活跃窗口）
                if self.ui.has_active_window():
                    self.ui.process_gui_events()

                # 使用标准输入模式，避免GUI焦点问题
                user_input = self.ui.prompt_user_input()

                if not user_input:
                    continue

                # 解析输入
                try:
                    command_type, action, extra_info = self.input_parser.parse_input(user_input)
                except InputParseError as e:
                    self.ui.display_error(str(e))
                    continue

                # 处理命令
                if command_type == "control":
                    control_result = await self._handle_control_command(extra_info["command"])
                    if control_result == "exit_task":
                        break
                    elif control_result == "quit":
                        return

                elif command_type == "action":
                    # 如果是stop动作且执行成功，评估任务
                    if action.action_type == 17:  # ActionTypes.STOP
                        self.env_controller.current_trajectory.append(action)
                        await self._evaluate_and_finish_task()
                        break

                    success = await self._handle_browser_action(action, extra_info)




            except KeyboardInterrupt:
                self.ui.display_info("操作被中断")
                break
            except Exception as e:
                logger.exception(f"交互循环中发生异常: {e}")
                self.ui.display_error(f"处理输入时发生错误: {e}")

    async def _handle_control_command(self, command: str) -> Optional[str]:
        """
        处理控制命令

        Args:
            command: 命令名称

        Returns:
            处理结果，可能的值: "exit_task", "quit", None
        """
        if command == "help":
            self.ui.display_help()

        elif command == "reset":
            if self.ui.confirm_action("确认重置当前任务？"):
                await self._reset_current_task()

        elif command == "skip":
            if self.ui.confirm_action("确认跳过当前任务？"):
                self.task_manager.skip_current_task()
                self.ui.display_info("任务已跳过")
                return "exit_task"

        elif command == "quit":
            if self.ui.confirm_action("确认退出程序？"):
                return "quit"

        return None

    async def _handle_browser_action(self, action: Any,
                                   extra_info: Dict[str, Any]) -> bool:
        """
        处理浏览器动作

        Args:
            action: 动作对象
            extra_info: 额外信息

        Returns:
            是否执行成功
        """
        if not self.env_controller:
            self.ui.display_error("环境未初始化")
            return False

        # 获取动作描述
        action_description = self.input_parser.get_action_description(action)

        try:
            # 执行动作
            success, error_message, new_state = await self.env_controller.execute_action(action)

            # 显示结果
            self.ui.display_action_result(success, action_description, error_message)

            if success:
                # 更新步骤计数
                current_step = self.ui.current_task_context.get('step_count', 0) + 1

                # 提取当前URL并更新任务上下文
                current_url = new_state.get('url', '') if new_state else ''

                self.ui.set_task_context(
                    step_count=current_step,
                    current_url=current_url
                )

                # 显示新的截图
                screenshot_path = self.env_controller.get_latest_screenshot_path()
                if screenshot_path:
                    self.ui.display_screenshot_ready(screenshot_path)

            return success

        except Exception as e:
            logger.exception(f"执行动作时发生异常: {e}")
            self.ui.display_error(f"执行动作失败: {e}")
            return False

    async def _evaluate_and_finish_task(self):
        """评估并完成任务"""
        try:
            # 评估任务
            score = await self.env_controller.evaluate_trajectory()
            task_id_str = f"{self.current_env_name}_{self.current_task['task_id']}"

            success = score >= 1.0
            self.ui.display_task_evaluation_result(success, score, task_id_str)

            if success:
                # 保存轨迹
                trajectory = self.env_controller.get_current_trajectory()
                annotation_file = self.trajectory_manager.save_trajectory(
                    env_name=self.current_env_name,
                    task_id=self.current_task["task_id"],
                    trajectory=trajectory,
                    task_info=self.current_task_info,
                    score=score
                )

                # 标记任务完成
                self.task_manager.mark_task_completed(
                    env_name=self.current_env_name,
                    task=self.current_task,
                    annotation_file=annotation_file
                )

                self.ui.display_info("🎉 任务成功完成并保存!")

            else:
                self.ui.display_warning("任务未通过评估，请选择下一步操作")

        except Exception as e:
            logger.exception(f"任务评估时发生异常: {e}")
            self.ui.display_error(f"任务评估失败: {e}")

    async def _reset_current_task(self):
        """重置当前任务"""
        try:
            if self.env_controller:
                await self.env_controller.reset_environment()

                # 重置步骤计数
                self.ui.set_task_context(step_count=0)

                self.ui.display_info("✅ 任务已重置到初始状态")

                # 显示重置后的截图
                screenshot_path = self.env_controller.get_latest_screenshot_path()
                if screenshot_path:
                    self.ui.display_screenshot_ready(screenshot_path)
        except Exception as e:
            logger.exception(f"重置任务时发生异常: {e}")
            self.ui.display_error(f"重置任务失败: {e}")

    async def _cleanup(self):
        """清理资源"""
        if self.env_controller:
            try:
                await self.env_controller.close_environment()
            except Exception as e:
                logger.exception(f"清理环境时发生异常: {e}")

        # 清理UI资源（包括关闭图片窗口）
        if self.ui:
            try:
                # 清除任务上下文
                self.ui.clear_task_context()
                self.ui.cleanup()
            except Exception as e:
                logger.exception(f"清理UI资源时发生异常: {e}")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="交互式Web任务标注工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python src/interactive_annotator.py --render
  python src/interactive_annotator.py --no-render --max-steps 50
  python src/interactive_annotator.py --annotate-dir ./my_annotations
        """
    )

    # 基本参数
    parser.add_argument(
        "--annotate-dir",
        default="data/annotate",
        help="标注数据目录 (默认: data/annotate)"
    )

    parser.add_argument(
        "--progress-dir",
        default="data/annotation_progress",
        help="进度数据目录 (默认: data/annotation_progress)"
    )

    parser.add_argument(
        "--result-dir",
        default="data/annotation_results",
        help="结果输出目录 (默认: data/annotation_results)"
    )

    # 任务过滤参数
    parser.add_argument(
        "--difficulty",
        nargs='+',
        choices=["easy", "medium", "hard"],
        default="easy medium",
        help="过滤任务难度,可多选 (例如: --difficulty easy medium)"
    )

    # 环境参数
    parser.add_argument(
        "--render",
        action="store_true",
        default=True,
        help="显示浏览器界面 (默认: True)"
    )

    parser.add_argument(
        "--no-render",
        action="store_false",
        dest="render",
        help="隐藏浏览器界面"
    )

    parser.add_argument(
        "--slow-mo",
        type=int,
        default=0,
        help="浏览器操作延迟毫秒数 (默认: 0)"
    )

    parser.add_argument(
        "--observation-type",
        choices=["accessibility_tree", "html", "image", "image_som"],
        default="image_som",
        help="观察类型 (默认: image_som)"
    )

    parser.add_argument(
        "--viewport-width",
        type=int,
        default=1280,
        help="浏览器视窗宽度 (默认: 1280)"
    )

    parser.add_argument(
        "--viewport-height",
        type=int,
        default=2048,
        help="浏览器视窗高度 (默认: 2048)"
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=30,
        help="每个任务的最大步数 (默认: 30)"
    )

    # 图片显示参数
    parser.add_argument(
        "--image-display-method",
        choices=["tkinter", "ascii", "system", "off", "auto"],
        default="auto",
        help="图片显示方法 (默认: auto)"
    )

    parser.add_argument(
        "--image-window-size",
        default="800x600",
        help="Tkinter窗口大小 (默认: 800x600)"
    )

    parser.add_argument(
        "--ascii-width",
        type=int,
        default=80,
        help="ASCII显示宽度 (默认: 80)"
    )

    # 图片显示行为控制参数
    parser.add_argument(
        "--no-keep-aspect-ratio",
        action="store_false",
        dest="keep_aspect_ratio",
        help="不保持图片原始比例，按窗口尺寸缩放"
    )

    parser.add_argument(
        "--no-persistent-window",
        action="store_false",
        dest="persistent_window",
        help="不使用持续窗口显示，每次显示后需要手动关闭"
    )

    parser.add_argument(
        "--display-screen",
        type=int,
        default=0,
        help="指定图片窗口显示的屏幕编号 (默认: 0 - 主屏幕)"
    )

    # 调试参数
    parser.add_argument(
        "--debug",
        action="store_true",
        help="启用调试模式"
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="日志级别 (默认: INFO)"
    )

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    # 设置日志级别
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # 检查必要的目录
    required_dirs = [args.annotate_dir, args.progress_dir, args.result_dir]
    for dir_path in required_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

    # 创建并运行标注器
    annotator = InteractiveAnnotator(args)

    try:
        asyncio.run(annotator.run())
    except KeyboardInterrupt:
        print("\\n程序被用户中断")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"程序异常退出: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()