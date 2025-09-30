"""
标注界面模块
负责终端用户界面显示和交互
"""

import os
import shutil
from typing import Dict, List, Optional, Any, Tuple
from PIL import Image
import requests
from pathlib import Path

# 尝试导入GUI相关库
try:
    import tkinter as tk
    from tkinter import ttk
    HAS_TKINTER = True
except ImportError:
    HAS_TKINTER = False

try:
    import subprocess
    import platform
    HAS_SYSTEM_VIEWER = True
except ImportError:
    HAS_SYSTEM_VIEWER = False


class AnnotationUI:
    """标注界面类"""

    def __init__(self, image_display_method='auto', image_window_size='800x600', ascii_width=80,
                 keep_aspect_ratio=True, persistent_window=True, target_screen=0):
        """
        初始化界面

        Args:
            image_display_method: 图片显示方法 ('tkinter', 'ascii', 'system', 'off', 'auto')
            image_window_size: Tkinter窗口大小 (默认: '800x600')
            ascii_width: ASCII显示宽度 (默认: 80)
            keep_aspect_ratio: 是否保持图片原始比例 (默认: True)
            persistent_window: 是否持续显示窗口直到用户输入 (默认: True)
            target_screen: 目标屏幕编号 (默认: 0=主屏幕)
        """
        self.terminal_width = shutil.get_terminal_size().columns
        self.commands_help = self._build_commands_help()
        self.image_display_method = self._resolve_display_method(image_display_method)
        self.image_window_size = image_window_size
        self.ascii_width = ascii_width
        self.keep_aspect_ratio = keep_aspect_ratio
        self.persistent_window = persistent_window
        self.target_screen = target_screen
        self.image_cache = {}  # 图片缓存

        # 持续窗口管理
        self.current_window = None  # 当前显示的Tkinter窗口引用

        # 任务上下文信息
        self.current_task_context = {
            'env_name': None,
            'task_id': None,
            'task_intent': None,
            'start_url': None,
            'current_url': None,
            'step_count': 0
        }

    def _get_screen_info(self):
        """
        获取所有可用屏幕的信息

        Returns:
            list: 屏幕信息列表，每个元素包含 {'index', 'width', 'height', 'x', 'y'}
        """
        try:
            import tkinter as tk

            # 创建临时根窗口来获取屏幕信息
            temp_root = tk.Tk()
            temp_root.withdraw()  # 隐藏窗口

            screens = []

            # 获取主屏幕信息
            main_width = temp_root.winfo_screenwidth()
            main_height = temp_root.winfo_screenheight()

            screens.append({
                'index': 0,
                'width': main_width,
                'height': main_height,
                'x': 0,
                'y': 0,
                'is_primary': True
            })

            try:
                # 尝试检测多显示器（Windows）
                if hasattr(temp_root, 'wm_maxsize'):
                    # 通过移动窗口到不同位置来检测额外的显示器
                    # 这是一个简化的多屏检测方法

                    # 尝试检测右侧显示器
                    test_positions = [
                        (main_width + 100, 100),     # 右侧屏幕
                        (-main_width + 100, 100),    # 左侧屏幕
                        (100, main_height + 100),    # 下方屏幕
                        (100, -main_height + 100),   # 上方屏幕
                    ]

                    screen_index = 1
                    for pos_x, pos_y in test_positions:
                        try:
                            # 创建测试窗口
                            test_window = tk.Toplevel(temp_root)
                            test_window.withdraw()
                            test_window.geometry(f"1x1+{pos_x}+{pos_y}")
                            test_window.deiconify()
                            test_window.update()

                            # 检查窗口是否在可见区域
                            actual_x = test_window.winfo_x()
                            actual_y = test_window.winfo_y()

                            if abs(actual_x - pos_x) < 50 and abs(actual_y - pos_y) < 50:
                                # 发现了新的显示器
                                if pos_x > main_width:  # 右侧屏幕
                                    screens.append({
                                        'index': screen_index,
                                        'width': main_width,  # 假设同样尺寸
                                        'height': main_height,
                                        'x': main_width,
                                        'y': 0,
                                        'is_primary': False
                                    })
                                    screen_index += 1
                                elif pos_x < 0:  # 左侧屏幕
                                    screens.append({
                                        'index': screen_index,
                                        'width': main_width,
                                        'height': main_height,
                                        'x': -main_width,
                                        'y': 0,
                                        'is_primary': False
                                    })
                                    screen_index += 1

                            test_window.destroy()

                        except Exception:
                            # 测试失败，忽略这个位置
                            continue

                        # 限制检测的屏幕数量
                        if len(screens) >= 4:
                            break

            except Exception:
                # 多屏检测失败，只使用主屏幕
                pass

            temp_root.destroy()

            return screens

        except Exception as e:
            print(f"[WARNING] 屏幕信息获取失败: {e}")
            # 返回默认主屏幕信息
            return [{
                'index': 0,
                'width': 1920,  # 默认分辨率
                'height': 1080,
                'x': 0,
                'y': 0,
                'is_primary': True
            }]

    def _get_target_screen_bounds(self):
        """
        获取目标屏幕的边界信息

        Returns:
            dict: 目标屏幕的边界信息 {'x', 'y', 'width', 'height'}
        """
        screens = self._get_screen_info()

        # 查找目标屏幕
        target_screen = None
        for screen in screens:
            if screen['index'] == self.target_screen:
                target_screen = screen
                break

        # 如果找不到目标屏幕，使用主屏幕
        if target_screen is None:
            print(f"[WARNING] 屏幕 {self.target_screen} 不存在，使用主屏幕")
            target_screen = screens[0]

        return {
            'x': target_screen['x'],
            'y': target_screen['y'],
            'width': target_screen['width'],
            'height': target_screen['height'],
            'screen_index': target_screen['index']
        }

    def _get_target_screen_display_size(self):
        """
        获取目标屏幕的可用显示尺寸

        Returns:
            tuple: (可用宽度, 可用高度)
        """
        try:
            screen_bounds = self._get_target_screen_bounds()
            screen_width = screen_bounds['width']
            screen_height = screen_bounds['height']

            # 留出边距以避免窗口超出屏幕边界
            # 通常预留10%的边距用于任务栏、窗口标题栏等
            usable_width = int(screen_width * 0.9)
            usable_height = int(screen_height * 0.9)

            return usable_width, usable_height

        except Exception as e:
            print(f"[WARNING] 获取目标屏幕尺寸失败: {e}")
            # 回退到默认尺寸
            return 800, 600

    def _build_commands_help(self) -> Dict[str, str]:
        """构建命令帮助信息"""
        return {
            "基本操作": {
                "click [ID]": "点击元素，如: click [10]",
                "type [ID] [文本] [1]": "在元素中输入文本，如: type [5] [blue kayak] [1]",
                "hover [ID]": "悬停在元素上，如: hover [15]",
                "scroll [方向]": "滚动页面，如: scroll up 或 scroll down",
            },
            "页面操作": {
                "key_press [按键]": "按键操作，如: key_press Enter",
                "goto [URL]": "跳转到URL，如: goto http://example.com",
                "go_back": "返回上一页",
                "go_forward": "前进到下一页",
                "new_tab": "打开新标签",
                "close_tab": "关闭当前标签"
            },
            "控制命令": {
                "stop [答案]": "停止当前任务并提交答案",
                "reset": "重置当前任务",
                "skip": "跳过当前任务",
                "help": "显示帮助信息",
                "quit": "退出程序"
            }
        }

    def _print_separator(self, char: str = "=", length: Optional[int] = None) -> None:
        """打印分隔线"""
        if length is None:
            length = self.terminal_width
        print(char * length)

    def _print_centered(self, text: str, char: str = " ", length: Optional[int] = None) -> None:
        """打印居中文本"""
        if length is None:
            length = self.terminal_width
        padding = max(0, (length - len(text)) // 2)
        print(char * padding + text + char * (length - padding - len(text)))

    def clear_screen(self) -> None:
        """清屏"""
        os.system('cls' if os.name == 'nt' else 'clear')

    def display_welcome(self) -> None:
        """显示欢迎信息"""
        self.clear_screen()
        self._print_separator()
        self._print_centered("交互式Web任务标注工具")
        self._print_centered("Interactive Web Task Annotation Tool")
        self._print_separator()
        print()

    def display_progress(self, progress_info: Dict[str, Any]) -> None:
        """显示进度信息"""
        print("📊 进度统计:")
        print(f"   总任务数: {progress_info['总任务数']}")
        print(f"   已完成: {progress_info['已完成']}")
        print(f"   剩余: {progress_info['剩余']}")
        print(f"   完成率: {progress_info['完成率']}")

        print("\n📋 按环境统计:")
        for env, stats in progress_info['按环境统计'].items():
            print(f"   {env}: {stats['已完成']}/{stats['总数']} "
                  f"(剩余: {stats['剩余']})")

        if progress_info.get('当前任务'):
            print(f"\n🔄 当前任务: {progress_info['当前任务']}")

        print()

    def display_task_info(self, env_name: str, task: Dict[str, Any]) -> None:
        """
        显示任务信息

        Args:
            env_name: 环境名
            task: 任务数据
        """
        self._print_separator("-")
        print(f"🎯 当前任务 - {env_name.upper()} 环境")
        self._print_separator("-")

        print(f"📝 任务ID: {task['task_id']}")
        print(f"🎯 任务描述: {task['intent']}")

        if task.get('start_url'):
            print(f"🌐 起始URL: {task['start_url']}")

        if task.get('require_login'):
            print(f"🔐 需要登录: {'是' if task['require_login'] else '否'}")

        if task.get('storage_state'):
            print(f"🍪 登录状态文件: {task['storage_state']}")

        # 显示图片信息
        if task.get('image'):
            print(f"\n[IMAGES] 任务相关图片:")
            self.display_task_images(task['image'])

        print()

    def display_task_images(self, image_data: Any) -> None:
        """
        显示任务相关图片信息

        Args:
            image_data: 图片数据，可能是字符串、列表或None
        """
        if not image_data:
            return

        images = []
        if isinstance(image_data, str):
            images = [image_data]
        elif isinstance(image_data, list):
            images = image_data

        for i, image_path in enumerate(images, 1):
            print(f"   图片{i}: {image_path}")

            # 尝试获取图片信息和显示图片
            try:
                if image_path.startswith('http'):
                    # 网络图片
                    response = requests.head(image_path, timeout=5)
                    if response.status_code == 200:
                        print(f"        类型: 网络图片 (状态码: {response.status_code})")
                        # 网络图片暂不支持直接显示
                        print(f"        [INFO] 请在浏览器中打开: {image_path}")
                    else:
                        print(f"        类型: 网络图片 (无法访问: {response.status_code})")
                else:
                    # 本地图片
                    image_file = Path(image_path)
                    if image_file.exists():
                        try:
                            with Image.open(image_file) as img:
                                print(f"        类型: 本地图片 ({img.format}, {img.size})")

                            # 尝试直接显示本地图片
                            if self.image_display_method != 'off':
                                print(f"        [IMAGE] 正在显示图片{i}...")
                                success = self.display_image_direct(str(image_file), f"任务图片 {i}")
                                if not success:
                                    print(f"        [WARNING] 显示失败，请手动查看")

                        except Exception as e:
                            print(f"        类型: 本地文件 (无法解析: {e})")
                    else:
                        print(f"        类型: 本地图片 (文件不存在)")
            except Exception as e:
                print(f"        类型: 未知 (检查失败: {e})")

    def display_screenshot_ready(self, screenshot_path: Optional[str] = None) -> None:
        """
        显示屏幕截图就绪信息

        Args:
            screenshot_path: 截图文件路径
        """
        # 显示任务上下文提醒
        context_summary = self.get_task_context_summary()
        if context_summary:
            print(f"[任务状态] {context_summary}")

        print("[SCREENSHOT] 环境屏幕截图已准备就绪")
        if screenshot_path:
            print(f"   截图保存位置: {screenshot_path}")

            # 尝试直接显示图片
            if self.image_display_method != 'off':
                success = self.display_image_direct(screenshot_path, "当前页面截图")
                if success:
                    print("   [OK] 图片已显示")
                else:
                    print("   [WARNING] 图片显示失败，请手动查看文件")
            else:
                print("   [INFO] 提示: 可手动打开图片文件查看")
        print()

    def display_help(self) -> None:
        """显示帮助信息"""
        print("📖 命令帮助:")
        self._print_separator("-")

        for category, commands in self.commands_help.items():
            print(f"\n{category}:")
            for cmd, desc in commands.items():
                print(f"  {cmd:<20} - {desc}")

        print()
        print("[INFO] 提示:")
        print("  - 元素ID可以从截图上的标注中找到")
        print("  - 输入文本时请使用英文方括号 []")
        print("  - 使用 'stop [答案]' 完成任务")
        print("  - 使用 'reset' 重新开始当前任务")
        print()

    def prompt_user_input(self) -> str:
        """
        提示用户输入操作

        Returns:
            用户输入的命令字符串
        """
        # 显示任务上下文信息
        self.display_current_task_context()

        print("[INPUT] 请输入操作命令 (输入 'help' 查看帮助):")
        try:
            user_input = input(">>> ").strip()
            return user_input
        except KeyboardInterrupt:
            print("\n\n[EXIT] 检测到 Ctrl+C，正在退出...")
            return "quit"
        except EOFError:
            print("\n\n[EXIT] 检测到输入结束，正在退出...")
            return "quit"

    def prompt_user_input_with_gui(self) -> str:
        """
        提示用户输入操作，同时处理GUI事件

        Returns:
            用户输入的命令字符串
        """
        # 显示任务上下文信息
        self.display_current_task_context()

        print("[INPUT] 请输入操作命令 (输入 'help' 查看帮助):")
        print(">>> ", end="", flush=True)

        try:
            import sys
            import select
            import time

            # 调试信息已注释，保持输出简洁
            # print(f"\n[DEBUG] 系统平台: {sys.platform}")
            # print(f"[DEBUG] 是否有活跃窗口: {self.has_active_window()}")
            # print(f"[DEBUG] 持续窗口模式: {self.persistent_window}")

            # Windows平台的非阻塞输入实现
            if sys.platform == 'win32':
                try:
                    import msvcrt
                    # print("[DEBUG] 成功导入 msvcrt 模块")
                    # print("[DEBUG] 测试 msvcrt.kbhit 可用性...")
                    # test_result = msvcrt.kbhit()
                    # print(f"[DEBUG] msvcrt.kbhit() 初始测试结果: {test_result}")

                except ImportError as e:
                    print(f"[ERROR] msvcrt 导入失败: {e}")
                    print("[FALLBACK] 回退到标准输入模式")
                    return self.prompt_user_input()

                input_buffer = ""
                loop_count = 0
                last_debug_time = time.time()

                # print("[DEBUG] 进入输入检测循环...")

                # 如果有GUI窗口，给用户提示
                if self.has_active_window():
                    print("[INFO] GUI输入模式已启用，如果输入无响应会自动回退到标准输入")
                    # print("  - 请确保控制台窗口有焦点（点击控制台窗口）")
                    # print("  - 或等待10秒后程序会自动回退到标准输入模式")

                while True:
                    loop_count += 1
                    current_time = time.time()

                    # 定期检查状态（调试信息已注释）
                    if current_time - last_debug_time >= 3.0:
                        # print(f"\n[DEBUG] 循环运行中... 执行次数: {loop_count}")
                        # print(f"[DEBUG] 当前输入缓冲区: '{input_buffer}'")
                        # print(f"[DEBUG] 窗口状态: {self.has_active_window()}")
                        # if self.has_active_window():
                        #     print("[HINT] 如果无法输入，请点击控制台窗口获得键盘焦点")
                        last_debug_time = current_time

                    # 处理GUI事件
                    if self.has_active_window():
                        gui_result = self.process_gui_events()
                        # 如果需要详细GUI调试，可以取消下面的注释
                        # if loop_count % 1000 == 0:
                        #     print(f"[DEBUG] GUI事件处理结果: {gui_result}")

                    # 检查是否有键盘输入
                    kbhit_result = msvcrt.kbhit()
                    if kbhit_result:
                        # print(f"\n[DEBUG] ✅ 检测到键盘输入!")
                        char = msvcrt.getch()
                        # print(f"[DEBUG] 字符: {char} (十六进制: {char.hex()})")

                        # 处理特殊键
                        if char == b'\r' or char == b'\n':  # Enter (支持多种编码)
                            # print(f"[DEBUG] 检测到Enter键: {char}")
                            print()  # 换行
                            result = input_buffer.strip()
                            # print(f"[DEBUG] 返回输入结果: '{result}'")
                            return result
                        elif char == b'\x03':  # Ctrl+C
                            print("\n\n[EXIT] 检测到 Ctrl+C，正在退出...")
                            return "quit"
                        elif char == b'\x08':  # Backspace
                            if input_buffer:
                                input_buffer = input_buffer[:-1]
                                print('\b \b', end='', flush=True)
                                # print(f"[DEBUG] Backspace处理，当前缓冲区: '{input_buffer}'")
                        elif char == b'\x1a':  # Ctrl+Z (EOF)
                            print("\n\n[EXIT] 检测到输入结束，正在退出...")
                            return "quit"
                        else:
                            try:
                                # 尝试解码字符
                                decoded_char = char.decode('utf-8', errors='ignore')
                                # print(f"[DEBUG] 解码字符: '{decoded_char}' (可打印: {decoded_char.isprintable()})")
                                if decoded_char.isprintable():
                                    input_buffer += decoded_char
                                    print(decoded_char, end='', flush=True)
                                    # print(f"[DEBUG] 字符已添加，当前缓冲区: '{input_buffer}'")
                            except Exception as decode_error:
                                # print(f"[DEBUG] 字符解码失败: {decode_error}")
                                pass

                    # 快速超时检测（当有GUI窗口时）
                    if self.has_active_window() and loop_count > 1000:  # 10秒后快速超时
                        print(f"\n[TIMEOUT] 检测到焦点问题，循环执行了 {loop_count} 次")
                        print("[INFO] 这通常是因为GUI窗口占用了键盘焦点")
                        print("[FALLBACK] 自动回退到标准输入模式")
                        return self.prompt_user_input()
                    elif loop_count > 30000:  # 无GUI窗口时的正常超时（约5分钟）
                        print(f"\n[TIMEOUT] 输入检测超时，循环执行了 {loop_count} 次")
                        print("[FALLBACK] 回退到标准输入模式")
                        return self.prompt_user_input()

                    # 短暂休眠，避免100% CPU占用
                    time.sleep(0.01)

            else:
                # Unix/Linux平台的实现
                while True:
                    # 处理GUI事件
                    if self.has_active_window():
                        self.process_gui_events()

                    # 检查stdin是否有数据
                    ready, _, _ = select.select([sys.stdin], [], [], 0.01)
                    if ready:
                        line = sys.stdin.readline()
                        if not line:  # EOF
                            print("\n[EXIT] 检测到输入结束，正在退出...")
                            return "quit"
                        return line.strip()

        except KeyboardInterrupt:
            print("\n\n[EXIT] 检测到 Ctrl+C，正在退出...")
            return "quit"
        except Exception as e:
            print(f"\n[WARNING] GUI输入处理异常: {e}")
            # print(f"[DEBUG] 异常类型: {type(e).__name__}")
            # import traceback
            # print(f"[DEBUG] 异常堆栈: {traceback.format_exc()}")
            print("[FALLBACK] 回退到标准输入方式")
            # 回退到标准输入方式
            return self.prompt_user_input()

    def display_action_result(self, success: bool, action_description: str,
                            error_message: Optional[str] = None) -> None:
        """
        显示动作执行结果

        Args:
            success: 是否执行成功
            action_description: 动作描述
            error_message: 错误信息（如果有）
        """
        if success:
            print(f"[OK] 动作执行成功: {action_description}")
            # 成功时显示当前任务进度
            context_summary = self.get_task_context_summary()
            if context_summary:
                print(f"[进度更新] {context_summary}")
        else:
            print(f"[FAIL] 动作执行失败: {action_description}")
            if error_message:
                print(f"   错误详情: {error_message}")
        print()

    def display_task_evaluation_result(self, success: bool, score: float,
                                     task_id: str) -> None:
        """
        显示任务评估结果

        Args:
            success: 任务是否成功
            score: 评估分数
            task_id: 任务ID
        """
        self._print_separator("=")
        if success:
            print(f"🎉 任务完成! 任务 {task_id} 评估成功")
            print(f"⭐ 评估分数: {score}")
            print("💾 轨迹已保存到标注目录")
        else:
            print(f"😔 任务失败! 任务 {task_id} 评估未通过")
            print(f"⭐ 评估分数: {score}")
            print("🔄 请选择:")
            print("   - 输入 'reset' 重新开始当前任务")
            print("   - 输入 'skip' 跳过当前任务")
        self._print_separator("=")
        print()

    def display_error(self, error_message: str) -> None:
        """
        显示错误信息

        Args:
            error_message: 错误信息
        """
        print(f"[ERROR] 错误: {error_message}")
        print()

    def display_warning(self, warning_message: str) -> None:
        """
        显示警告信息

        Args:
            warning_message: 警告信息
        """
        print(f"[WARNING] 警告: {warning_message}")
        print()

    def display_info(self, info_message: str) -> None:
        """
        显示信息

        Args:
            info_message: 信息内容
        """
        print(f"ℹ️ 信息: {info_message}")
        print()

    def confirm_action(self, message: str) -> bool:
        """
        确认操作

        Args:
            message: 确认消息

        Returns:
            用户是否确认
        """
        try:
            response = input(f"❓ {message} (y/N): ").strip().lower()
            return response in ['y', 'yes', '是', 'Y']
        except (KeyboardInterrupt, EOFError):
            return False

    def display_completion_message(self) -> None:
        """显示所有任务完成信息"""
        self._print_separator("=")
        self._print_centered("🎊 恭喜! 所有任务已完成! 🎊")
        self._print_separator("=")
        print()
        print("感谢您使用交互式Web任务标注工具!")
        print("所有标注轨迹已保存到 data/annotate 目录中。")
        print()

    def set_task_context(self, env_name: str = None, task_id: str = None,
                        task_intent: str = None, start_url: str = None,
                        current_url: str = None, step_count: int = None) -> None:
        """
        设置任务上下文信息

        Args:
            env_name: 环境名称
            task_id: 任务ID
            task_intent: 任务描述
            start_url: 起始URL
            current_url: 当前URL
            step_count: 步骤计数
        """
        if env_name is not None:
            self.current_task_context['env_name'] = env_name
        if task_id is not None:
            self.current_task_context['task_id'] = task_id
        if task_intent is not None:
            self.current_task_context['task_intent'] = task_intent
        if start_url is not None:
            self.current_task_context['start_url'] = start_url
        if current_url is not None:
            self.current_task_context['current_url'] = current_url
        if step_count is not None:
            self.current_task_context['step_count'] = step_count

    def get_task_context_summary(self) -> str:
        """
        获取任务上下文摘要字符串

        Returns:
            格式化的任务上下文摘要
        """
        context = self.current_task_context

        if not context['env_name']:
            return ""

        # 构建摘要信息
        summary_parts = []

        # 环境和任务ID
        if context['env_name'] and context['task_id']:
            summary_parts.append(f"环境: {context['env_name'].upper()} | ID: {context['task_id']}")

        # 任务描述（完整显示）
        if context['task_intent']:
            summary_parts.append(f"目标: {context['task_intent']}")

        # 当前URL信息
        if context['current_url']:
            # 显示完整URL，不截断
            try:
                full_url = context['current_url']

                # 判断是否为起始页面
                if context['start_url'] and context['current_url'] != context['start_url']:
                    summary_parts.append(f"页面: {full_url}")
                else:
                    summary_parts.append(f"起始页面: {full_url}")
            except:
                summary_parts.append(f"页面: {context['current_url']}")
        elif context['start_url']:
            try:
                full_url = context['start_url']
                summary_parts.append(f"起始页面: {full_url}")
            except:
                summary_parts.append(f"起始页面: {context['start_url']}")

        # 步骤计数
        if context['step_count'] is not None and context['step_count'] > 0:
            summary_parts.append(f"步骤: {context['step_count']}")

        return " | ".join(summary_parts)

    def display_current_task_context(self) -> None:
        """显示当前任务上下文信息"""
        summary = self.get_task_context_summary()
        if summary:
            print(f"[当前任务] {summary}")

    def clear_task_context(self) -> None:
        """清除任务上下文信息"""
        self.current_task_context = {
            'env_name': None,
            'task_id': None,
            'task_intent': None,
            'start_url': None,
            'current_url': None,
            'step_count': 0
        }

    def wait_for_continue(self) -> None:
        """等待用户按键继续"""
        try:
            input("按 Enter 键继续...")
        except (KeyboardInterrupt, EOFError):
            pass

    def _resolve_display_method(self, method: str) -> str:
        """
        解析和确定图片显示方法

        Args:
            method: 用户指定的显示方法

        Returns:
            实际可用的显示方法
        """
        if method == 'auto':
            # 自动选择最佳方法
            if HAS_TKINTER:
                return 'tkinter'
            elif HAS_SYSTEM_VIEWER:
                return 'system'
            else:
                return 'ascii'
        elif method == 'tkinter' and not HAS_TKINTER:
            print("[WARNING] Tkinter不可用，降级为ASCII显示")
            return 'ascii'
        elif method == 'system' and not HAS_SYSTEM_VIEWER:
            print("[WARNING] 系统图片查看器不可用，降级为ASCII显示")
            return 'ascii'
        else:
            return method

    def display_image_direct(self, image_path: str, title: str = "图片显示") -> bool:
        """
        直接在程序中显示图片

        Args:
            image_path: 图片文件路径
            title: 显示标题

        Returns:
            是否成功显示
        """
        if not image_path or not os.path.exists(image_path):
            self.display_error(f"图片文件不存在: {image_path}")
            return False

        try:
            if self.image_display_method == 'tkinter':
                return self._display_with_tkinter(image_path, title)
            elif self.image_display_method == 'ascii':
                return self._display_with_ascii(image_path, title)
            elif self.image_display_method == 'system':
                return self._display_with_system_viewer(image_path)
            elif self.image_display_method == 'off':
                # 只显示路径，不显示图片
                print(f"[IMAGE] 图片路径: {image_path}")
                return True
            else:
                self.display_warning(f"不支持的显示方法: {self.image_display_method}")
                return False

        except Exception as e:
            self.display_error(f"显示图片时发生错误: {e}")
            return False

    def _display_with_tkinter(self, image_path: str, title: str) -> bool:
        """
        使用Tkinter显示图片

        Args:
            image_path: 图片路径
            title: 窗口标题

        Returns:
            是否成功显示
        """
        try:
            # 如果已有窗口且是持续显示模式，更新现有窗口
            if self.current_window and self.persistent_window:
                return self._update_existing_window(image_path, title)

            # 关闭旧窗口（如果存在）
            self._close_current_window()

            # 创建新窗口
            root = tk.Tk()
            root.title(title)
            self.current_window = root

            # 加载并显示图片
            success = self._setup_image_display(root, image_path)
            if not success:
                self._close_current_window()
                return False

            # 设置窗口行为
            self._configure_window_behavior(root)

            print(f"[IMAGE] 图片已在新窗口中打开: {os.path.basename(image_path)}")
            if self.persistent_window:
                print("   窗口将保持打开状态，您可以继续输入操作")
                # 非阻塞显示，由主程序负责调用事件处理
                # 完整的窗口初始化，确保响应性
                root.update_idletasks()  # 处理几何管理
                root.update()            # 处理初始用户交互事件
            else:
                print("   按 ESC 或 Enter 键关闭窗口")
                # 阻塞显示，等待用户关闭窗口
                root.mainloop()

            return True

        except Exception as e:
            print(f"Tkinter显示失败: {e}")
            self._close_current_window()
            return False

    def _setup_image_display(self, root: tk.Tk, image_path: str) -> bool:
        """
        设置图片显示

        Args:
            root: Tkinter窗口对象
            image_path: 图片路径

        Returns:
            是否设置成功
        """
        try:
            # 加载图片
            pil_image = Image.open(image_path)
            original_size = pil_image.size

            # 根据配置决定是否缩放
            display_image = self._process_image_scaling(pil_image)

            # 转换为Tkinter格式
            from PIL import ImageTk
            photo = ImageTk.PhotoImage(display_image)

            # 清除旧内容
            for widget in root.winfo_children():
                widget.destroy()

            # 创建图片框架
            image_frame = tk.Frame(root)
            image_frame.pack(padx=10, pady=10)

            # 显示图片
            label = tk.Label(image_frame, image=photo)
            label.image = photo  # 保持引用防止垃圾回收
            label.pack()

            # 显示图片信息
            self._add_image_info(root, image_path, original_size, display_image.size)

            return True

        except Exception as e:
            print(f"图片设置失败: {e}")
            return False

    def _process_image_scaling(self, pil_image: Image.Image) -> Image.Image:
        """
        处理图片缩放

        Args:
            pil_image: PIL图片对象

        Returns:
            处理后的图片对象
        """
        if not self.keep_aspect_ratio:
            # 如果不保持比例，按窗口尺寸缩放
            try:
                # 如果用户指定了具体尺寸，使用指定尺寸
                if self.image_window_size != "auto":
                    width, height = map(int, self.image_window_size.split('x'))
                else:
                    # 如果是自动尺寸，使用目标屏幕的合适尺寸
                    width, height = self._get_target_screen_display_size()

                # 调试信息
                if self.target_screen > 0:
                    print(f"[DEBUG] 不保持比例模式，目标尺寸: {width}x{height}")

                return pil_image.resize((width, height), Image.Resampling.LANCZOS)
            except Exception as e:
                print(f"[WARNING] 窗口尺寸解析失败: {e}")
                # 如果解析失败，使用目标屏幕尺寸
                try:
                    width, height = self._get_target_screen_display_size()
                    return pil_image.resize((width, height), Image.Resampling.LANCZOS)
                except:
                    # 最后回退到原图
                    return pil_image

        # 保持原始比例，但限制最大尺寸以适应目标屏幕
        img_width, img_height = pil_image.size

        # 获取目标屏幕尺寸（而不是主屏幕尺寸）
        max_height, max_width = self._get_target_screen_display_size()

        # 调试信息
        if self.target_screen > 0:
            print(f"[DEBUG] 目标屏幕 {self.target_screen} 可用尺寸: {max_width}x{max_height}")
            print(f"[DEBUG] 原始图片尺寸: {img_width}x{img_height}")

        # 如果图片超出屏幕，按比例缩放
        if img_width > max_width or img_height > max_height:
            scale_width = max_width / img_width
            scale_height = max_height / img_height
            scale = min(scale_width, scale_height)

            new_width = int(img_width * scale)
            new_height = int(img_height * scale)

            # 调试信息
            if self.target_screen > 0:
                print(f"[DEBUG] 缩放比例: {scale:.3f}, 缩放后尺寸: {new_width}x{new_height}")

            return pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # 原尺寸适合屏幕，直接返回
        return pil_image

    def _add_image_info(self, root: tk.Tk, image_path: str, original_size: tuple, display_size: tuple):
        """
        添加图片信息显示

        Args:
            root: Tkinter窗口
            image_path: 图片路径
            original_size: 原始尺寸
            display_size: 显示尺寸
        """
        info_frame = tk.Frame(root)
        info_frame.pack(pady=(0, 10))

        # 图片信息
        info_text = f"文件: {os.path.basename(image_path)}\n"
        info_text += f"原始尺寸: {original_size[0]}x{original_size[1]}"

        if original_size != display_size:
            scale = display_size[0] / original_size[0]
            info_text += f"\n显示尺寸: {display_size[0]}x{display_size[1]}"
            info_text += f"\n显示比例: {scale:.1%}"

        info_label = tk.Label(info_frame, text=info_text, fg="gray", justify=tk.LEFT)
        info_label.pack()

    def _configure_window_behavior(self, root: tk.Tk):
        """
        配置窗口行为，支持跨屏幕显示

        Args:
            root: Tkinter窗口
        """
        try:
            # 获取目标屏幕边界
            screen_bounds = self._get_target_screen_bounds()
            screen_x = screen_bounds['x']
            screen_y = screen_bounds['y']
            screen_width = screen_bounds['width']
            screen_height = screen_bounds['height']

            # 获取窗口实际尺寸
            root.update_idletasks()
            window_width = root.winfo_reqwidth()
            window_height = root.winfo_reqheight()

            # 计算在目标屏幕上的左上角位置，留适当边距
            pos_x = screen_x + 30  # 左边距20像素
            pos_y = screen_y + 50  # 上边距50像素，避开系统菜单栏

            # 确保窗口完全在目标屏幕内
            pos_x = max(screen_x, min(pos_x, screen_x + screen_width - window_width))
            pos_y = max(screen_y, min(pos_y, screen_y + screen_height - window_height))

            pos_x = 2078
            pos_y = -200

            print(f"pos_x: {pos_x}, pos_y: {pos_y}")

            # 设置窗口位置和大小
            root.geometry(f"{window_width}x{window_height}+{pos_x}+{pos_y}")

            # 调试信息
            if self.target_screen > 0:
                print(f"[DEBUG] 窗口显示在屏幕 {self.target_screen}: 位置({pos_x}, {pos_y}), 尺寸({window_width}x{window_height})")
                print(f"[DEBUG] 目标屏幕范围: x={screen_x}, y={screen_y}, 宽={screen_width}, 高={screen_height}")

        except Exception as e:
            print(f"[WARNING] 跨屏幕窗口配置失败: {e}")
            # 回退到原始行为
            root.update_idletasks()
            window_width = root.winfo_reqwidth()
            window_height = root.winfo_reqheight()
            screen_width = root.winfo_screenwidth()
            screen_height = root.winfo_screenheight()
            pos_x = 20  # 左上角位置，左边距20像素
            pos_y = 50  # 上边距50像素
            root.geometry(f"{window_width}x{window_height}+{pos_x}+{pos_y}")

        # 设置快捷键
        if not self.persistent_window:
            # 非持续模式：ESC和Enter关闭窗口
            root.bind('<Escape>', lambda e: root.destroy())
            root.bind('<Return>', lambda e: root.destroy())

        # 窗口关闭事件处理
        root.protocol("WM_DELETE_WINDOW", self._on_window_close)

        # 不设置GUI窗口焦点，保持控制台键盘输入焦点
        # root.focus_set()  # 注释掉，避免抢夺控制台焦点

    def _update_existing_window(self, image_path: str, title: str) -> bool:
        """
        更新现有窗口的内容

        Args:
            image_path: 新的图片路径
            title: 新的窗口标题

        Returns:
            是否更新成功
        """
        try:
            if not self.current_window or not self.current_window.winfo_exists():
                # 窗口不存在，创建新窗口
                self.current_window = None
                return self._display_with_tkinter(image_path, title)

            # 更新窗口标题
            self.current_window.title(title)

            # 更新图片内容
            success = self._setup_image_display(self.current_window, image_path)

            # 将窗口置于前台，但不设置topmost以避免抢夺焦点
            self.current_window.lift()
            # 移除topmost设置，保持控制台键盘输入焦点
            # self.current_window.attributes('-topmost', True)
            # self.current_window.after_idle(lambda: self.current_window.attributes('-topmost', False))

            print(f"[IMAGE] 图片已更新: {os.path.basename(image_path)}")

            return success

        except Exception as e:
            print(f"窗口更新失败: {e}")
            # 创建新窗口
            self.current_window = None
            return self._display_with_tkinter(image_path, title)

    def process_gui_events(self):
        """
        处理GUI事件（被动调用模式）
        主程序应该定期调用此方法来保持窗口响应性

        Returns:
            bool: 窗口是否仍然存在
        """
        try:
            if self.current_window and self.current_window.winfo_exists():
                # 处理所有待处理的GUI事件
                self.current_window.update_idletasks()  # 处理几何管理等
                self.current_window.update()  # 处理用户交互事件
                return True
            else:
                # 窗口不存在，清理引用
                self.current_window = None
                return False
        except tk.TclError:
            # 窗口已关闭或出错
            self.current_window = None
            return False
        except Exception as e:
            print(f"[WARNING] GUI事件处理异常: {e}")
            return False

    def has_active_window(self):
        """
        检查是否有活跃的窗口

        Returns:
            bool: 是否有活跃窗口
        """
        try:
            return (self.current_window is not None and
                    self.current_window.winfo_exists())
        except:
            return False

    def get_window_status(self):
        """
        获取窗口状态信息

        Returns:
            dict: 窗口状态信息
        """
        status = {
            "has_window": self.current_window is not None,
            "window_exists": False,
            "window_title": "",
            "persistent_mode": self.persistent_window
        }

        try:
            if self.current_window:
                status["window_exists"] = self.current_window.winfo_exists()
                if status["window_exists"]:
                    status["window_title"] = self.current_window.title()
        except:
            status["window_exists"] = False

        return status

    def ensure_window_responsiveness(self):
        """
        确保窗口保持响应性
        定期调用此方法以维持GUI响应

        Returns:
            bool: 窗口是否仍然活跃
        """
        if not self.has_active_window():
            return False

        try:
            # 处理GUI事件
            self.process_gui_events()

            # 检查窗口是否被用户关闭
            if not self.current_window.winfo_exists():
                self.current_window = None
                return False

            return True

        except Exception as e:
            print(f"[WARNING] 窗口响应性维护异常: {e}")
            self.current_window = None
            return False

    def _close_current_window(self):
        """关闭当前窗口"""
        if self.current_window:
            try:
                if self.current_window.winfo_exists():
                    self.current_window.destroy()
            except:
                pass
            finally:
                self.current_window = None

    def _on_window_close(self):
        """窗口关闭事件处理"""
        self._close_current_window()

    def _display_with_ascii(self, image_path: str, title: str, width: int = None) -> bool:
        """
        使用ASCII艺术在终端显示图片

        Args:
            image_path: 图片路径
            title: 显示标题
            width: ASCII显示宽度

        Returns:
            是否成功显示
        """
        try:
            # 使用实例属性设置默认宽度
            if width is None:
                width = self.ascii_width

            # 打开图片
            image = Image.open(image_path)

            # 转换为灰度
            gray_image = image.convert('L')

            # 计算高度（保持宽高比）
            img_width, img_height = gray_image.size
            aspect_ratio = img_height / img_width
            height = int(width * aspect_ratio * 0.5)  # 0.5是因为字符高度约为宽度的2倍

            # 调整图片尺寸
            resized = gray_image.resize((width, height))

            # ASCII字符集（从暗到亮）
            ascii_chars = "@%#*+=-:. "

            print(f"\n[IMAGE] {title}")
            print(f"文件: {os.path.basename(image_path)}")
            print(f"原始尺寸: {img_width}x{img_height}")
            print("-" * width)

            # 转换为ASCII
            pixels = resized.getdata()
            ascii_str = ""
            for pixel_value in pixels:
                ascii_str += ascii_chars[pixel_value * (len(ascii_chars) - 1) // 255]

            # 按行显示
            for i in range(0, len(ascii_str), width):
                print(ascii_str[i:i + width])

            print("-" * width)
            print()

            return True

        except Exception as e:
            print(f"ASCII显示失败: {e}")
            return False

    def _display_with_system_viewer(self, image_path: str) -> bool:
        """
        使用系统默认图片查看器显示图片

        Args:
            image_path: 图片路径

        Returns:
            是否成功启动查看器
        """
        try:
            system = platform.system()

            if system == "Windows":
                os.startfile(image_path)
            elif system == "Darwin":  # macOS
                subprocess.run(["open", image_path])
            elif system == "Linux":
                subprocess.run(["xdg-open", image_path])
            else:
                print(f"不支持的操作系统: {system}")
                return False

            print(f"[IMAGE] 已使用系统默认查看器打开: {os.path.basename(image_path)}")
            return True

        except Exception as e:
            print(f"系统查看器打开失败: {e}")
            return False

    def set_image_display_method(self, method: str) -> bool:
        """
        设置图片显示方法

        Args:
            method: 显示方法 ('tkinter', 'ascii', 'system', 'off', 'auto')

        Returns:
            是否设置成功
        """
        old_method = self.image_display_method
        new_method = self._resolve_display_method(method)

        if new_method != method and method != 'auto':
            print(f"[WARNING] 请求的显示方法 '{method}' 不可用，使用 '{new_method}'")

        self.image_display_method = new_method
        print(f"📺 图片显示方法已更改: {old_method} -> {new_method}")

        return True

    def get_image_display_info(self) -> Dict[str, Any]:
        """
        获取图片显示相关信息

        Returns:
            显示信息字典
        """
        return {
            "current_method": self.image_display_method,
            "available_methods": {
                "tkinter": HAS_TKINTER,
                "ascii": True,  # 总是可用
                "system": HAS_SYSTEM_VIEWER,
                "off": True,  # 总是可用
            },
            "capabilities": {
                "gui_window": HAS_TKINTER,
                "terminal_display": True,
                "system_viewer": HAS_SYSTEM_VIEWER,
            }
        }

    def show_image_display_help(self) -> None:
        """显示图片显示帮助信息"""
        print("\n📺 图片显示功能说明:")
        print("-" * 40)

        info = self.get_image_display_info()
        print(f"当前显示方法: {info['current_method']}")

        print("\n可用的显示方法:")
        methods = {
            "tkinter": "GUI窗口显示 (推荐)",
            "ascii": "终端ASCII艺术显示",
            "system": "系统默认图片查看器",
            "off": "关闭图片显示",
            "auto": "自动选择最佳方法"
        }

        for method, description in methods.items():
            available = info['available_methods'].get(method, False)
            status = "[OK]" if available else "[NO]"
            current = "👈 当前" if method == info['current_method'] else ""
            print(f"  {status} {method:<8} - {description} {current}")

        print("\n[INFO] 使用提示:")
        print("  - Tkinter显示效果最佳，支持缩放和快捷键")
        print("  - ASCII显示适用于无GUI环境")
        print("  - 系统查看器会打开默认图片应用")
        print("  - 可在程序运行时更改显示方法")
        print()

    def cleanup(self):
        """
        清理资源，关闭所有打开的窗口
        """
        self._close_current_window()
        print("[INFO] 图片显示资源已清理")