"""
输入解析器模块
负责将用户的文本输入转换为可执行的Action对象
"""

import re
from typing import Dict, Any, Optional, Tuple, Union
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent / "visualwebarena"))

from visualwebarena.src.envs.actions import (
    Action,
    ActionTypes,
    create_click_action,
    create_type_action,
    create_hover_action,
    create_scroll_action,
    create_key_press_action,
    create_goto_url_action,
    create_stop_action,
    create_go_back_action,
    create_go_forward_action,
    create_new_tab_action,
    create_page_close_action,
    create_none_action,
)


class InputParseError(Exception):
    """输入解析错误"""
    pass


class InputParser:
    """输入解析器类"""

    def __init__(self):
        """初始化解析器"""
        self.command_patterns = {
            # 基本操作
            'click': r'^click\s+\[(\d+)\]$',
            'type': r'^type\s+\[(\d+)\]\s+\[([^\]]+)\](?:\s+\[(\d+)\])?$',
            'hover': r'^hover\s+\[(\d+)\]$',

            # 滚动操作
            'scroll': r'^scroll\s+(up|down)$',

            # 键盘操作
            'key_press': r'^(?:key_press|press)\s+([^\s]+)$',

            # 页面导航
            'goto': r'^goto\s+(.+)$',
            'go_back': r'^go_back$',
            'go_forward': r'^go_forward$',

            # 标签操作
            'new_tab': r'^new_tab$',
            'close_tab': r'^close_tab$',

            # 控制命令
            'stop': r'^stop(?:\s+(.*))?$',
            'reset': r'^reset$',
            'skip': r'^skip$',
            'help': r'^help$',
            'quit': r'^quit$',
        }

        # 编译正则表达式
        self.compiled_patterns = {
            name: re.compile(pattern, re.IGNORECASE)
            for name, pattern in self.command_patterns.items()
        }

    def parse_input(self, user_input: str) -> Tuple[str, Optional[Action], Optional[Dict[str, Any]]]:
        """
        解析用户输入

        Args:
            user_input: 用户输入的命令字符串

        Returns:
            (命令类型, Action对象, 额外信息) 的元组
            - 命令类型: 'action' 表示浏览器动作，'control' 表示控制命令
            - Action对象: 如果是浏览器动作，返回对应的Action对象，否则为None
            - 额外信息: 包含解析出的参数等信息
        """
        user_input = user_input.strip()
        if not user_input:
            raise InputParseError("输入不能为空")

        # 尝试匹配各种命令模式
        for command_name, pattern in self.compiled_patterns.items():
            match = pattern.match(user_input)
            if match:
                return self._process_command(command_name, match, user_input)

        # 如果没有匹配的模式
        raise InputParseError(f"无法识别的命令: '{user_input}'\\n请输入 'help' 查看可用命令")

    def _process_command(self, command_name: str, match: re.Match,
                        original_input: str) -> Tuple[str, Optional[Action], Optional[Dict[str, Any]]]:
        """
        处理匹配的命令

        Args:
            command_name: 命令名称
            match: 正则匹配结果
            original_input: 原始输入

        Returns:
            处理结果元组
        """
        groups = match.groups()

        try:
            if command_name == 'click':
                element_id = groups[0]
                action = create_click_action(
                    element_id=element_id,
                    element_role="generic",
                    element_name="",
                    pw_code=""
                )
                return 'action', action, {'element_id': element_id}

            elif command_name == 'type':
                element_id = groups[0]
                text = groups[1]
                # groups[2] 是可选的第三个参数，通常用于指示输入方式，这里忽略
                action = create_type_action(
                    text=text,
                    element_id=element_id,
                    element_role="generic",
                    element_name="",
                    pw_code=""
                )
                return 'action', action, {'element_id': element_id, 'text': text}

            elif command_name == 'hover':
                element_id = groups[0]
                action = create_hover_action(
                    element_id=element_id,
                    element_role="generic",
                    element_name="",
                    pw_code=""
                )
                return 'action', action, {'element_id': element_id}

            elif command_name == 'scroll':
                direction = groups[0].lower()
                action = create_scroll_action(direction)
                return 'action', action, {'direction': direction}

            elif command_name == 'key_press':
                key_combination = groups[0]
                # 处理一些常见的按键映射
                key_mapping = {
                    'enter': 'Enter',
                    'space': 'Space',
                    'tab': 'Tab',
                    'escape': 'Escape',
                    'esc': 'Escape',
                    'backspace': 'Backspace',
                    'delete': 'Delete',
                    'home': 'Home',
                    'end': 'End',
                    'pageup': 'PageUp',
                    'pagedown': 'PageDown',
                    'arrowup': 'ArrowUp',
                    'arrowdown': 'ArrowDown',
                    'arrowleft': 'ArrowLeft',
                    'arrowright': 'ArrowRight',
                    'ctrl+a': 'Control+a',
                    'ctrl+c': 'Control+c',
                    'ctrl+v': 'Control+v',
                    'ctrl+x': 'Control+x',
                    'ctrl+z': 'Control+z',
                    'ctrl+y': 'Control+y',
                }
                mapped_key = key_mapping.get(key_combination.lower(), key_combination)
                action = create_key_press_action(mapped_key)
                return 'action', action, {'key': mapped_key}

            elif command_name == 'goto':
                url = groups[0].strip()
                action = create_goto_url_action(url)
                return 'action', action, {'url': url}

            elif command_name == 'go_back':
                action = create_go_back_action()
                return 'action', action, {}

            elif command_name == 'go_forward':
                action = create_go_forward_action()
                return 'action', action, {}

            elif command_name == 'new_tab':
                action = create_new_tab_action()
                return 'action', action, {}

            elif command_name == 'close_tab':
                action = create_page_close_action()
                return 'action', action, {}

            elif command_name == 'stop':
                answer = groups[0] if groups[0] else ""
                action = create_stop_action(answer)
                return 'action', action, {'answer': answer}

            # 控制命令（不是浏览器动作）
            elif command_name in ['reset', 'skip', 'help', 'quit']:
                return 'control', None, {'command': command_name}

            else:
                raise InputParseError(f"未实现的命令: {command_name}")

        except Exception as e:
            raise InputParseError(f"处理命令 '{command_name}' 时发生错误: {str(e)}")

    def get_action_description(self, action: Action) -> str:
        """
        获取动作的描述文本

        Args:
            action: Action对象

        Returns:
            动作描述字符串
        """
        action_type = action.action_type

        try:
            if action_type == ActionTypes.CLICK:
                return f"点击元素 [{action.element_id or '未知'}]"

            elif action_type == ActionTypes.TYPE:
                element_id = action.element_id or '未知'
                # 从action.text中恢复文本（这是一个整数列表）
                text_ids = action.text or []
                if text_ids:
                    # 这里简化处理，实际应该使用browser_env的解码机制
                    text = f"文本内容(长度:{len(text_ids)})"
                else:
                    text = "空文本"
                return f"在元素 [{element_id}] 中输入 {text}"

            elif action_type == ActionTypes.HOVER:
                return f"悬停在元素 [{action.element_id or '未知'}]"

            elif action_type == ActionTypes.SCROLL:
                direction = action.direction or '未知'
                return f"滚动页面 ({direction})"

            elif action_type == ActionTypes.KEY_PRESS:
                key = action.key_comb or '未知'
                return f"按键 {key}"

            elif action_type == ActionTypes.GOTO_URL:
                url = action.url or '未知'
                return f"跳转到 {url}"

            elif action_type == ActionTypes.GO_BACK:
                return "返回上一页"

            elif action_type == ActionTypes.GO_FORWARD:
                return "前进到下一页"

            elif action_type == ActionTypes.NEW_TAB:
                return "打开新标签"

            elif action_type == ActionTypes.PAGE_CLOSE:
                return "关闭标签"

            elif action_type == ActionTypes.STOP:
                answer = action.answer or ''
                return f"停止任务" + (f" (答案: {answer})" if answer else "")

            else:
                return f"未知动作类型 {action_type}"

        except Exception as e:
            return f"动作描述生成失败: {str(e)}"

    def validate_element_id(self, element_id: str) -> bool:
        """
        验证元素ID的有效性

        Args:
            element_id: 元素ID字符串

        Returns:
            是否有效
        """
        try:
            # 检查是否为数字
            id_num = int(element_id)
            # 检查范围（根据browser_env的限制）
            return 0 <= id_num <= 9999  # 假设元素ID在这个范围内
        except ValueError:
            return False

    def get_command_suggestions(self, partial_input: str) -> list[str]:
        """
        根据部分输入获取命令建议

        Args:
            partial_input: 部分输入的命令

        Returns:
            建议的命令列表
        """
        partial = partial_input.lower().strip()
        suggestions = []

        command_examples = {
            'click': ['click [10]', 'click [5]'],
            'type': ['type [5] [hello] [1]', 'type [3] [用户名]'],
            'hover': ['hover [15]', 'hover [8]'],
            'scroll': ['scroll up', 'scroll down'],
            'key_press': ['key_press Enter', 'key_press Tab', 'key_press Escape'],
            'goto': ['goto http://example.com'],
            'go_back': ['go_back'],
            'go_forward': ['go_forward'],
            'new_tab': ['new_tab'],
            'close_tab': ['close_tab'],
            'stop': ['stop', 'stop 任务完成'],
            'reset': ['reset'],
            'skip': ['skip'],
            'help': ['help'],
            'quit': ['quit']
        }

        for command, examples in command_examples.items():
            if command.startswith(partial) or partial in command:
                suggestions.extend(examples)

        return suggestions[:10]  # 最多返回10个建议