#!/usr/bin/env python3
"""
测试交互式标注程序的任务信息显示功能
"""

import sys
from pathlib import Path

# 添加路径以便导入模块
sys.path.append(str(Path(__file__).parent))

from src.annotation.annotation_ui import AnnotationUI


def test_task_context_display():
    """测试任务上下文显示功能"""
    print("=" * 60)
    print("测试交互式标注程序的任务信息显示功能")
    print("=" * 60)

    # 创建UI实例
    ui = AnnotationUI(image_display_method='off')  # 关闭图片显示避免干扰

    # 测试1: 初始状态（无任务）
    print("\n[测试1] 初始状态 - 无任务信息:")
    print("get_task_context_summary():", repr(ui.get_task_context_summary()))
    ui.display_current_task_context()

    # 测试2: 设置基本任务信息
    print("\n[测试2] 设置基本任务信息:")
    ui.set_task_context(
        env_name="reddit",
        task_id="123",
        task_intent="Find blue kayak post on subreddit",
        start_url="https://reddit.com/",
        current_url="https://reddit.com/",
        step_count=0
    )
    print("get_task_context_summary():", repr(ui.get_task_context_summary()))
    ui.display_current_task_context()

    # 测试3: 更新步骤计数
    print("\n[测试3] 更新步骤计数:")
    ui.set_task_context(step_count=3)
    print("get_task_context_summary():", repr(ui.get_task_context_summary()))
    ui.display_current_task_context()

    # 测试4: 更新当前URL
    print("\n[测试4] 更新当前URL:")
    ui.set_task_context(current_url="https://reddit.com/search?q=blue+kayak")
    print("get_task_context_summary():", repr(ui.get_task_context_summary()))
    ui.display_current_task_context()

    # 测试5: 长任务描述截断
    print("\n[测试5] 长任务描述截断测试:")
    ui.set_task_context(
        task_intent="This is a very long task description that should be truncated when displayed to avoid cluttering the interface with too much text"
    )
    print("get_task_context_summary():", repr(ui.get_task_context_summary()))
    ui.display_current_task_context()

    # 测试6: 模拟输入提示显示
    print("\n[测试6] 模拟输入提示显示:")
    print("模拟调用 prompt_user_input() 时的显示效果:")
    ui.display_current_task_context()
    print("[INPUT] 请输入操作命令 (输入 'help' 查看帮助):")
    print(">>> [这里是用户输入位置]")

    # 测试7: 模拟动作执行结果显示
    print("\n[测试7] 模拟动作执行结果显示:")
    ui.display_action_result(True, "点击搜索按钮")

    # 测试8: 模拟截图就绪显示
    print("\n[测试8] 模拟截图就绪显示:")
    ui.display_screenshot_ready("fake_screenshot_path.png")

    # 测试9: 清除任务上下文
    print("\n[测试9] 清除任务上下文:")
    ui.clear_task_context()
    print("get_task_context_summary():", repr(ui.get_task_context_summary()))
    ui.display_current_task_context()

    print("\n" + "=" * 60)
    print("所有测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    test_task_context_display()