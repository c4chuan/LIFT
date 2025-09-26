#!/usr/bin/env python3
"""
演示交互式标注程序中任务信息的完整显示流程
"""

import sys
from pathlib import Path

# 添加路径以便导入模块
sys.path.append(str(Path(__file__).parent))

from src.annotation.annotation_ui import AnnotationUI


def demo_interactive_flow():
    """演示完整的交互流程中任务信息显示效果"""
    print("=" * 80)
    print("  演示: 交互式标注程序中的任务信息持续显示")
    print("=" * 80)

    # 创建UI实例
    ui = AnnotationUI(image_display_method='off')  # 关闭图片显示避免干扰

    print("\n[场景] 用户开始Reddit任务 - 查找蓝色皮划艇帖子")
    print("-" * 60)

    # 1. 任务开始 - 设置初始上下文
    print("\n[任务初始化]")
    ui.set_task_context(
        env_name="reddit",
        task_id="42",
        task_intent="Find a blue kayak post and get the price information",
        start_url="https://reddit.com/",
        current_url="https://reddit.com/",
        step_count=0
    )

    # 模拟显示任务信息
    print("[任务ID] 42")
    print("[任务描述] Find a blue kayak post and get the price information")
    print("[起始URL] https://reddit.com/")
    print()

    # 2. 第一个用户输入提示
    print("=" * 60)
    print("第1步: 初始页面加载完毕，等待用户操作")
    print("=" * 60)

    ui.display_screenshot_ready("screenshots/step_0_homepage.png")
    ui.display_current_task_context()
    print("[INPUT] 请输入操作命令 (输入 'help' 查看帮助):")
    print(">>> click [search_box]")
    print()

    # 3. 模拟执行搜索框点击
    ui.set_task_context(step_count=1)
    ui.display_action_result(True, "点击搜索框")

    # 4. 第二个用户输入提示
    print("=" * 60)
    print("第2步: 搜索框已激活，等待输入搜索关键词")
    print("=" * 60)

    ui.display_screenshot_ready("screenshots/step_1_search_box_active.png")
    ui.display_current_task_context()
    print("[INPUT] 请输入操作命令 (输入 'help' 查看帮助):")
    print(">>> type [search_box] [blue kayak] [1]")
    print()

    # 5. 模拟输入搜索关键词
    ui.set_task_context(step_count=2)
    ui.display_action_result(True, "在搜索框中输入 'blue kayak'")

    # 6. 第三个用户输入提示
    print("=" * 60)
    print("第3步: 已输入搜索关键词，准备执行搜索")
    print("=" * 60)

    ui.display_screenshot_ready("screenshots/step_2_search_text_entered.png")
    ui.display_current_task_context()
    print("[INPUT] 请输入操作命令 (输入 'help' 查看帮助):")
    print(">>> key_press Enter")
    print()

    # 7. 模拟按Enter搜索
    ui.set_task_context(
        step_count=3,
        current_url="https://reddit.com/search?q=blue+kayak"
    )
    ui.display_action_result(True, "按下 Enter 键执行搜索")

    # 8. 搜索结果页面
    print("=" * 60)
    print("第4步: 搜索完成，显示结果页面")
    print("=" * 60)

    ui.display_screenshot_ready("screenshots/step_3_search_results.png")
    ui.display_current_task_context()
    print("[INPUT] 请输入操作命令 (输入 'help' 查看帮助):")
    print(">>> click [15]  # 点击第一个蓝色皮划艇帖子")
    print()

    # 9. 模拟点击帖子
    ui.set_task_context(
        step_count=4,
        current_url="https://reddit.com/r/kayaking/comments/abc123/blue_kayak_for_sale"
    )
    ui.display_action_result(True, "点击蓝色皮划艇帖子链接")

    # 10. 帖子详情页面
    print("=" * 60)
    print("第5步: 进入帖子详情页面，查看价格信息")
    print("=" * 60)

    ui.display_screenshot_ready("screenshots/step_4_post_details.png")
    ui.display_current_task_context()
    print("[INPUT] 请输入操作命令 (输入 'help' 查看帮助):")
    print(">>> scroll down  # 滚动查看完整帖子内容")
    print()

    # 11. 模拟滚动操作
    ui.set_task_context(step_count=5)
    ui.display_action_result(True, "向下滚动页面")

    # 12. 找到价格信息，准备完成任务
    print("=" * 60)
    print("第6步: 找到价格信息，准备完成任务")
    print("=" * 60)

    ui.display_screenshot_ready("screenshots/step_5_price_visible.png")
    ui.display_current_task_context()
    print("[INPUT] 请输入操作命令 (输入 'help' 查看帮助):")
    print(">>> stop [$350]  # 提交找到的价格信息")
    print()

    # 13. 任务完成
    ui.set_task_context(step_count=6)
    ui.display_action_result(True, "提交任务答案: $350")

    print("=" * 60)
    print("[完成] 任务完成! 用户成功找到了蓝色皮划艇的价格信息")
    print("=" * 60)
    print("最终任务上下文:", ui.get_task_context_summary())
    print()

    print("[总结] 演示完成! 可以看到在整个交互过程中:")
    print("   1. 每次用户输入前都显示当前任务信息")
    print("   2. 动作执行后显示更新的进度信息")
    print("   3. 截图显示时包含任务状态提醒")
    print("   4. URL变化被正确跟踪和显示")
    print("   5. 步骤计数实时更新")


if __name__ == "__main__":
    demo_interactive_flow()