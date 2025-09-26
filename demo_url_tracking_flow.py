#!/usr/bin/env python3
"""
演示URL实时跟踪在完整交互流程中的效果
"""

import sys
from pathlib import Path

# 添加路径以便导入模块
sys.path.append(str(Path(__file__).parent))

from src.annotation.annotation_ui import AnnotationUI


def demo_url_tracking_flow():
    """演示URL实时跟踪的完整交互流程"""
    print("=" * 80)
    print("  演示: URL实时跟踪在交互式标注中的应用")
    print("=" * 80)

    # 创建UI实例
    ui = AnnotationUI(image_display_method='off')

    print("\n[场景] 用户标注任务: 在购物网站查找商品价格")
    print("-" * 60)

    # 1. 任务开始
    print("\n[步骤1] 任务初始化 - 访问购物网站首页")
    print("=" * 60)
    ui.set_task_context(
        env_name="shopping",
        task_id="S001",
        task_intent="Find price of wireless headphones on e-commerce site",
        start_url="https://shop.example.com/",
        current_url="https://shop.example.com/",
        step_count=0
    )

    ui.display_screenshot_ready("screenshot_homepage.png")
    ui.display_current_task_context()
    print("[INPUT] 请输入操作命令:")
    print(">>> click [search_button]")
    print()

    # 2. 第一次页面跳转 - 搜索页面
    print("\n[步骤2] 点击搜索按钮 - 跳转到搜索页面")
    print("=" * 60)
    # 模拟环境控制器返回的new_state，包含新URL
    new_state = {"url": "https://shop.example.com/search"}

    # 更新任务上下文（模拟InteractiveAnnotator的逻辑）
    ui.set_task_context(
        step_count=1,
        current_url=new_state.get('url', '')
    )

    ui.display_action_result(True, "点击搜索按钮")
    ui.display_screenshot_ready("screenshot_search_page.png")
    ui.display_current_task_context()
    print("[INPUT] 请输入操作命令:")
    print(">>> type [search_input] [wireless headphones] [1]")
    print()

    # 3. 输入搜索关键词（URL不变）
    print("\n[步骤3] 输入搜索关键词")
    print("=" * 60)
    ui.set_task_context(step_count=2)

    ui.display_action_result(True, "输入搜索关键词: wireless headphones")
    ui.display_current_task_context()
    print("[INPUT] 请输入操作命令:")
    print(">>> key_press Enter")
    print()

    # 4. 执行搜索 - 搜索结果页面（带参数的URL）
    print("\n[步骤4] 执行搜索 - 跳转到搜索结果页面")
    print("=" * 60)
    new_state = {"url": "https://shop.example.com/search?query=wireless+headphones&category=electronics"}

    ui.set_task_context(
        step_count=3,
        current_url=new_state.get('url', '')
    )

    ui.display_action_result(True, "执行搜索操作")
    ui.display_screenshot_ready("screenshot_search_results.png")
    ui.display_current_task_context()
    print("[INPUT] 请输入操作命令:")
    print(">>> click [product_1]  # 点击第一个商品")
    print()

    # 5. 点击商品 - 商品详情页面
    print("\n[步骤5] 点击商品 - 跳转到商品详情页面")
    print("=" * 60)
    new_state = {"url": "https://shop.example.com/products/wireless-headphones-xyz123"}

    ui.set_task_context(
        step_count=4,
        current_url=new_state.get('url', '')
    )

    ui.display_action_result(True, "点击商品链接")
    ui.display_screenshot_ready("screenshot_product_details.png")
    ui.display_current_task_context()
    print("[INPUT] 请输入操作命令:")
    print(">>> scroll down")
    print()

    # 6. 滚动页面（URL不变）
    print("\n[步骤6] 滚动查看更多商品信息")
    print("=" * 60)
    ui.set_task_context(step_count=5)

    ui.display_action_result(True, "向下滚动页面")
    ui.display_screenshot_ready("screenshot_product_details_scrolled.png")
    ui.display_current_task_context()
    print("[INPUT] 请输入操作命令:")
    print(">>> click [reviews_tab]")
    print()

    # 7. 切换到评论标签（URL包含锚点）
    print("\n[步骤7] 切换到评论标签")
    print("=" * 60)
    new_state = {"url": "https://shop.example.com/products/wireless-headphones-xyz123#reviews"}

    ui.set_task_context(
        step_count=6,
        current_url=new_state.get('url', '')
    )

    ui.display_action_result(True, "切换到评论标签")
    ui.display_screenshot_ready("screenshot_product_reviews.png")
    ui.display_current_task_context()
    print("[INPUT] 请输入操作命令:")
    print(">>> goto https://competitor.com/similar-product")
    print()

    # 8. 跳转到竞品网站（跨域名）
    print("\n[步骤8] 跳转到竞品网站进行价格比较")
    print("=" * 60)
    new_state = {"url": "https://competitor.com/similar-product"}

    ui.set_task_context(
        step_count=7,
        current_url=new_state.get('url', '')
    )

    ui.display_action_result(True, "跳转到竞品网站")
    ui.display_screenshot_ready("screenshot_competitor_site.png")
    ui.display_current_task_context()
    print("[INPUT] 请输入操作命令:")
    print(">>> go_back")
    print()

    # 9. 返回原商品页面
    print("\n[步骤9] 返回原商品页面")
    print("=" * 60)
    new_state = {"url": "https://shop.example.com/products/wireless-headphones-xyz123#reviews"}

    ui.set_task_context(
        step_count=8,
        current_url=new_state.get('url', '')
    )

    ui.display_action_result(True, "返回原商品页面")
    ui.display_screenshot_ready("screenshot_back_to_product.png")
    ui.display_current_task_context()
    print("[INPUT] 请输入操作命令:")
    print(">>> stop [$89.99]  # 提交找到的价格")
    print()

    # 10. 任务完成
    print("\n[步骤10] 任务完成")
    print("=" * 60)
    ui.set_task_context(step_count=9)

    ui.display_action_result(True, "提交价格信息: $89.99")
    print("\n[完成] 任务成功完成!")
    print("最终状态:", ui.get_task_context_summary())

    print("\n" + "=" * 80)
    print("[URL跟踪效果总结]")
    print("=" * 80)
    print("在整个交互流程中，URL显示正确跟踪了以下变化:")
    print("1. 起始页面: shop.example.com")
    print("2. 搜索页面: shop.example.com (从起始页面跳转)")
    print("3. 搜索结果: shop.example.com (包含搜索参数)")
    print("4. 商品详情: shop.example.com (具体商品页面)")
    print("5. 评论标签: shop.example.com (包含锚点)")
    print("6. 竞品网站: competitor.com (跨域名跳转)")
    print("7. 返回原站: shop.example.com (返回原页面)")
    print("\n每次页面跳转都能准确显示当前所在位置，帮助用户保持导航清晰！")


if __name__ == "__main__":
    demo_url_tracking_flow()