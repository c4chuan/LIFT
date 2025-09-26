#!/usr/bin/env python3
"""
单任务测试脚本

用于测试单个任务的playwright codegen启动，验证修复效果。
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.task_manager import TaskManager
from modules.playwright_controller import PlaywrightController


def test_single_task():
    """测试单个任务的playwright启动"""
    print("=== 单任务测试 ===")

    # 加载任务
    task_manager = TaskManager()
    tasks = task_manager.load_tasks()

    if not tasks:
        print("ERROR: 没有找到任务")
        return False

    # 选择第一个任务进行测试
    test_task = tasks[0]
    print(f"测试任务: {test_task.unique_id}")
    print(f"起始URL: {test_task.start_url}")
    print(f"需要登录: {test_task.require_login}")
    print(f"Storage文件: {test_task.storage_state}")

    # 创建playwright控制器
    playwright_controller = PlaywrightController()

    print("\n--- 开始测试playwright codegen启动 ---")

    try:
        # 使用调试模式启动
        success = playwright_controller.start_codegen(test_task, debug=True)

        if success:
            print("OK: Playwright CodeGen启动成功！")
            print("INFO: 等待5秒后自动终止测试...")

            import time
            time.sleep(5)

            # 终止进程
            if playwright_controller.is_process_running():
                print("INFO: 正在终止测试进程...")
                playwright_controller.kill_process()

            print("OK: 测试完成")
            return True
        else:
            print("ERROR: Playwright CodeGen启动失败")
            return False

    except Exception as e:
        print(f"ERROR: 测试时出错: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_single_task()
    sys.exit(0 if success else 1)