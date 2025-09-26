#!/usr/bin/env python3
"""
基本功能测试脚本
"""

from modules.task_manager import TaskManager
from modules.progress_tracker import ProgressTracker
from modules.file_manager import FileManager
from modules.ui_presenter import UIPresenter

def test_task_loading():
    """测试任务加载"""
    print("=== 测试任务加载 ===")
    tm = TaskManager()
    tasks = tm.load_tasks()
    print(f"成功加载了 {len(tasks)} 个任务")

    if tasks:
        print("\n前3个任务:")
        for i, task in enumerate(tasks[:3]):
            print(f"{i+1}. {task.unique_id}: {task.intent[:50]}...")

    return len(tasks) > 0

def test_progress_tracker():
    """测试进度跟踪"""
    print("\n=== 测试进度跟踪 ===")
    pt = ProgressTracker()
    progress = pt.load_progress()
    print(f"加载进度数据: {progress}")

    # 测试保存进度
    success = pt.save_progress("test_task")
    print(f"保存进度测试: {'成功' if success else '失败'}")

    return True

def test_file_manager():
    """测试文件管理"""
    print("\n=== 测试文件管理 ===")
    fm = FileManager()

    # 测试目录创建
    test_dir = "data/annotation_progress/test"
    success = fm.ensure_directory_exists(test_dir)
    print(f"创建目录测试: {'成功' if success else '失败'}")

    return success

def test_ui_presenter():
    """测试UI展示"""
    print("\n=== 测试UI展示 ===")
    ui = UIPresenter()
    ui.show_welcome_message()
    print("UI展示测试完成")

    return True

def main():
    """主测试函数"""
    print("开始基本功能测试...\n")

    tests = [
        ("任务加载", test_task_loading),
        ("进度跟踪", test_progress_tracker),
        ("文件管理", test_file_manager),
        ("UI展示", test_ui_presenter)
    ]

    results = {}
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"测试 {name} 失败: {e}")
            results[name] = False

    print("\n=== 测试结果 ===")
    for name, result in results.items():
        status = "通过" if result else "失败"
        print(f"{name}: {status}")

    all_passed = all(results.values())
    print(f"\n总体结果: {'所有测试通过' if all_passed else '部分测试失败'}")

if __name__ == "__main__":
    main()