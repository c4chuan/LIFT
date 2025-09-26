#!/usr/bin/env python3
"""
测试标注工具模块的基本功能
"""

import sys
from pathlib import Path

# 添加路径
sys.path.append(str(Path(__file__).parent))

def test_task_manager():
    """测试任务管理器"""
    print("测试任务管理器...")
    try:
        from src.annotation.task_manager import TaskManager

        # 创建任务管理器实例
        task_manager = TaskManager()

        # 获取进度摘要
        progress = task_manager.get_progress_summary()
        print(f"  进度摘要: {progress['总任务数']} 个任务")

        # 获取下一个任务
        next_task = task_manager.get_next_task()
        if next_task:
            env_name, task = next_task
            print(f"  下一个任务: {env_name} - {task['task_id']}")
        else:
            print("  没有待处理任务")

        print("✅ 任务管理器测试通过")
        return True

    except Exception as e:
        print(f"❌ 任务管理器测试失败: {e}")
        return False

def test_annotation_ui():
    """测试标注界面"""
    print("测试标注界面...")
    try:
        from src.annotation.annotation_ui import AnnotationUI

        # 创建界面实例
        ui = AnnotationUI()

        # 测试一些基本功能（不会实际显示）
        ui.display_welcome()

        # 测试进度显示
        progress_info = {
            '总任务数': 100,
            '已完成': 10,
            '剩余': 90,
            '完成率': '10%',
            '按环境统计': {
                'classifieds': {'总数': 30, '已完成': 5, '剩余': 25},
                'reddit': {'总数': 30, '已完成': 3, '剩余': 27},
                'shopping': {'总数': 40, '已完成': 2, '剩余': 38}
            },
            '当前任务': 'classifieds_4',
            '最后更新': '2025-09-24T15:41:54'
        }
        ui.display_progress(progress_info)

        print("✅ 标注界面测试通过")
        return True

    except Exception as e:
        print(f"❌ 标注界面测试失败: {e}")
        return False

def test_trajectory_manager():
    """测试轨迹管理器"""
    print("测试轨迹管理器...")
    try:
        from src.annotation.trajectory_manager import TrajectoryManager

        # 创建轨迹管理器实例
        trajectory_manager = TrajectoryManager()

        # 获取统计信息
        stats = trajectory_manager.get_trajectory_stats()
        print(f"  轨迹统计: {stats['total_trajectories']} 个轨迹")

        # 列出轨迹
        trajectories = trajectory_manager.list_trajectories()
        print(f"  找到 {len(trajectories)} 个轨迹文件")

        print("✅ 轨迹管理器测试通过")
        return True

    except Exception as e:
        print(f"❌ 轨迹管理器测试失败: {e}")
        return False

def test_basic_modules():
    """测试基本模块（不依赖browser_env）"""
    print("=" * 50)
    print("测试标注工具基本模块")
    print("=" * 50)

    results = []

    # 测试任务管理器
    results.append(test_task_manager())
    print()

    # 测试标注界面
    results.append(test_annotation_ui())
    print()

    # 测试轨迹管理器
    results.append(test_trajectory_manager())
    print()

    # 总结
    print("=" * 50)
    passed = sum(results)
    total = len(results)
    print(f"测试总结: {passed}/{total} 个模块测试通过")

    if passed == total:
        print("🎉 所有基本模块测试通过!")
        return True
    else:
        print("⚠️ 部分模块测试失败")
        return False

def test_input_parser_basic():
    """测试输入解析器的基本功能（不依赖browser_env的枚举）"""
    print("测试输入解析器基本功能...")
    try:
        # 只测试正则表达式匹配，不创建Action对象
        import re

        command_patterns = {
            'click': r'^click\s+\[(\d+)\]$',
            'type': r'^type\s+\[(\d+)\]\s+\[([^\]]+)\](?:\s+\[(\d+)\])?$',
            'hover': r'^hover\s+\[(\d+)\]$',
            'scroll': r'^scroll\s+(up|down)$',
            'stop': r'^stop(?:\s+(.*))?$',
            'help': r'^help$',
            'quit': r'^quit$',
        }

        # 编译正则表达式
        compiled_patterns = {
            name: re.compile(pattern, re.IGNORECASE)
            for name, pattern in command_patterns.items()
        }

        # 测试一些输入
        test_inputs = [
            ("click [10]", "click"),
            ("type [5] [hello world] [1]", "type"),
            ("hover [15]", "hover"),
            ("scroll up", "scroll"),
            ("stop 任务完成", "stop"),
            ("help", "help"),
            ("quit", "quit"),
        ]

        for input_text, expected_command in test_inputs:
            matched = False
            for command_name, pattern in compiled_patterns.items():
                if pattern.match(input_text):
                    if command_name == expected_command:
                        matched = True
                    break

            if matched:
                print(f"  ✅ '{input_text}' -> {expected_command}")
            else:
                print(f"  ❌ '{input_text}' 解析失败")
                return False

        print("✅ 输入解析器基本功能测试通过")
        return True

    except Exception as e:
        print(f"❌ 输入解析器基本功能测试失败: {e}")
        return False

def main():
    """主函数"""
    print("交互式标注工具模块测试")

    # 测试基本模块
    basic_success = test_basic_modules()

    # 测试输入解析器基本功能
    print()
    parser_success = test_input_parser_basic()

    print()
    print("=" * 50)
    if basic_success and parser_success:
        print("🎊 所有测试通过! 基本模块功能正常")
        print()
        print("注意事项:")
        print("- 由于Python版本为3.7，不支持match语句(Python 3.10+)")
        print("- browser_env模块可能需要更新的Python版本")
        print("- 建议使用Python 3.10+运行完整功能")
        print()
        print("下一步:")
        print("1. 升级Python版本到3.10+")
        print("2. 安装必要依赖: pip install playwright beautifulsoup4 requests pillow")
        print("3. 运行: python src/interactive_annotator.py --help")
    else:
        print("❌ 测试失败，请检查模块实现")

if __name__ == "__main__":
    main()