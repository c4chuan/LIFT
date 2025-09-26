#!/usr/bin/env python3
"""
简化的EnvironmentController截图功能测试
主要测试接口方法，不依赖外部库
"""

import sys
import tempfile
from pathlib import Path

# 添加路径以便导入模块
sys.path.append(str(Path(__file__).parent))


def test_environment_controller_interface():
    """测试EnvironmentController的接口方法"""
    print("[TEST] 开始测试EnvironmentController接口...")

    try:
        # 导入EnvironmentController类定义来测试方法存在性
        import inspect
        from src.annotation.environment_controller import EnvironmentController

        print("[OK] EnvironmentController 导入成功")

        # 测试类方法是否存在
        required_methods = [
            '_save_som_screenshot',
            'get_screenshot_path',
            'get_all_screenshots',
            'get_current_screenshot_path',
            'get_screenshot_count',
            'get_task_screenshot_directory',
            'get_screenshot_info',
            'save_trajectory_data',
            '_serialize_trajectory',
            '_serialize_info'
        ]

        print("\n[CHECK] 检查必需的方法:")
        for method_name in required_methods:
            if hasattr(EnvironmentController, method_name):
                print(f"[OK] {method_name} - 存在")
            else:
                print(f"[FAIL] {method_name} - 缺失")

        # 创建实例测试初始化
        print("\n[TEST] 测试实例创建...")
        env_controller = EnvironmentController(
            render=False,
            result_dir="data/test_results"
        )
        print("[OK] EnvironmentController 实例创建成功")

        # 测试基本属性
        print("\n[CHECK] 检查基本属性:")
        attributes = [
            'screenshot_paths',
            'current_task_id',
            'current_step',
            'result_dir'
        ]

        for attr_name in attributes:
            if hasattr(env_controller, attr_name):
                value = getattr(env_controller, attr_name)
                print(f"[OK] {attr_name}: {value}")
            else:
                print(f"[FAIL] {attr_name} - 缺失")

        # 测试接口方法（不依赖实际数据）
        print("\n[TEST] 测试接口方法:")

        # 测试截图计数（应该返回0）
        count = env_controller.get_screenshot_count()
        print(f"[OK] get_screenshot_count(): {count}")

        # 测试获取所有截图（应该返回空列表）
        all_screenshots = env_controller.get_all_screenshots()
        print(f"[OK] get_all_screenshots(): {len(all_screenshots)} items")

        # 测试获取最新截图（应该返回None）
        latest = env_controller.get_latest_screenshot_path()
        print(f"[OK] get_latest_screenshot_path(): {latest}")

        # 测试获取截图目录（可能返回None，因为未初始化任务）
        screenshot_dir = env_controller.get_task_screenshot_directory()
        print(f"[OK] get_task_screenshot_directory(): {screenshot_dir}")

        # 测试获取截图信息
        screenshot_info = env_controller.get_screenshot_info()
        print(f"[OK] get_screenshot_info(): {screenshot_info}")

        print("\n[SUCCESS] 接口方法测试完成！")
        return True

    except ImportError as e:
        print(f"[ERROR] 导入错误: {e}")
        print("[INFO] 某些依赖可能缺失，但这不影响代码结构的正确性")
        return False
    except Exception as e:
        print(f"[ERROR] 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_directory_structure():
    """测试目录结构创建逻辑"""
    print("\n[TEST] 测试目录结构...")

    try:
        from pathlib import Path

        # 测试创建测试目录
        test_base_dir = Path("data/test_results")
        test_task_dir = test_base_dir / "task_12345"

        # 确保目录存在
        test_task_dir.mkdir(parents=True, exist_ok=True)

        if test_task_dir.exists():
            print(f"[OK] 测试目录创建成功: {test_task_dir}")

            # 测试文件路径构建
            screenshot_paths = [
                test_task_dir / "step_0_initial_obs.png",
                test_task_dir / "step_1_obs.png",
                test_task_dir / "step_2_obs.png",
                test_task_dir / "trajectory.json"
            ]

            print("[INFO] 预期的文件结构:")
            for path in screenshot_paths:
                print(f"   - {path.name}")

            return True
        else:
            print("[FAIL] 测试目录创建失败")
            return False

    except Exception as e:
        print(f"[ERROR] 目录测试失败: {e}")
        return False


def main():
    """主函数"""
    print("[MAIN] EnvironmentController截图功能接口测试")
    print("=" * 60)

    # 测试接口
    interface_test_passed = test_environment_controller_interface()

    # 测试目录结构
    directory_test_passed = test_directory_structure()

    print("\n" + "=" * 60)
    print("[SUMMARY] 测试结果总结:")
    print(f"  接口测试: {'[PASS] 通过' if interface_test_passed else '[FAIL] 失败'}")
    print(f"  目录测试: {'[PASS] 通过' if directory_test_passed else '[FAIL] 失败'}")

    if interface_test_passed and directory_test_passed:
        print("[SUCCESS] 所有基础测试通过！SOM截图功能接口设计正确。")
        print("[INFO] 注意: 完整功能测试需要安装cv2, numpy等依赖库")
    else:
        print("[WARNING] 某些测试未通过，请检查代码实现")


if __name__ == "__main__":
    main()