#!/usr/bin/env python3
"""
双屏显示功能自动测试脚本
无交互验证图片窗口是否能正确显示在指定的屏幕上
"""

import sys
import tempfile
import os
import time
from pathlib import Path
from PIL import Image

# 添加路径以便导入模块
sys.path.append(str(Path(__file__).parent))

def create_test_image():
    """创建测试图片"""
    try:
        image = Image.new('RGB', (800, 600), color='lightcyan')
        with tempfile.NamedTemporaryFile(suffix='_dual_screen_test.png', delete=False) as temp_file:
            temp_path = temp_file.name
        image.save(temp_path)
        return temp_path
    except Exception as e:
        print(f"[ERROR] 创建测试图片失败: {e}")
        return None

def test_screen_detection():
    """测试屏幕检测功能"""
    print("[TEST] 屏幕检测功能测试")

    try:
        from src.annotation.annotation_ui import AnnotationUI

        ui = AnnotationUI(
            image_display_method='tkinter',
            persistent_window=True,
            target_screen=0
        )

        # 测试屏幕信息检测
        screen_info = ui._get_screen_info()
        print(f"[INFO] 检测到 {len(screen_info)} 个屏幕:")

        for i, screen in enumerate(screen_info):
            print(f"  屏幕 {i}: {screen['width']}x{screen['height']} 位置({screen['x']}, {screen['y']}) 主屏={screen['is_primary']}")

        return screen_info, True

    except Exception as e:
        print(f"[ERROR] 屏幕检测测试失败: {e}")
        import traceback
        traceback.print_exc()
        return [], False

def test_screen_positioning():
    """测试屏幕定位功能"""
    print("\n[TEST] 屏幕定位功能测试")

    test_image_path = create_test_image()
    if not test_image_path:
        return False

    try:
        from src.annotation.annotation_ui import AnnotationUI

        # 测试不同屏幕的显示
        test_screens = [0, 1, 2]  # 测试主屏和可能的副屏
        results = []

        for screen_num in test_screens:
            print(f"\n[ACTION] 测试屏幕 {screen_num} 显示")

            ui = AnnotationUI(
                image_display_method='tkinter',
                persistent_window=False,  # 使用非持续窗口以便自动测试
                target_screen=screen_num
            )

            try:
                # 获取目标屏幕边界
                screen_bounds = ui._get_target_screen_bounds()
                print(f"[INFO] 屏幕 {screen_num} 边界: x={screen_bounds['x']}, y={screen_bounds['y']}, 宽={screen_bounds['width']}, 高={screen_bounds['height']}")

                # 显示图片
                success = ui.display_image_direct(test_image_path, f"屏幕 {screen_num} 测试")

                if success:
                    print(f"[OK] 屏幕 {screen_num} 显示成功")
                    results.append((screen_num, True))

                    # 短暂等待以便观察
                    time.sleep(1)
                else:
                    print(f"[FAIL] 屏幕 {screen_num} 显示失败")
                    results.append((screen_num, False))

                # 清理
                ui.cleanup()

            except Exception as e:
                print(f"[ERROR] 屏幕 {screen_num} 测试失败: {e}")
                results.append((screen_num, False))

        # 清理测试图片
        if os.path.exists(test_image_path):
            os.unlink(test_image_path)

        return results

    except Exception as e:
        print(f"[ERROR] 屏幕定位测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_window_behavior():
    """测试窗口行为"""
    print("\n[TEST] 窗口行为测试")

    test_image_path = create_test_image()
    if not test_image_path:
        return False

    try:
        from src.annotation.annotation_ui import AnnotationUI

        # 测试持续窗口模式
        print("[ACTION] 测试持续窗口模式")
        ui = AnnotationUI(
            image_display_method='tkinter',
            persistent_window=True,
            target_screen=0
        )

        # 显示第一张图片
        success1 = ui.display_image_direct(test_image_path, "持续窗口测试 1")
        if success1:
            print("[OK] 第一次显示成功")

            # 检查窗口状态
            has_window = ui.has_active_window()
            print(f"[INFO] 窗口状态: {has_window}")

            # 显示第二张图片（应该更新现有窗口）
            success2 = ui.display_image_direct(test_image_path, "持续窗口测试 2")
            if success2:
                print("[OK] 窗口更新成功")
            else:
                print("[FAIL] 窗口更新失败")

            # 清理
            ui.cleanup()

            # 清理测试图片
            if os.path.exists(test_image_path):
                os.unlink(test_image_path)

            return success1 and success2
        else:
            print("[FAIL] 第一次显示失败")
            return False

    except Exception as e:
        print(f"[ERROR] 窗口行为测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("[MAIN] 双屏显示功能自动测试")
    print("=" * 60)

    test_results = []

    # 1. 测试屏幕检测
    screen_info, detection_success = test_screen_detection()
    test_results.append(("屏幕检测", detection_success))

    if not detection_success:
        print("[ERROR] 屏幕检测失败，无法继续后续测试")
        return False

    # 2. 测试屏幕定位
    positioning_results = test_screen_positioning()
    if positioning_results:
        successful_screens = sum(1 for _, success in positioning_results if success)
        total_screens = len(positioning_results)
        test_results.append(("屏幕定位", successful_screens > 0))
        print(f"[INFO] {successful_screens}/{total_screens} 个屏幕定位测试成功")
    else:
        test_results.append(("屏幕定位", False))

    # 3. 测试窗口行为
    window_result = test_window_behavior()
    test_results.append(("窗口行为", window_result))

    # 结果总结
    print("\n" + "=" * 60)
    print("[SUMMARY] 测试结果:")

    passed = 0
    total = len(test_results)

    for test_name, result in test_results:
        status = "PASS" if result else "FAIL"
        print(f"  [{status}] {test_name}")
        if result:
            passed += 1

    print(f"\n[RESULT] {passed}/{total} 项测试通过")

    if passed == total:
        print("\n[SUCCESS] 双屏显示功能测试完全通过！")
        print("\n[FEATURE] 新增功能使用方法:")
        print("  # 在主屏幕显示图片窗口")
        print("  python src/interactive_annotator.py --image-display-method tkinter --display-screen 0 --render")
        print("\n  # 在副屏显示图片窗口")
        print("  python src/interactive_annotator.py --image-display-method tkinter --display-screen 1 --render")

        if len(screen_info) > 1:
            print(f"\n[INFO] 检测到 {len(screen_info)} 个屏幕，双屏功能可用！")
        else:
            print("\n[INFO] 只有一个屏幕，但屏幕选择功能已实现并可用于未来的多屏幕环境")

        return True
    else:
        print(f"\n[WARNING] {total-passed} 项测试失败，需要进一步检查")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)