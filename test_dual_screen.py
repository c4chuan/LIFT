#!/usr/bin/env python3
"""
双屏显示功能测试脚本
验证图片窗口是否能正确显示在指定的屏幕上
"""

import sys
import tempfile
import os
import time
from pathlib import Path
from PIL import Image

# 添加路径以便导入模块
sys.path.append(str(Path(__file__).parent))

def create_test_images():
    """创建测试图片"""
    test_images = []

    try:
        # 创建主屏幕测试图片
        image1 = Image.new('RGB', (800, 600), color='lightblue')
        with tempfile.NamedTemporaryFile(suffix='_screen0.png', delete=False) as temp_file:
            temp_path1 = temp_file.name
        image1.save(temp_path1)
        test_images.append(temp_path1)

        # 创建副屏测试图片
        image2 = Image.new('RGB', (800, 600), color='lightgreen')
        with tempfile.NamedTemporaryFile(suffix='_screen1.png', delete=False) as temp_file:
            temp_path2 = temp_file.name
        image2.save(temp_path2)
        test_images.append(temp_path2)

        return test_images

    except Exception as e:
        print(f"[ERROR] 创建测试图片失败: {e}")
        return []

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

        return screen_info

    except Exception as e:
        print(f"[ERROR] 屏幕检测测试失败: {e}")
        import traceback
        traceback.print_exc()
        return []

def test_dual_screen_display(screen_info, test_images):
    """测试双屏显示功能"""
    print("\n[TEST] 双屏显示功能测试")

    if len(screen_info) < 2:
        print("[WARNING] 只检测到一个屏幕，无法进行双屏测试")
        print("[INFO] 将在主屏幕上演示屏幕选择功能")

    try:
        from src.annotation.annotation_ui import AnnotationUI

        # 测试主屏幕显示
        print("\n[ACTION] 测试主屏幕显示 (screen=0)")
        ui_screen0 = AnnotationUI(
            image_display_method='tkinter',
            persistent_window=True,
            target_screen=0
        )

        if test_images:
            success = ui_screen0.display_image_direct(test_images[0], "主屏幕测试 - 蓝色")
            if success:
                print("[OK] 主屏幕显示成功")
                input("[INPUT] 请检查主屏幕上是否显示了蓝色图片，然后按回车继续...")
            else:
                print("[FAIL] 主屏幕显示失败")

        # 测试副屏显示（如果有多个屏幕）
        if len(screen_info) > 1:
            print("\n[ACTION] 测试副屏显示 (screen=1)")
            ui_screen1 = AnnotationUI(
                image_display_method='tkinter',
                persistent_window=True,
                target_screen=1
            )

            if len(test_images) > 1:
                success = ui_screen1.display_image_direct(test_images[1], "副屏测试 - 绿色")
                if success:
                    print("[OK] 副屏显示成功")
                    input("[INPUT] 请检查副屏上是否显示了绿色图片，然后按回车继续...")
                else:
                    print("[FAIL] 副屏显示失败")

        # 清理
        ui_screen0.cleanup()
        if len(screen_info) > 1:
            ui_screen1.cleanup()

        return True

    except Exception as e:
        print(f"[ERROR] 双屏显示测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_command_line_integration():
    """测试命令行集成"""
    print("\n[TEST] 命令行参数集成测试")

    print("[INFO] 命令行使用方法:")
    print("  主屏幕显示:")
    print("    python src/interactive_annotator.py --image-display-method tkinter --display-screen 0")
    print("  副屏显示:")
    print("    python src/interactive_annotator.py --image-display-method tkinter --display-screen 1")
    print("  自动选择:")
    print("    python src/interactive_annotator.py --image-display-method tkinter")

    return True

def main():
    """主函数"""
    print("[MAIN] 双屏显示功能测试")
    print("=" * 60)

    test_results = []

    # 创建测试图片
    print("[SETUP] 创建测试图片...")
    test_images = create_test_images()
    if not test_images:
        print("[ERROR] 无法创建测试图片，测试终止")
        return False

    try:
        # 1. 测试屏幕检测
        screen_info = test_screen_detection()
        if screen_info:
            test_results.append(("屏幕检测", True))
        else:
            test_results.append(("屏幕检测", False))
            print("[ERROR] 屏幕检测失败，后续测试可能不准确")

        # 2. 测试双屏显示
        display_result = test_dual_screen_display(screen_info, test_images)
        test_results.append(("双屏显示", display_result))

        # 3. 测试命令行集成
        cli_result = test_command_line_integration()
        test_results.append(("命令行集成", cli_result))

    finally:
        # 清理测试图片
        for img_path in test_images:
            try:
                if os.path.exists(img_path):
                    os.unlink(img_path)
            except Exception as e:
                print(f"[WARNING] 清理测试图片失败 {img_path}: {e}")

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
        print("\n[SUCCESS] 双屏显示功能测试完成！")
        print("\n[USAGE] 现在可以使用以下命令启用双屏显示:")
        print("  python src/interactive_annotator.py --image-display-method tkinter --display-screen 1 --render")
        return True
    else:
        print("\n[WARNING] 部分测试失败，需要进一步检查")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)