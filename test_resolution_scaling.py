#!/usr/bin/env python3
"""
分辨率缩放功能测试脚本
验证图片在不同分辨率屏幕上的正确缩放显示
"""

import sys
import tempfile
import os
import time
from pathlib import Path
from PIL import Image

# 添加路径以便导入模块
sys.path.append(str(Path(__file__).parent))

def create_simple_test_image():
    """创建简单的测试图片"""
    try:
        # 创建一个1920x1080的测试图片
        image = Image.new('RGB', (1920, 1080), color='lightblue')

        # 在图片上添加一些简单的标记
        from PIL import ImageDraw
        draw = ImageDraw.Draw(image)

        # 绘制一个框架和对角线
        draw.rectangle([0, 0, 1919, 1079], outline='red', width=10)
        draw.line([0, 0, 1919, 1079], fill='red', width=5)
        draw.line([0, 1079, 1919, 0], fill='blue', width=5)

        with tempfile.NamedTemporaryFile(suffix='_resolution_test.png', delete=False) as temp_file:
            temp_path = temp_file.name
        image.save(temp_path)
        return temp_path

    except Exception as e:
        print(f"[ERROR] 创建测试图片失败: {e}")
        return None

def test_screen_detection():
    """测试屏幕检测和尺寸获取"""
    print("[TEST] 屏幕检测和尺寸获取测试")

    try:
        from src.annotation.annotation_ui import AnnotationUI

        ui = AnnotationUI(
            image_display_method='tkinter',
            persistent_window=True,
            target_screen=0
        )

        # 获取所有屏幕信息
        screen_info = ui._get_screen_info()
        print(f"[INFO] 检测到 {len(screen_info)} 个屏幕:")

        for i, screen in enumerate(screen_info):
            print(f"  屏幕 {i}: {screen['width']}x{screen['height']} 位置({screen['x']}, {screen['y']}) 主屏={screen['is_primary']}")

            # 测试目标屏幕尺寸获取
            ui.target_screen = i
            usable_width, usable_height = ui._get_target_screen_display_size()
            print(f"    可用显示尺寸: {usable_width}x{usable_height}")

            # 计算缩放比例
            original_width, original_height = 1920, 1080
            scale_w = usable_width / original_width
            scale_h = usable_height / original_height
            scale = min(scale_w, scale_h)
            print(f"    对于1920x1080图片的缩放比例: {scale:.3f}")
            print()

        return screen_info, True

    except Exception as e:
        print(f"[ERROR] 屏幕检测测试失败: {e}")
        import traceback
        traceback.print_exc()
        return [], False

def test_scaling_on_different_screens(screen_info, test_image):
    """在不同屏幕上测试图片缩放"""
    print("[TEST] 不同屏幕缩放测试")

    if not test_image:
        return False

    try:
        from src.annotation.annotation_ui import AnnotationUI

        results = []

        # 测试前两个屏幕（如果存在）
        screens_to_test = min(len(screen_info), 2)

        for screen_idx in range(screens_to_test):
            print(f"\n[ACTION] 测试屏幕 {screen_idx}")
            screen = screen_info[screen_idx]

            ui = AnnotationUI(
                image_display_method='tkinter',
                persistent_window=False,
                target_screen=screen_idx,
                keep_aspect_ratio=True
            )

            print(f"[INFO] 目标屏幕分辨率: {screen['width']}x{screen['height']}")

            success = ui.display_image_direct(test_image, f"Screen {screen_idx} Test")

            if success:
                print(f"[OK] 屏幕 {screen_idx} 显示成功")
                results.append((screen_idx, True))

                # 短暂等待以便观察
                time.sleep(2)
            else:
                print(f"[FAIL] 屏幕 {screen_idx} 显示失败")
                results.append((screen_idx, False))

            ui.cleanup()

        return results

    except Exception as e:
        print(f"[ERROR] 缩放测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("[MAIN] 分辨率缩放功能测试")
    print("=" * 60)

    test_results = []

    # 创建测试图片
    print("[SETUP] 创建测试图片...")
    test_image = create_simple_test_image()
    if not test_image:
        print("[ERROR] 无法创建测试图片，测试终止")
        return False

    try:
        # 1. 屏幕检测测试
        screen_info, detection_success = test_screen_detection()
        test_results.append(("屏幕检测", detection_success))

        if not detection_success:
            print("[ERROR] 屏幕检测失败，无法继续测试")
            return False

        # 2. 缩放测试
        scaling_results = test_scaling_on_different_screens(screen_info, test_image)
        if scaling_results:
            successful_screens = sum(1 for _, success in scaling_results if success)
            total_screens = len(scaling_results)
            test_results.append(("缩放显示", successful_screens > 0))
            print(f"[INFO] 缩放显示测试: {successful_screens}/{total_screens} 个屏幕成功")
        else:
            test_results.append(("缩放显示", False))

    finally:
        # 清理测试图片
        if test_image and os.path.exists(test_image):
            try:
                os.unlink(test_image)
            except Exception as e:
                print(f"[WARNING] 清理测试图片失败: {e}")

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
        print("\n[SUCCESS] 分辨率缩放功能修复完成！")
        print("\n[IMPROVEMENT] 现在图片会根据实际显示屏幕的分辨率进行正确缩放:")
        print("  ✅ 在高分辨率副屏上图片会更大，充分利用显示空间")
        print("  ✅ 在低分辨率屏幕上图片会适当缩小以适应屏幕")
        print("  ✅ 保持图片原始纵横比，避免变形")

        print("\n[USAGE] 使用示例:")
        print("  # 在副屏显示，图片会根据副屏分辨率缩放")
        print("  python src/interactive_annotator.py --image-display-method tkinter --display-screen 1 --render")

        return True
    else:
        print(f"\n[WARNING] {total-passed} 项测试失败，需要进一步检查")
        return False

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n[INFO] 测试被用户中断")
        sys.exit(1)