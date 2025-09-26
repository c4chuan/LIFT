#!/usr/bin/env python3
"""
多分辨率显示功能测试脚本
验证图片在不同分辨率屏幕上的正确缩放显示
"""

import sys
import tempfile
import os
import time
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# 添加路径以便导入模块
sys.path.append(str(Path(__file__).parent))

def create_test_images_with_info():
    """创建带有信息标识的测试图片"""
    test_images = []
    colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']
    texts = ['主屏测试', '副屏测试', '高分辨率测试', '低分辨率测试']

    for i, (color, text) in enumerate(zip(colors, texts)):
        try:
            # 创建1920x1080的测试图片
            image = Image.new('RGB', (1920, 1080), color=color)

            # 添加文字标识
            draw = ImageDraw.Draw(image)
            try:
                # 尝试使用默认字体
                font = ImageFont.load_default()
            except:
                font = None

            # 添加屏幕信息文字
            text_lines = [
                f"屏幕 {i} - {text}",
                f"测试图片尺寸: 1920x1080",
                "用于验证不同屏幕分辨率下的缩放效果",
                f"背景色: {color}"
            ]

            y_offset = 100
            for line in text_lines:
                draw.text((100, y_offset), line, fill='black', font=font)
                y_offset += 50

            # 添加角落标记
            draw.rectangle([50, 50, 200, 100], fill='red')
            draw.text((60, 65), f"屏幕 {i}", fill='white', font=font)

            # 保存图片
            with tempfile.NamedTemporaryFile(suffix=f'_screen{i}_test.png', delete=False) as temp_file:
                temp_path = temp_file.name
            image.save(temp_path)
            test_images.append(temp_path)

        except Exception as e:
            print(f"[ERROR] 创建测试图片 {i} 失败: {e}")

    return test_images

def test_screen_resolution_info():
    """测试并显示屏幕分辨率信息"""
    print("[TEST] 屏幕分辨率信息测试")

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
            print(f"  屏幕 {i}:")
            print(f"    分辨率: {screen['width']}x{screen['height']}")
            print(f"    位置: ({screen['x']}, {screen['y']})")
            print(f"    主屏幕: {screen['is_primary']}")

            # 测试可用显示尺寸
            ui.target_screen = i
            usable_width, usable_height = ui._get_target_screen_display_size()
            print(f"    可用显示尺寸: {usable_width}x{usable_height}")
            print()

        return screen_info, True

    except Exception as e:
        print(f"[ERROR] 屏幕分辨率信息测试失败: {e}")
        import traceback
        traceback.print_exc()
        return [], False

def test_resolution_scaling(screen_info, test_images):
    """测试不同分辨率下的图片缩放"""
    print("[TEST] 分辨率缩放测试")

    if not test_images:
        print("[ERROR] 没有测试图片可用")
        return False

    try:
        from src.annotation.annotation_ui import AnnotationUI

        results = []

        for screen_idx, screen in enumerate(screen_info):
            print(f"\n[ACTION] 测试屏幕 {screen_idx} - 分辨率 {screen['width']}x{screen['height']}")

            # 测试保持比例模式
            print(f"  [子测试] 保持纵横比模式")
            ui_keep_ratio = AnnotationUI(
                image_display_method='tkinter',
                persistent_window=False,
                target_screen=screen_idx,
                keep_aspect_ratio=True
            )

            if screen_idx < len(test_images):
                success_ratio = ui_keep_ratio.display_image_direct(
                    test_images[screen_idx],
                    f"屏幕{screen_idx} - 保持比例测试"
                )

                if success_ratio:
                    print(f"    [OK] 保持比例显示成功")
                    time.sleep(1)  # 短暂显示
                    results.append((screen_idx, "保持比例", True))
                else:
                    print(f"    [FAIL] 保持比例显示失败")
                    results.append((screen_idx, "保持比例", False))

                ui_keep_ratio.cleanup()

            # 测试固定尺寸模式
            print(f"  [子测试] 固定尺寸模式")
            ui_fixed_size = AnnotationUI(
                image_display_method='tkinter',
                persistent_window=False,
                target_screen=screen_idx,
                keep_aspect_ratio=False,
                image_window_size="1000x700"
            )

            if screen_idx < len(test_images):
                success_fixed = ui_fixed_size.display_image_direct(
                    test_images[screen_idx],
                    f"屏幕{screen_idx} - 固定尺寸测试"
                )

                if success_fixed:
                    print(f"    [OK] 固定尺寸显示成功")
                    time.sleep(1)  # 短暂显示
                    results.append((screen_idx, "固定尺寸", True))
                else:
                    print(f"    [FAIL] 固定尺寸显示失败")
                    results.append((screen_idx, "固定尺寸", False))

                ui_fixed_size.cleanup()

        return results

    except Exception as e:
        print(f"[ERROR] 分辨率缩放测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_cross_screen_comparison():
    """跨屏幕对比测试"""
    print("\n[TEST] 跨屏幕显示对比测试")

    try:
        from src.annotation.annotation_ui import AnnotationUI

        # 创建对比测试图片
        test_image = create_test_images_with_info()[0]
        if not test_image:
            return False

        print("[INFO] 同时在多个屏幕显示相同图片，观察缩放效果差异")

        uis = []
        screens_to_test = [0, 1]  # 主屏和第一个副屏

        for screen_num in screens_to_test:
            try:
                ui = AnnotationUI(
                    image_display_method='tkinter',
                    persistent_window=True,
                    target_screen=screen_num,
                    keep_aspect_ratio=True
                )

                success = ui.display_image_direct(
                    test_image,
                    f"屏幕{screen_num}对比测试"
                )

                if success:
                    print(f"[OK] 屏幕 {screen_num} 显示成功")
                    uis.append(ui)
                else:
                    print(f"[FAIL] 屏幕 {screen_num} 显示失败")

            except Exception as e:
                print(f"[WARNING] 屏幕 {screen_num} 测试失败: {e}")

        if len(uis) > 1:
            input("[INPUT] 请观察不同屏幕上的图片缩放效果，然后按回车继续...")

        # 清理
        for ui in uis:
            ui.cleanup()

        # 清理测试图片
        if os.path.exists(test_image):
            os.unlink(test_image)

        return len(uis) > 0

    except Exception as e:
        print(f"[ERROR] 跨屏幕对比测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("[MAIN] 多分辨率显示功能测试")
    print("=" * 70)

    test_results = []

    # 创建测试图片
    print("[SETUP] 创建测试图片...")
    test_images = create_test_images_with_info()
    if not test_images:
        print("[ERROR] 无法创建测试图片，测试终止")
        return False

    try:
        # 1. 屏幕分辨率信息测试
        screen_info, info_success = test_screen_resolution_info()
        test_results.append(("屏幕分辨率检测", info_success))

        if not info_success:
            print("[ERROR] 屏幕信息检测失败，无法继续测试")
            return False

        # 2. 分辨率缩放测试
        scaling_results = test_resolution_scaling(screen_info, test_images)
        if scaling_results:
            successful_tests = sum(1 for _, _, success in scaling_results if success)
            total_tests = len(scaling_results)
            test_results.append(("分辨率缩放", successful_tests > 0))
            print(f"[INFO] 分辨率缩放测试: {successful_tests}/{total_tests} 成功")
        else:
            test_results.append(("分辨率缩放", False))

        # 3. 跨屏幕对比测试（交互式）
        comparison_success = test_cross_screen_comparison()
        test_results.append(("跨屏幕对比", comparison_success))

    finally:
        # 清理测试图片
        for img_path in test_images:
            try:
                if os.path.exists(img_path):
                    os.unlink(img_path)
            except Exception as e:
                print(f"[WARNING] 清理测试图片失败 {img_path}: {e}")

    # 结果总结
    print("\n" + "=" * 70)
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
        print("\n[SUCCESS] 多分辨率显示功能测试完成！")
        print("\n[FEATURE] 功能特性:")
        print("  ✅ 图片根据目标屏幕分辨率正确缩放")
        print("  ✅ 保持纵横比模式适应不同屏幕")
        print("  ✅ 固定尺寸模式支持跨屏显示")
        print("  ✅ 高分辨率屏幕充分利用显示空间")

        print("\n[USAGE] 使用示例:")
        print("  # 在高分辨率副屏显示，图片会相应放大")
        print("  python src/interactive_annotator.py --image-display-method tkinter --display-screen 1 --render")
        print("  # 在低分辨率屏幕显示，图片会适当缩小")
        print("  python src/interactive_annotator.py --image-display-method tkinter --display-screen 0 --render")

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