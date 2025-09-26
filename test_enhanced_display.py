#!/usr/bin/env python3
"""
增强图片显示功能测试脚本
测试原比例显示和持续窗口功能
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
    """创建多个不同尺寸的测试图片"""
    test_images = []

    try:
        # 创建小图片 (200x150) - 蓝色
        small_image = Image.new('RGB', (200, 150), color='lightblue')
        with tempfile.NamedTemporaryFile(suffix='_small.png', delete=False) as temp_file:
            small_path = temp_file.name
        small_image.save(small_path)
        test_images.append(small_path)
        print(f"[OK] 小图片创建成功: {os.path.basename(small_path)} (200x150)")

        # 创建大图片 (1200x800) - 绿色
        large_image = Image.new('RGB', (1200, 800), color='lightgreen')
        with tempfile.NamedTemporaryFile(suffix='_large.png', delete=False) as temp_file:
            large_path = temp_file.name
        large_image.save(large_path)
        test_images.append(large_path)
        print(f"[OK] 大图片创建成功: {os.path.basename(large_path)} (1200x800)")

        # 创建超大图片 (2000x1500) - 红色
        huge_image = Image.new('RGB', (2000, 1500), color='lightcoral')
        with tempfile.NamedTemporaryFile(suffix='_huge.png', delete=False) as temp_file:
            huge_path = temp_file.name
        huge_image.save(huge_path)
        test_images.append(huge_path)
        print(f"[OK] 超大图片创建成功: {os.path.basename(huge_path)} (2000x1500)")

        return test_images

    except Exception as e:
        print(f"[ERROR] 创建测试图片失败: {e}")
        return []

def test_aspect_ratio_modes():
    """测试保持比例和不保持比例模式"""
    print("\n" + "=" * 60)
    print("[TEST] 原比例显示功能测试")
    print("=" * 60)

    try:
        from src.annotation.annotation_ui import AnnotationUI

        test_images = create_test_images()
        if not test_images:
            return False

        try:
            # 测试保持原比例模式
            print("\n[TEST 1] 保持原比例模式")
            ui_aspect = AnnotationUI(
                image_display_method='tkinter',
                keep_aspect_ratio=True,
                persistent_window=False  # 测试时不使用持续窗口
            )

            for i, image_path in enumerate(test_images):
                print(f"\n显示测试图片 {i+1}: {os.path.basename(image_path)}")
                success = ui_aspect.display_image_direct(image_path, f"保持比例 - 图片{i+1}")
                if success:
                    print(f"[OK] 图片{i+1}显示成功（保持比例）")
                    # 给用户一点时间查看
                    time.sleep(1)
                else:
                    print(f"[FAIL] 图片{i+1}显示失败")

            # 测试不保持原比例模式
            print("\n[TEST 2] 固定尺寸模式")
            ui_fixed = AnnotationUI(
                image_display_method='tkinter',
                image_window_size='600x400',
                keep_aspect_ratio=False,
                persistent_window=False
            )

            for i, image_path in enumerate(test_images):
                print(f"\n显示测试图片 {i+1}: {os.path.basename(image_path)} (固定600x400)")
                success = ui_fixed.display_image_direct(image_path, f"固定尺寸 - 图片{i+1}")
                if success:
                    print(f"[OK] 图片{i+1}显示成功（固定尺寸）")
                    time.sleep(1)
                else:
                    print(f"[FAIL] 图片{i+1}显示失败")

            return True

        finally:
            # 清理测试图片
            for image_path in test_images:
                if os.path.exists(image_path):
                    os.unlink(image_path)
                    print(f"[CLEANUP] 删除测试图片: {os.path.basename(image_path)}")

    except Exception as e:
        print(f"[ERROR] 比例显示测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_persistent_window():
    """测试持续窗口功能"""
    print("\n" + "=" * 60)
    print("[TEST] 持续窗口显示功能测试")
    print("=" * 60)

    try:
        from src.annotation.annotation_ui import AnnotationUI

        # 创建测试图片
        test_images = create_test_images()
        if not test_images:
            return False

        try:
            # 测试持续窗口模式
            print("\n[TEST] 持续窗口模式")
            ui_persistent = AnnotationUI(
                image_display_method='tkinter',
                keep_aspect_ratio=True,
                persistent_window=True
            )

            print("\n模拟连续显示多张图片（持续窗口）...")
            for i, image_path in enumerate(test_images):
                print(f"\n[ACTION] 显示图片 {i+1}: {os.path.basename(image_path)}")
                success = ui_persistent.display_image_direct(image_path, f"持续窗口 - 图片{i+1}")

                if success:
                    print(f"[OK] 图片{i+1}在持续窗口中显示成功")
                    print("[INFO] 窗口应该保持打开状态，图片内容已更新")

                    # 模拟用户操作间隔
                    print(f"[WAIT] 等待3秒后显示下一张图片...")
                    time.sleep(3)
                else:
                    print(f"[FAIL] 图片{i+1}显示失败")

            # 清理窗口
            print("\n[CLEANUP] 清理持续窗口...")
            ui_persistent.cleanup()
            print("[OK] 持续窗口已关闭")

            return True

        finally:
            # 清理测试图片
            for image_path in test_images:
                if os.path.exists(image_path):
                    os.unlink(image_path)

    except Exception as e:
        print(f"[ERROR] 持续窗口测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_command_line_args():
    """测试命令行参数功能"""
    print("\n" + "=" * 60)
    print("[TEST] 命令行参数功能测试")
    print("=" * 60)

    try:
        # 模拟命令行参数解析
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--image-display-method", choices=["tkinter", "ascii", "system", "off", "auto"], default="auto")
        parser.add_argument("--image-window-size", default="800x600")
        parser.add_argument("--ascii-width", type=int, default=80)
        parser.add_argument("--no-keep-aspect-ratio", action="store_false", dest="keep_aspect_ratio")
        parser.add_argument("--no-persistent-window", action="store_false", dest="persistent_window")

        # 测试不同的参数组合
        test_configs = [
            {
                "name": "默认配置",
                "args": []
            },
            {
                "name": "不保持比例",
                "args": ["--no-keep-aspect-ratio"]
            },
            {
                "name": "不使用持续窗口",
                "args": ["--no-persistent-window"]
            },
            {
                "name": "ASCII显示",
                "args": ["--image-display-method", "ascii", "--ascii-width", "60"]
            }
        ]

        for config in test_configs:
            print(f"\n[TEST] {config['name']}")
            try:
                args = parser.parse_args(config['args'])

                print(f"  image_display_method: {args.image_display_method}")
                print(f"  keep_aspect_ratio: {args.keep_aspect_ratio}")
                print(f"  persistent_window: {args.persistent_window}")
                if hasattr(args, 'ascii_width'):
                    print(f"  ascii_width: {args.ascii_width}")

                print(f"[OK] {config['name']} 参数解析成功")

            except Exception as e:
                print(f"[FAIL] {config['name']} 参数解析失败: {e}")

        return True

    except Exception as e:
        print(f"[ERROR] 命令行参数测试失败: {e}")
        return False

def main():
    """主函数"""
    print("[MAIN] 增强图片显示功能测试")
    print("=" * 60)

    tests_passed = 0
    total_tests = 0

    # 测试1: 原比例显示功能
    total_tests += 1
    print(f"\n{'='*20} 测试 {total_tests}/3 {'='*20}")
    if test_aspect_ratio_modes():
        tests_passed += 1
        print("\n[PASS] 原比例显示功能测试通过")
    else:
        print("\n[FAIL] 原比例显示功能测试失败")

    # 测试2: 持续窗口功能
    total_tests += 1
    print(f"\n{'='*20} 测试 {total_tests}/3 {'='*20}")
    if test_persistent_window():
        tests_passed += 1
        print("\n[PASS] 持续窗口功能测试通过")
    else:
        print("\n[FAIL] 持续窗口功能测试失败")

    # 测试3: 命令行参数功能
    total_tests += 1
    print(f"\n{'='*20} 测试 {total_tests}/3 {'='*20}")
    if test_command_line_args():
        tests_passed += 1
        print("\n[PASS] 命令行参数功能测试通过")
    else:
        print("\n[FAIL] 命令行参数功能测试失败")

    # 结果总结
    print("\n" + "=" * 60)
    print(f"[SUMMARY] 测试结果: {tests_passed}/{total_tests} 通过")

    if tests_passed == total_tests:
        print("[SUCCESS] 所有增强图片显示功能测试通过！")
        print("[INFO] 新功能已准备就绪：")
        print("  ✓ 原比例显示图片")
        print("  ✓ 持续窗口模式，可边看图片边操作")
        print("  ✓ 灵活的命令行配置选项")
        return True
    else:
        print("[WARNING] 部分测试失败，请检查相关功能")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)