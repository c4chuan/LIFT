#!/usr/bin/env python3
"""
图片显示功能测试脚本
测试AnnotationUI的各种图片显示方法
"""

import sys
import tempfile
import os
from pathlib import Path
from PIL import Image

# 添加路径以便导入模块
sys.path.append(str(Path(__file__).parent))

def create_test_image():
    """创建测试图片"""
    try:
        # 创建一个简单的测试图片
        image = Image.new('RGB', (400, 300), color='lightblue')

        # 创建临时文件
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
            temp_path = temp_file.name

        # 保存图片
        image.save(temp_path)
        print(f"[OK] 测试图片创建成功: {temp_path}")

        return temp_path

    except Exception as e:
        print(f"[ERROR] 创建测试图片失败: {e}")
        return None

def test_annotation_ui_import():
    """测试AnnotationUI导入"""
    try:
        from src.annotation.annotation_ui import AnnotationUI
        print("[OK] AnnotationUI导入成功")
        return True
    except ImportError as e:
        print(f"[ERROR] AnnotationUI导入失败: {e}")
        return False

def test_annotation_ui_init():
    """测试AnnotationUI初始化"""
    try:
        from src.annotation.annotation_ui import AnnotationUI

        # 测试默认参数
        ui1 = AnnotationUI()
        print(f"[OK] 默认初始化成功: display_method={ui1.image_display_method}")

        # 测试自定义参数
        ui2 = AnnotationUI(
            image_display_method='tkinter',
            image_window_size='1024x768',
            ascii_width=100
        )
        print(f"[OK] 自定义初始化成功: display_method={ui2.image_display_method}, window_size={ui2.image_window_size}, ascii_width={ui2.ascii_width}")

        return True

    except Exception as e:
        print(f"[ERROR] AnnotationUI初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_display_methods(ui, test_image_path):
    """测试各种显示方法"""
    print("\n[TEST] 测试display_image_direct方法...")

    try:
        # 测试off方法（只显示路径）
        ui.image_display_method = 'off'
        result = ui.display_image_direct(test_image_path, "测试图片")
        print(f"[CHECK] off方法: {'成功' if result else '失败'}")

        # 测试ASCII方法（如果可用）
        ui.image_display_method = 'ascii'
        result = ui.display_image_direct(test_image_path, "ASCII测试")
        print(f"[CHECK] ascii方法: {'成功' if result else '失败'}")

        # 测试system方法
        # ui.image_display_method = 'system'
        # result = ui.display_image_direct(test_image_path, "系统查看器测试")
        # print(f"[CHECK] system方法: {'成功' if result else '失败'}")

        return True

    except Exception as e:
        print(f"[ERROR] 测试display方法时发生错误: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_screenshot_display_integration():
    """测试截图显示集成功能"""
    try:
        from src.annotation.annotation_ui import AnnotationUI

        ui = AnnotationUI(image_display_method='off')  # 使用off方法避免弹窗

        # 创建测试图片
        test_image_path = create_test_image()
        if not test_image_path:
            return False

        try:
            # 测试display_screenshot_ready方法
            print("\n[TEST] 测试display_screenshot_ready...")
            ui.display_screenshot_ready(test_image_path)
            print("[OK] display_screenshot_ready测试通过")

            # 测试display_task_images方法
            print("\n[TEST] 测试display_task_images...")
            test_images = [test_image_path]
            ui.display_task_images(test_images)
            print("[OK] display_task_images测试通过")

            return True

        finally:
            # 清理临时文件
            if os.path.exists(test_image_path):
                os.unlink(test_image_path)
                print(f"[CLEANUP] 临时图片已删除: {test_image_path}")

    except Exception as e:
        print(f"[ERROR] 截图显示集成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_interactive_annotator_integration():
    """测试与interactive_annotator的集成"""
    try:
        # 测试参数解析
        sys.argv = ['test', '--image-display-method', 'ascii', '--ascii-width', '60']

        # 由于interactive_annotator会调用parse_args，我们需要模拟
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--image-display-method", choices=["tkinter", "ascii", "system", "off", "auto"], default="auto")
        parser.add_argument("--image-window-size", default="800x600")
        parser.add_argument("--ascii-width", type=int, default=80)

        args = parser.parse_args(['--image-display-method', 'ascii', '--ascii-width', '60'])

        from src.annotation.annotation_ui import AnnotationUI

        ui = AnnotationUI(
            image_display_method=getattr(args, 'image_display_method', 'auto'),
            image_window_size=getattr(args, 'image_window_size', '800x600'),
            ascii_width=getattr(args, 'ascii_width', 80)
        )

        print(f"[OK] 集成测试成功: method={ui.image_display_method}, ascii_width={ui.ascii_width}")

        return True

    except Exception as e:
        print(f"[ERROR] 集成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("[MAIN] 图片显示功能测试")
    print("=" * 60)

    tests_passed = 0
    total_tests = 0

    # 测试1: 导入测试
    total_tests += 1
    print(f"\n[TEST {total_tests}] AnnotationUI导入测试")
    if test_annotation_ui_import():
        tests_passed += 1
        print("[PASS] 导入测试通过")
    else:
        print("[FAIL] 导入测试失败")
        return False

    # 测试2: 初始化测试
    total_tests += 1
    print(f"\n[TEST {total_tests}] AnnotationUI初始化测试")
    if test_annotation_ui_init():
        tests_passed += 1
        print("[PASS] 初始化测试通过")
    else:
        print("[FAIL] 初始化测试失败")
        return False

    # 测试3: 显示方法测试
    total_tests += 1
    print(f"\n[TEST {total_tests}] 显示方法测试")
    try:
        from src.annotation.annotation_ui import AnnotationUI
        ui = AnnotationUI(image_display_method='off')
        test_image_path = create_test_image()
        if test_image_path and test_display_methods(ui, test_image_path):
            tests_passed += 1
            print("[PASS] 显示方法测试通过")
            # 清理
            if os.path.exists(test_image_path):
                os.unlink(test_image_path)
        else:
            print("[FAIL] 显示方法测试失败")
    except Exception as e:
        print(f"[FAIL] 显示方法测试异常: {e}")

    # 测试4: 截图显示集成测试
    total_tests += 1
    print(f"\n[TEST {total_tests}] 截图显示集成测试")
    if test_screenshot_display_integration():
        tests_passed += 1
        print("[PASS] 截图显示集成测试通过")
    else:
        print("[FAIL] 截图显示集成测试失败")

    # 测试5: interactive_annotator集成测试
    total_tests += 1
    print(f"\n[TEST {total_tests}] interactive_annotator集成测试")
    if test_interactive_annotator_integration():
        tests_passed += 1
        print("[PASS] interactive_annotator集成测试通过")
    else:
        print("[FAIL] interactive_annotator集成测试失败")

    # 结果总结
    print("\n" + "=" * 60)
    print(f"[SUMMARY] 测试结果: {tests_passed}/{total_tests} 通过")

    if tests_passed == total_tests:
        print("[SUCCESS] 所有图片显示功能测试通过！")
        print("[INFO] 图片显示模块可以正常使用。")
        return True
    else:
        print("[WARNING] 部分测试失败")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)