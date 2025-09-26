#!/usr/bin/env python3
"""
图片显示逻辑测试脚本 - 不实际显示窗口
测试配置和方法是否正确工作
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
        image = Image.new('RGB', (400, 300), color='lightblue')
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
            temp_path = temp_file.name
        image.save(temp_path)
        return temp_path
    except Exception as e:
        print(f"[ERROR] 创建测试图片失败: {e}")
        return None

def test_ui_initialization():
    """测试AnnotationUI初始化"""
    print("[TEST] AnnotationUI初始化测试")

    try:
        from src.annotation.annotation_ui import AnnotationUI

        # 测试新参数的初始化
        ui = AnnotationUI(
            image_display_method='off',  # 使用off模式避免实际显示
            keep_aspect_ratio=True,
            persistent_window=True
        )

        # 检查新属性
        assert hasattr(ui, 'keep_aspect_ratio'), "缺少keep_aspect_ratio属性"
        assert hasattr(ui, 'persistent_window'), "缺少persistent_window属性"
        assert hasattr(ui, 'current_window'), "缺少current_window属性"

        assert ui.keep_aspect_ratio == True, "keep_aspect_ratio值不正确"
        assert ui.persistent_window == True, "persistent_window值不正确"
        assert ui.current_window is None, "current_window初始值应为None"

        print("[OK] AnnotationUI新属性初始化正确")

        # 测试不同配置
        ui_no_aspect = AnnotationUI(
            image_display_method='off',
            keep_aspect_ratio=False,
            persistent_window=False
        )

        assert ui_no_aspect.keep_aspect_ratio == False
        assert ui_no_aspect.persistent_window == False

        print("[OK] AnnotationUI配置参数工作正常")
        return True

    except Exception as e:
        print(f"[ERROR] AnnotationUI初始化测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_new_methods():
    """测试新方法是否存在"""
    print("\n[TEST] 新方法存在性测试")

    try:
        from src.annotation.annotation_ui import AnnotationUI

        ui = AnnotationUI(image_display_method='off')

        # 检查新方法
        new_methods = [
            '_setup_image_display',
            '_process_image_scaling',
            '_add_image_info',
            '_configure_window_behavior',
            '_update_existing_window',
            '_process_window_events',
            '_close_current_window',
            '_on_window_close',
            'cleanup'
        ]

        for method_name in new_methods:
            assert hasattr(ui, method_name), f"缺少方法: {method_name}"
            assert callable(getattr(ui, method_name)), f"方法不可调用: {method_name}"

        print("[OK] 所有新方法都存在且可调用")

        # 测试cleanup方法
        ui.cleanup()  # 应该不报错
        print("[OK] cleanup方法调用成功")

        return True

    except Exception as e:
        print(f"[ERROR] 新方法测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_image_scaling_logic():
    """测试图片缩放逻辑"""
    print("\n[TEST] 图片缩放逻辑测试")

    try:
        from src.annotation.annotation_ui import AnnotationUI

        # 创建测试图片
        test_image_path = create_test_image()
        if not test_image_path:
            return False

        try:
            # 测试保持比例的缩放逻辑
            ui_aspect = AnnotationUI(
                image_display_method='off',
                keep_aspect_ratio=True
            )

            # 加载图片测试缩放处理
            test_image = Image.open(test_image_path)
            original_size = test_image.size

            # 调用缩放处理方法
            processed_image = ui_aspect._process_image_scaling(test_image)

            print(f"[CHECK] 原始尺寸: {original_size}")
            print(f"[CHECK] 处理后尺寸: {processed_image.size}")

            # 验证比例是否保持
            original_ratio = original_size[0] / original_size[1]
            processed_ratio = processed_image.size[0] / processed_image.size[1]

            ratio_diff = abs(original_ratio - processed_ratio)
            assert ratio_diff < 0.01, f"比例保持失败，差异: {ratio_diff}"

            print("[OK] 保持比例的缩放逻辑正确")

            # 测试不保持比例的逻辑
            ui_no_aspect = AnnotationUI(
                image_display_method='off',
                image_window_size='200x100',
                keep_aspect_ratio=False
            )

            processed_image_fixed = ui_no_aspect._process_image_scaling(test_image)
            expected_size = (200, 100)

            print(f"[CHECK] 固定尺寸处理后: {processed_image_fixed.size}")
            assert processed_image_fixed.size == expected_size, f"固定尺寸失败，期望{expected_size}，实际{processed_image_fixed.size}"

            print("[OK] 固定尺寸的缩放逻辑正确")

            return True

        finally:
            # 清理测试图片
            if os.path.exists(test_image_path):
                os.unlink(test_image_path)

    except Exception as e:
        print(f"[ERROR] 图片缩放逻辑测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_interactive_annotator_integration():
    """测试与interactive_annotator的集成"""
    print("\n[TEST] interactive_annotator集成测试")

    try:
        # 模拟参数对象
        class MockArgs:
            def __init__(self):
                self.image_display_method = 'off'
                self.image_window_size = '800x600'
                self.ascii_width = 80
                self.keep_aspect_ratio = True
                self.persistent_window = True

        args = MockArgs()

        from src.annotation.annotation_ui import AnnotationUI

        # 模拟interactive_annotator的初始化逻辑
        ui = AnnotationUI(
            image_display_method=getattr(args, 'image_display_method', 'auto'),
            image_window_size=getattr(args, 'image_window_size', '800x600'),
            ascii_width=getattr(args, 'ascii_width', 80),
            keep_aspect_ratio=getattr(args, 'keep_aspect_ratio', True),
            persistent_window=getattr(args, 'persistent_window', True)
        )

        # 验证参数传递正确
        assert ui.image_display_method == 'off'
        assert ui.keep_aspect_ratio == True
        assert ui.persistent_window == True

        print("[OK] interactive_annotator集成参数传递正确")

        # 测试cleanup调用
        ui.cleanup()
        print("[OK] cleanup方法集成正常")

        return True

    except Exception as e:
        print(f"[ERROR] interactive_annotator集成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_off_mode_display():
    """测试off模式显示（实际可以运行的显示测试）"""
    print("\n[TEST] off模式显示测试")

    try:
        from src.annotation.annotation_ui import AnnotationUI

        test_image_path = create_test_image()
        if not test_image_path:
            return False

        try:
            ui = AnnotationUI(
                image_display_method='off',
                keep_aspect_ratio=True,
                persistent_window=True
            )

            # 测试显示（off模式只显示路径）
            success = ui.display_image_direct(test_image_path, "测试图片")

            assert success, "off模式显示应该返回True"
            print("[OK] off模式显示功能正常")

            # 测试多次显示（模拟持续显示）
            for i in range(3):
                success = ui.display_image_direct(test_image_path, f"测试图片{i+1}")
                assert success, f"第{i+1}次显示失败"

            print("[OK] 持续显示逻辑正常（off模式）")

            return True

        finally:
            if os.path.exists(test_image_path):
                os.unlink(test_image_path)

    except Exception as e:
        print(f"[ERROR] off模式显示测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("[MAIN] 图片显示逻辑测试")
    print("=" * 50)

    tests_passed = 0
    total_tests = 0

    tests = [
        ("AnnotationUI初始化", test_ui_initialization),
        ("新方法存在性", test_new_methods),
        ("图片缩放逻辑", test_image_scaling_logic),
        ("interactive_annotator集成", test_interactive_annotator_integration),
        ("off模式显示", test_off_mode_display)
    ]

    for test_name, test_func in tests:
        total_tests += 1
        print(f"\n{'='*20} {test_name} {'='*20}")

        if test_func():
            tests_passed += 1
            print(f"[PASS] {test_name}测试通过")
        else:
            print(f"[FAIL] {test_name}测试失败")

    # 结果总结
    print("\n" + "=" * 50)
    print(f"[SUMMARY] 测试结果: {tests_passed}/{total_tests} 通过")

    if tests_passed == total_tests:
        print("[SUCCESS] 所有图片显示逻辑测试通过！")
        print("[INFO] 新功能的核心逻辑已正确实现：")
        print("  [OK] 配置参数正确传递和存储")
        print("  [OK] 新方法都已正确添加")
        print("  [OK] 图片缩放逻辑按预期工作")
        print("  [OK] 与主程序集成正常")
        print("  [OK] 基础显示功能正常")
        return True
    else:
        print("[WARNING] 部分测试失败")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)