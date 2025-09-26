#!/usr/bin/env python3
"""
窗口响应性修复验证测试
测试修复后的Tkinter窗口是否能正常响应用户交互
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
        image = Image.new('RGB', (600, 400), color='lightcyan')
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
            temp_path = temp_file.name
        image.save(temp_path)
        return temp_path
    except Exception as e:
        print(f"[ERROR] 创建测试图片失败: {e}")
        return None

def test_window_creation():
    """测试窗口创建"""
    print("[TEST] 窗口创建测试")

    try:
        from src.annotation.annotation_ui import AnnotationUI

        ui = AnnotationUI(
            image_display_method='tkinter',
            persistent_window=True,
            keep_aspect_ratio=True
        )

        # 检查初始状态
        assert not ui.has_active_window(), "初始状态不应该有活跃窗口"
        print("[OK] 初始状态正确")

        # 测试窗口状态查询
        status = ui.get_window_status()
        assert not status["has_window"], "初始状态不应该有窗口引用"
        assert not status["window_exists"], "初始状态窗口不应该存在"
        assert status["persistent_mode"], "应该是持续模式"
        print("[OK] 窗口状态查询正确")

        return True

    except Exception as e:
        print(f"[ERROR] 窗口创建测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_gui_event_processing():
    """测试GUI事件处理"""
    print("\n[TEST] GUI事件处理测试")

    try:
        from src.annotation.annotation_ui import AnnotationUI

        test_image_path = create_test_image()
        if not test_image_path:
            return False

        try:
            ui = AnnotationUI(
                image_display_method='tkinter',
                persistent_window=True
            )

            # 显示图片
            print("[ACTION] 显示测试图片...")
            success = ui.display_image_direct(test_image_path, "响应性测试")
            assert success, "图片显示应该成功"

            # 检查窗口状态
            assert ui.has_active_window(), "应该有活跃窗口"
            print("[OK] 窗口创建成功")

            # 测试GUI事件处理
            print("[ACTION] 测试GUI事件处理...")
            for i in range(5):
                window_active = ui.process_gui_events()
                if not window_active:
                    print(f"[INFO] 窗口在第{i+1}次处理后被关闭（用户关闭）")
                    break
                print(f"[OK] 第{i+1}次GUI事件处理成功")
                time.sleep(0.1)

            # 测试窗口响应性维护
            print("[ACTION] 测试窗口响应性维护...")
            responsive = ui.ensure_window_responsiveness()
            print(f"[INFO] 窗口响应性状态: {responsive}")

            # 获取窗口状态
            status = ui.get_window_status()
            print(f"[INFO] 窗口状态: {status}")

            # 清理
            ui.cleanup()
            print("[OK] 窗口清理完成")

            return True

        finally:
            # 清理测试图片
            if os.path.exists(test_image_path):
                os.unlink(test_image_path)

    except Exception as e:
        print(f"[ERROR] GUI事件处理测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_input_with_gui_simulation():
    """测试GUI集成输入模拟"""
    print("\n[TEST] GUI集成输入模拟测试")

    try:
        from src.annotation.annotation_ui import AnnotationUI

        ui = AnnotationUI(
            image_display_method='off',  # 使用off模式避免实际窗口
            persistent_window=True
        )

        # 测试基础输入方法检查
        assert hasattr(ui, 'prompt_user_input_with_gui'), "应该有GUI集成输入方法"
        assert hasattr(ui, 'has_active_window'), "应该有窗口状态检查方法"
        assert hasattr(ui, 'process_gui_events'), "应该有GUI事件处理方法"

        print("[OK] GUI集成输入方法存在")

        # 测试窗口状态检查
        has_window = ui.has_active_window()
        print(f"[INFO] 当前窗口状态: {has_window}")

        # 测试事件处理方法调用
        event_result = ui.process_gui_events()
        print(f"[INFO] 事件处理结果: {event_result}")

        return True

    except Exception as e:
        print(f"[ERROR] GUI集成输入测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_window_lifecycle():
    """测试窗口生命周期管理"""
    print("\n[TEST] 窗口生命周期管理测试")

    try:
        from src.annotation.annotation_ui import AnnotationUI

        test_image_path = create_test_image()
        if not test_image_path:
            return False

        try:
            ui = AnnotationUI(
                image_display_method='tkinter',
                persistent_window=True
            )

            # 第一次显示
            print("[ACTION] 第一次显示图片...")
            success1 = ui.display_image_direct(test_image_path, "生命周期测试 - 第1张")
            assert success1, "第一次显示应该成功"
            status1 = ui.get_window_status()
            print(f"[INFO] 第一次显示后状态: {status1}")

            # 第二次显示（应该更新现有窗口）
            print("[ACTION] 第二次显示图片...")
            success2 = ui.display_image_direct(test_image_path, "生命周期测试 - 第2张")
            assert success2, "第二次显示应该成功"
            status2 = ui.get_window_status()
            print(f"[INFO] 第二次显示后状态: {status2}")

            # 检查窗口是否复用
            assert ui.has_active_window(), "应该有活跃窗口"

            # 测试清理
            print("[ACTION] 测试窗口清理...")
            ui.cleanup()
            status_after_cleanup = ui.get_window_status()
            print(f"[INFO] 清理后状态: {status_after_cleanup}")

            assert not ui.has_active_window(), "清理后不应该有活跃窗口"
            print("[OK] 窗口生命周期管理正确")

            return True

        finally:
            # 清理测试图片
            if os.path.exists(test_image_path):
                os.unlink(test_image_path)

    except Exception as e:
        print(f"[ERROR] 窗口生命周期测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("[MAIN] 窗口响应性修复验证测试")
    print("=" * 60)

    tests_passed = 0
    total_tests = 0

    tests = [
        ("窗口创建", test_window_creation),
        ("GUI事件处理", test_gui_event_processing),
        ("GUI集成输入模拟", test_input_with_gui_simulation),
        ("窗口生命周期管理", test_window_lifecycle)
    ]

    for test_name, test_func in tests:
        total_tests += 1
        print(f"\n{'='*20} {test_name} {'='*20}")

        try:
            if test_func():
                tests_passed += 1
                print(f"[PASS] {test_name}测试通过")
            else:
                print(f"[FAIL] {test_name}测试失败")
        except Exception as e:
            print(f"[ERROR] {test_name}测试异常: {e}")

    # 结果总结
    print("\n" + "=" * 60)
    print(f"[SUMMARY] 测试结果: {tests_passed}/{total_tests} 通过")

    if tests_passed == total_tests:
        print("[SUCCESS] 窗口响应性修复验证通过！")
        print("[INFO] 修复效果：")
        print("  [OK] 窗口创建和状态管理正常")
        print("  [OK] GUI事件处理机制工作正常")
        print("  [OK] 窗口生命周期管理完善")
        print("  [OK] 支持GUI集成的用户输入")
        print("\n[NEXT] 现在可以测试实际的窗口响应性：")
        print("  1. 启动程序: python src/interactive_annotator.py --image-display-method tkinter")
        print("  2. 检查图片窗口是否可以:")
        print("     - 正常点击和拖拽")
        print("     - 响应关闭按钮")
        print("     - 在用户输入时保持响应")
        return True
    else:
        print("[WARNING] 部分测试失败，需要进一步修复")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)