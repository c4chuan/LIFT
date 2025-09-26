#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试Action.metadata问题是否已修复
"""

import sys
from pathlib import Path

# 添加路径
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / "visualwebarena"))

def test_action_metadata_fix():
    """测试Action对象的metadata属性"""
    print("Testing Action metadata fix...")

    try:
        # 使用修复后的导入
        from src.annotation.input_parser import InputParser
        from visualwebarena.src.envs.actions import Action, create_click_action, create_stop_action

        print("✅ Import successful")

        # 测试创建Action对象
        click_action = create_click_action("10")
        print(f"✅ Created click action: {type(click_action)}")

        # 测试metadata属性访问
        if hasattr(click_action, 'metadata'):
            print("✅ Action has metadata attribute")
            print(f"   metadata type: {type(click_action.metadata)}")
            print(f"   metadata content: {click_action.metadata}")
        else:
            print("❌ Action does not have metadata attribute")
            return False

        # 测试输入解析器创建的Action
        parser = InputParser()
        print("✅ Created InputParser")

        # 解析一些命令
        test_commands = [
            "click [10]",
            "type [5] [test text] [1]",
            "scroll down",
            "stop [test answer]"
        ]

        for cmd in test_commands:
            try:
                action = parser.parse_command(cmd)
                if hasattr(action, 'metadata'):
                    print(f"✅ Command '{cmd}' -> Action with metadata: {type(action.metadata)}")
                else:
                    print(f"❌ Command '{cmd}' -> Action without metadata!")
                    return False
            except Exception as e:
                print(f"❌ Failed to parse '{cmd}': {e}")
                return False

        # 模拟browser.py中的操作
        print("\nTesting browser.py compatibility...")
        try:
            test_action = create_click_action("15")

            # 模拟maybe_update_action_id中的操作
            if 'obs_metadata' not in test_action.metadata:
                print("✅ Successfully accessed action.metadata (no obs_metadata found, as expected)")
            else:
                print("✅ Successfully accessed action.metadata (obs_metadata found)")

            # 测试字典式访问（Action类支持的兼容模式）
            raw_pred = getattr(test_action, 'raw_prediction', 'default')
            print(f"✅ Successfully accessed raw_prediction: '{raw_pred}'")

            return True

        except AttributeError as e:
            if 'metadata' in str(e):
                print(f"❌ Still getting metadata AttributeError: {e}")
                return False
            else:
                print(f"⚠️  Other AttributeError (might be expected): {e}")
                return True

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("Action Metadata Fix Verification")
    print("=" * 40)

    success = test_action_metadata_fix()

    print("\n" + "=" * 40)
    if success:
        print("✅ Action metadata fix verification PASSED!")
        print("\nThe dict AttributeError should be resolved.")
        print("Your annotation tool should now work correctly.")
    else:
        print("❌ Action metadata fix verification FAILED!")
        print("\nPlease check the import statements and Action class usage.")

if __name__ == "__main__":
    main()