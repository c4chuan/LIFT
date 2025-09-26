#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试Action类型不匹配问题是否已修复
"""

import sys
from pathlib import Path

# 添加路径
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / "visualwebarena"))

def test_action_type_compatibility():
    """测试Action类型兼容性"""
    print("Testing Action type compatibility...")

    try:
        # 导入修复后的模块
        print("1. Testing imports...")

        # 测试browser.py的Action导入
        from visualwebarena.src.envs.browser import FastCachedwActionMatchingBrowserEnv
        print("   ✅ Browser environment imported successfully")

        # 测试input_parser的Action创建
        from src.annotation.input_parser import InputParser
        print("   ✅ InputParser imported successfully")

        # 创建Action实例测试类型
        print("\n2. Testing Action type consistency...")

        parser = InputParser()
        action = parser.parse_command("click [10]")
        print(f"   Action type: {type(action)}")
        print(f"   Action module: {type(action).__module__}")

        # 检查Action是否有metadata属性
        if hasattr(action, 'metadata'):
            print("   ✅ Action has metadata attribute")
        else:
            print("   ❌ Action missing metadata attribute")
            return False

        print("\n3. Testing type hint compatibility...")

        # 模拟类型检查（不实际调用，因为需要完整环境）
        from visualwebarena.src.envs.actions import Action as ExpectedAction

        if isinstance(action, ExpectedAction):
            print("   ✅ Action is compatible with expected type")
        else:
            print(f"   ❌ Type mismatch: {type(action)} vs {ExpectedAction}")
            return False

        return True

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("Action Type Compatibility Test")
    print("=" * 40)

    success = test_action_type_compatibility()

    print("\n" + "=" * 40)
    if success:
        print("✅ Action type compatibility test PASSED!")
        print("\nThe beartype error should be resolved.")
        print("Both browser.py and input_parser.py now use the same Action type.")
    else:
        print("❌ Action type compatibility test FAILED!")
        print("\nPlease check the import statements and type consistency.")

if __name__ == "__main__":
    main()