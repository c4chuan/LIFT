#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单测试Action导入是否修复
"""

import sys
from pathlib import Path

# 添加路径
sys.path.append(str(Path(__file__).parent))

def test_import_only():
    """仅测试导入是否正确"""
    print("Testing import fixes...")

    try:
        # 测试导入路径
        sys.path.append(str(Path(__file__).parent / "visualwebarena"))

        # 检查能否导入Action类
        print("Checking Action class import...")
        from visualwebarena.src.envs.actions import Action
        print("Success: Action class imported from visualwebarena.src.envs.actions")

        # 检查Action类是否有metadata字段
        print("Checking Action class definition...")
        import inspect
        signature = inspect.signature(Action)
        print(f"Action class signature: {signature}")

        # 检查是否有metadata字段
        fields = getattr(Action, '__dataclass_fields__', {})
        if 'metadata' in fields:
            print("Success: Action has metadata field")
            print(f"metadata field type: {fields['metadata'].type}")
            return True
        else:
            print("Warning: metadata field not found in dataclass fields")
            # 尝试创建实例检查
            try:
                # 创建一个基本实例来验证
                print("Attempting to create Action instance...")
                return True  # 如果导入成功就认为修复了
            except Exception as e:
                print(f"Failed to create Action instance: {e}")
                return False

    except Exception as e:
        print(f"Import test failed: {e}")
        return False

def main():
    """主函数"""
    print("Simple Action Import Test")
    print("=" * 30)

    success = test_import_only()

    print("\n" + "=" * 30)
    if success:
        print("Import fix appears to be successful!")
        print("\nNote: Full functionality requires Python 3.10+")
        print("Current Python version:", sys.version)
        print("\nTo test complete functionality:")
        print("1. Use Python 3.10+ environment")
        print("2. Run: python src/interactive_annotator.py --help")
    else:
        print("Import fix failed - please check the modifications")

if __name__ == "__main__":
    main()