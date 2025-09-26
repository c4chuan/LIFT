#!/usr/bin/env python3
"""
测试Action类型一致性
验证评估器类型检查问题是否修复
"""

import sys
from pathlib import Path

# 添加路径以便导入模块
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / "visualwebarena"))

def test_action_type_consistency():
    """测试Action类型一致性"""
    print("=" * 70)
    print("测试Action类型一致性")
    print("=" * 70)

    try:
        # 模拟environment_controller.py的导入
        print("\n[测试1] 导入environment_controller使用的Action类:")
        from visualwebarena.src.envs.actions import Action as ControllerAction
        print(f"[OK] Controller Action类: {ControllerAction}")
        print(f"   模块路径: {ControllerAction.__module__}")

        # 模拟评估器的导入
        print("\n[测试2] 导入评估器使用的Action类:")
        from src.envs.actions import Action as EvaluatorAction
        print(f"[OK] Evaluator Action类: {EvaluatorAction}")
        print(f"   模块路径: {EvaluatorAction.__module__}")

        # 检查是否是同一个类
        print("\n[测试3] 检查类型一致性:")
        if ControllerAction is EvaluatorAction:
            print("[SUCCESS] Controller和Evaluator使用相同的Action类")
            return True
        else:
            print("[FAIL] Controller和Evaluator使用不同的Action类")
            print(f"   Controller Action ID: {id(ControllerAction)}")
            print(f"   Evaluator Action ID: {id(EvaluatorAction)}")
            return False

    except Exception as e:
        print(f"[ERROR] 导入测试失败: {e}")
        return False

def test_action_creation():
    """测试Action对象创建和类型检查"""
    print("\n" + "=" * 70)
    print("测试Action对象创建和类型检查")
    print("=" * 70)

    try:
        from visualwebarena.src.envs.actions import Action, ActionTypes
        from src.envs.actions import Action as EvaluatorAction
        import numpy as np

        # 创建一个Action对象（模拟environment_controller创建的）
        print("\n[测试4] 创建Action对象:")
        test_action = Action(
            action_type=ActionTypes.STOP,
            coords=np.array([0.0, 0.0], dtype=np.float32),
            element_role=0,
            element_name="",
            element_id="",
            element_pw_code="",
            text="",
            raw_prediction="stop [$100.00]"
        )
        print(f"[OK] 成功创建Action对象: {test_action}")
        print(f"   对象类型: {type(test_action)}")
        print(f"   动作类型: {test_action.action_type}")

        # 测试类型检查
        print("\n[测试5] 类型检查测试:")
        if isinstance(test_action, EvaluatorAction):
            print("[SUCCESS] Action对象通过评估器类型检查")
            return True
        else:
            print("[FAIL] Action对象未通过评估器类型检查")
            print(f"   对象类: {type(test_action)}")
            print(f"   期望类: {EvaluatorAction}")
            return False

    except Exception as e:
        print(f"[ERROR] Action创建测试失败: {e}")
        return False

def test_beartype_compatibility():
    """测试beartype兼容性"""
    print("\n" + "=" * 70)
    print("测试beartype类型检查兼容性")
    print("=" * 70)

    try:
        from visualwebarena.src.envs.actions import Action
        from src.envs.actions import Action as EvaluatorAction
        from browser_env.utils import StateInfo
        from beartype.door import is_bearable
        from typing import Union, List
        import numpy as np

        # 创建测试对象
        print("\n[测试6] 创建测试轨迹数据:")

        # 创建Action对象
        test_action = Action(
            action_type=1,  # STOP
            coords=np.array([0.0, 0.0], dtype=np.float32),
            element_role=0,
            element_name="",
            element_id="",
            element_pw_code="",
            text="",
            raw_prediction="stop [$100.00]"
        )

        # 创建StateInfo对象
        state_info = {
            "observation": {"text": "test"},
            "info": {"url": "test.com"},
            "url": "test.com"
        }

        # 创建轨迹
        trajectory = [state_info, test_action]

        print(f"[OK] 轨迹创建成功，包含 {len(trajectory)} 个元素")

        # 测试beartype类型检查
        print("\n[测试7] beartype类型检查:")
        TrajectoryType = List[Union[EvaluatorAction, StateInfo]]

        if is_bearable(trajectory, TrajectoryType):
            print("[SUCCESS] 轨迹通过beartype类型检查")
            return True
        else:
            print("[FAIL] 轨迹未通过beartype类型检查")

            # 详细检查每个元素
            for i, item in enumerate(trajectory):
                print(f"   元素 {i}: {type(item)}")
                if hasattr(item, '__module__'):
                    print(f"      模块: {item.__module__}")
                is_action_bearable = is_bearable(item, EvaluatorAction)
                is_state_bearable = is_bearable(item, StateInfo)
                print(f"      Action类型检查: {is_action_bearable}")
                print(f"      StateInfo类型检查: {is_state_bearable}")

            return False

    except Exception as e:
        print(f"[ERROR] beartype兼容性测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("[Action类型一致性测试]")
    print("验证修复后的Action类导入是否解决了评估器类型检查问题")
    print()

    results = []

    # 运行所有测试
    results.append(test_action_type_consistency())
    results.append(test_action_creation())
    results.append(test_beartype_compatibility())

    # 输出总结
    print("\n" + "=" * 70)
    print("测试结果总结")
    print("=" * 70)

    success_count = sum(results)
    total_count = len(results)

    print(f"通过测试: {success_count}/{total_count}")

    if success_count == total_count:
        print("\n[SUCCESS] 所有测试通过! Action类型问题已修复")
        print("现在可以正常使用STOP动作进行评估了")
    else:
        print(f"\n[WARNING] {total_count - success_count}个测试失败")
        print("Action类型问题可能仍然存在，需要进一步检查")

    return success_count == total_count

if __name__ == "__main__":
    main()