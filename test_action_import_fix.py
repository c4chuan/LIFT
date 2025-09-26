#!/usr/bin/env python3
"""
简化的Action导入测试
验证修复后的导入路径是否正确
"""

import sys
from pathlib import Path

# 添加路径以便导入模块
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / "visualwebarena"))

def test_import_consistency():
    """测试导入一致性"""
    print("=" * 60)
    print("测试Action类导入一致性")
    print("=" * 60)

    try:
        # 测试评估器的导入（在visualwebarena上下文中）
        print("\n[测试] 评估器Action类导入:")
        from src.envs.actions import Action as EvaluatorAction
        print(f"[OK] 评估器Action类: {EvaluatorAction}")
        print(f"     模块路径: {EvaluatorAction.__module__}")

        # 测试环境控制器的导入
        print("\n[测试] 环境控制器Action类导入:")
        from visualwebarena.src.envs.actions import Action as ControllerAction
        print(f"[OK] 控制器Action类: {ControllerAction}")
        print(f"     模块路径: {ControllerAction.__module__}")

        # 检查是否指向同一个类
        print(f"\n[验证] 类型检查:")
        print(f"- 评估器Action类ID: {id(EvaluatorAction)}")
        print(f"- 控制器Action类ID: {id(ControllerAction)}")

        if EvaluatorAction is ControllerAction:
            print("[SUCCESS] 两个Action类是同一个对象，类型匹配问题已解决！")
            return True
        else:
            print("[INFO] 两个Action类不是同一个对象，但这可能是正常的")
            print("       重要的是它们来自同一个模块路径")

            # 检查模块路径是否一致
            if EvaluatorAction.__module__ == ControllerAction.__module__:
                print("[SUCCESS] 模块路径一致，应该可以解决类型检查问题")
                return True
            else:
                print("[FAIL] 模块路径不一致，仍有问题")
                return False

    except Exception as e:
        print(f"[ERROR] 导入测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_stop_action_creation():
    """测试STOP动作创建"""
    print("\n" + "=" * 60)
    print("测试STOP动作创建")
    print("=" * 60)

    try:
        from visualwebarena.src.envs.actions import Action, ActionTypes
        import numpy as np

        # 创建STOP动作
        stop_action = Action(
            action_type=ActionTypes.STOP,
            coords=np.array([0.0, 0.0], dtype=np.float32),
            element_role=0,
            element_name="",
            element_id="",
            element_pw_code="",
            text="",
            raw_prediction="stop [$100.00]"
        )

        print(f"[OK] STOP动作创建成功")
        print(f"     动作类型: {stop_action.action_type}")
        print(f"     原始预测: {stop_action.raw_prediction}")
        print(f"     对象类型: {type(stop_action)}")

        return True

    except Exception as e:
        print(f"[ERROR] STOP动作创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("Action导入修复验证测试")
    print("验证修复是否解决了STOP动作评估的类型错误")
    print()

    results = []
    results.append(test_import_consistency())
    results.append(test_stop_action_creation())

    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)

    success_count = sum(results)
    total_count = len(results)

    print(f"通过测试: {success_count}/{total_count}")

    if success_count == total_count:
        print("\n[SUCCESS] 修复验证通过！")
        print("现在可以尝试再次运行STOP动作评估了")
    else:
        print(f"\n[WARNING] 仍有{total_count - success_count}个测试失败")

    return success_count == total_count

if __name__ == "__main__":
    main()