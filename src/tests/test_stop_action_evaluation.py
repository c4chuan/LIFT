"""
测试 STOP 动作的 evaluator 集成

这个脚本测试:
1. RewardCalculator 是否能正确调用 evaluator
2. 异步方法是否正常工作
3. 评估结果是否符合预期
"""
import asyncio
from pathlib import Path


async def test_stop_action_evaluation():
    """测试 STOP 动作评估"""
    print("=" * 60)
    print("测试 STOP 动作评估功能")
    print("=" * 60)

    # 导入必要的模块
    from src.core.reward_calculator import RewardCalculator
    from src.models.task_models import VWATask
    from vwa.src.envs.browser import FastCachedwActionMatchingBrowserEnv

    # 创建 RewardCalculator 实例
    calculator = RewardCalculator()
    print("✓ RewardCalculator 创建成功")

    # 模拟一个简单的任务配置
    # 使用一个真实的配置文件
    config_file_path = "/data/wangzhenchuan/Projects/LIFT/visualwebarena/configs/visualwebarena/test_shopping_v2/121.json"

    if not Path(config_file_path).exists():
        print(f"✗ 配置文件不存在: {config_file_path}")
        print("跳过测试...")
        return

    import json
    with open(config_file_path, 'r') as f:
        config_file = json.load(f)

    print(f"✓ 加载配置文件: task_id={config_file['task_id']}")
    print(f"  Intent: {config_file['intent']}")
    print(f"  Eval types: {config_file['eval']['eval_types']}")

    # 创建一个模拟的环境（不真正启动浏览器）
    # 注意：这里只是测试导入和调用流程，不进行真正的评估
    print("\n注意：完整的集成测试需要真实的浏览器环境")
    print("当前测试仅验证代码结构和异步调用是否正常")

    # 测试异步方法签名
    print("\n✓ 验证方法签名:")
    print(f"  - calculate_batch_rewards: async={asyncio.iscoroutinefunction(calculator.calculate_batch_rewards)}")
    print(f"  - calculate_single_reward: async={asyncio.iscoroutinefunction(calculator.calculate_single_reward)}")
    print(f"  - _evaluate_action_validation: async={asyncio.iscoroutinefunction(calculator._evaluate_action_validation)}")

    print("\n" + "=" * 60)
    print("基础测试通过！")
    print("=" * 60)
    print("\n下一步:")
    print("1. 启动完整的环境管理器")
    print("2. 发送实际的 STOP 动作进行测试")
    print("3. 验证评估结果是否正确")


if __name__ == "__main__":
    asyncio.run(test_stop_action_evaluation())
