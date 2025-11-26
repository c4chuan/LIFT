"""
单个轨迹处理测试脚本

用于测试完整的处理流程
"""

import os
import sys
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.reasoning_filler.trajectory_loader import TrajectoryLoader
from src.reasoning_filler.prompt_builder import PromptBuilder
from src.reasoning_filler.qwen_caller import QwenCaller
from src.reasoning_filler.trajectory_filler import TrajectoryFiller
from src.reasoning_filler.progress_tracker import ProgressTracker
from visualwebarena.src.envs.actions import Action


def test_trajectory_loader():
    """测试轨迹加载"""
    print("=" * 70)
    print("测试 1: 轨迹加载")
    print("=" * 70)

    loader = TrajectoryLoader()

    # 获取统计
    stats = loader.get_statistics()
    print(f"\n总轨迹数: {stats['total']}")
    print(f"环境分布:")
    for env, count in stats['by_environment'].items():
        print(f"  {env}: {count}")

    # 扫描第一个轨迹
    trajectories = loader.scan_trajectories()
    if not trajectories:
        print("\n错误: 未找到轨迹文件")
        return None

    first_traj = trajectories[0]
    print(f"\n测试轨迹:")
    print(f"  环境: {first_traj.env_name}")
    print(f"  任务ID: {first_traj.task_id}")
    print(f"  文件: {first_traj.trajectory_file.name}")

    # 加载轨迹
    trajectory, metadata = loader.load_trajectory(first_traj)
    print(f"\n轨迹信息:")
    print(f"  轨迹长度: {len(trajectory)}")
    print(f"  任务目标: {metadata.get('intent', 'N/A')}")
    print(f"  得分: {metadata.get('score', 'N/A')}")

    print("\n✓ 轨迹加载测试通过\n")
    return first_traj, trajectory, metadata


def test_prompt_builder(trajectory, metadata):
    """测试 Prompt 构建"""
    print("=" * 70)
    print("测试 2: Prompt 构建")
    print("=" * 70)

    builder = PromptBuilder(prompt_style="LIFT")

    # 找到第一个 StateInfo 和 Action
    state_info = None
    action = None

    for item in trajectory:
        if isinstance(item, dict) and "observation" in item and "info" in item:
            # StateInfo
            state_info = item
        elif isinstance(item, Action):
            # Action
            if state_info:
                action = item
                break

    if not state_info or not action:
        print("错误: 未找到有效的 StateInfo 或 Action")
        return False

    print(f"\n找到第一个动作:")
    print(f"  action_type: {getattr(action, 'action_type', 'N/A')}")

    # 测试简化 prompt
    print("\n构建简化 Prompt...")
    messages = builder.build_simplified_prompt(
        intent=metadata.get('intent', ''),
        ground_truth_action=f"action_{action.action_type}",
        previous_action="None"
    )

    print(f"✓ Prompt 构建成功")
    print(f"  消息数: {len(messages)}")
    print(f"  系统提示长度: {len(messages[0]['content'])} 字符")

    print("\n✓ Prompt 构建测试通过\n")
    return True


def test_qwen_caller(dry_run=True):
    """测试 Qwen API 调用"""
    print("=" * 70)
    print("测试 3: Qwen API 调用")
    print("=" * 70)

    # 获取 API key
    api_key = os.environ.get("DASHSCOPE_API_KEY")

    if not api_key:
        print("\n⚠ 未设置 DASHSCOPE_API_KEY，跳过 API 测试")
        return True

    if dry_run:
        print("\n⚠ Dry run 模式，跳过实际 API 调用")
        print("  提示: 运行时加上 --no-dry-run 参数进行实际测试")
        return True

    try:
        caller = QwenCaller(api_key=api_key)

        # 简单测试
        messages = [
            {"role": "system", "content": "你是一个有帮助的助手。"},
            {"role": "user", "content": "请用一句话介绍你自己。"}
        ]

        print("\n调用 API...")
        response = caller.call(messages)

        print(f"✓ API 调用成功")
        print(f"  响应长度: {len(response)} 字符")
        print(f"  响应内容: {response[:100]}...")

        print("\n✓ Qwen API 测试通过\n")
        return True

    except Exception as e:
        print(f"\n✗ API 调用失败: {e}\n")
        return False


def test_full_pipeline(dry_run=True):
    """测试完整流程"""
    print("=" * 70)
    print("测试 4: 完整流程")
    print("=" * 70)

    # 1. 加载轨迹
    result = test_trajectory_loader()
    if not result:
        return False

    traj_info, trajectory, metadata = result

    # 2. 测试 Prompt 构建
    if not test_prompt_builder(trajectory, metadata):
        return False

    # 3. 测试 API 调用
    if not test_qwen_caller(dry_run=dry_run):
        return False

    print("=" * 70)
    print("所有测试通过!")
    print("=" * 70)

    if dry_run:
        print("\n提示: 这是 dry run 模式，未进行实际的轨迹填充")
        print("运行以下命令进行完整测试:")
        print("  python src/reasoning_filler/test_single.py --no-dry-run")

    return True


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="测试轨迹推理填充系统")
    parser.add_argument(
        "--no-dry-run",
        action="store_true",
        help="执行实际的 API 调用"
    )

    args = parser.parse_args()

    print("\n轨迹推理填充系统 - 测试脚本\n")

    try:
        success = test_full_pipeline(dry_run=not args.no_dry_run)

        if success:
            print("\n🎉 测试成功!\n")
            sys.exit(0)
        else:
            print("\n❌ 测试失败\n")
            sys.exit(1)

    except Exception as e:
        print(f"\n❌ 测试出错: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
