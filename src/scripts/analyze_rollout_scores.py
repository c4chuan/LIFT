#!/usr/bin/env python3
"""
Rollout 数据得分分析脚本

分析指定编号的 JSONL 文件中所有 rollout 样本的得分分布、均值和方差。

用法:
    python analyze_rollout_scores.py -n 100
    python analyze_rollout_scores.py -n 100 -d rollout_data/sup_rollout_data_dir
"""

import argparse
import json
from pathlib import Path
import numpy as np


def read_jsonl_scores(file_path: Path) -> list[dict]:
    """读取 JSONL 文件并提取所有记录的得分字段"""
    scores = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                score_data = {
                    'score': record.get('score', 0.0),
                    'shift_reward': record.get('shift_reward', 0.0),
                    'zoom_reward': record.get('zoom_reward', 0.0),
                    'format_reward': record.get('format_reward', 0.0),
                    'valid_action_reward': record.get('valid_action_reward', 0.0),
                }
                scores.append(score_data)
    return scores


def create_ascii_histogram(values: np.ndarray, bins: int = 10, width: int = 40) -> str:
    """创建 ASCII 直方图"""
    if len(values) == 0 or np.all(values == values[0]):
        return "▇" * width

    hist, bin_edges = np.histogram(values, bins=bins)
    max_count = max(hist) if max(hist) > 0 else 1

    # 使用 Unicode 方块字符表示高度
    blocks = ' ▁▂▃▄▅▆▇█'

    histogram_str = ""
    for count in hist:
        height = int((count / max_count) * (len(blocks) - 1))
        histogram_str += blocks[height]

    return histogram_str


def compute_statistics(values: list[float]) -> dict:
    """计算统计指标"""
    arr = np.array(values)
    return {
        'count': len(arr),
        'mean': np.mean(arr),
        'variance': np.var(arr),
        'std': np.std(arr),
        'min': np.min(arr),
        'max': np.max(arr),
        'median': np.median(arr),
        'q25': np.percentile(arr, 25),
        'q75': np.percentile(arr, 75),
    }


def print_score_analysis(name: str, values: list[float], show_histogram: bool = True):
    """打印单个得分字段的分析结果"""
    stats = compute_statistics(values)

    print(f"\n[{name}]")
    print(f"  均值: {stats['mean']:.4f}  方差: {stats['variance']:.4f}  标准差: {stats['std']:.4f}")
    print(f"  最小: {stats['min']:.4f}  最大: {stats['max']:.4f}  中位数: {stats['median']:.4f}")
    print(f"  Q25: {stats['q25']:.4f}  Q75: {stats['q75']:.4f}")

    if show_histogram:
        hist = create_ascii_histogram(np.array(values))
        print(f"  分布: {hist}")


def main():
    parser = argparse.ArgumentParser(
        description='分析 rollout 数据的得分分布、均值和方差'
    )
    parser.add_argument(
        '-n', '--number',
        type=int,
        required=False,
        default=5,
        help='JSONL 文件编号（如 100 对应 100.jsonl）'
    )
    parser.add_argument(
        '-d', '--data-dir',
        type=str,
        default='rollout_data/sup_rollout_wo_obs',
        help='数据目录路径（默认: rollout_data/sup_rollout_wo_obs）'
    )
    parser.add_argument(
        '--no-histogram',
        action='store_true',
        help='不显示直方图'
    )

    args = parser.parse_args()

    # 构建文件路径
    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / args.data_dir
    file_path = data_dir / f"{args.number}.jsonl"

    # 检查文件是否存在
    if not file_path.exists():
        print(f"错误: 文件不存在 - {file_path}")
        print(f"\n可用的文件编号:")
        if data_dir.exists():
            files = sorted(data_dir.glob("*.jsonl"), key=lambda x: int(x.stem) if x.stem.isdigit() else 0)
            if files:
                numbers = [f.stem for f in files[:10]]
                print(f"  {', '.join(numbers)}...")
                print(f"  共 {len(files)} 个文件")
            else:
                print("  目录中没有 JSONL 文件")
        else:
            print(f"  数据目录不存在: {data_dir}")
        return 1

    # 读取得分数据
    scores = read_jsonl_scores(file_path)

    if not scores:
        print(f"错误: 文件中没有有效数据 - {file_path}")
        return 1

    # 打印分析结果
    print("=" * 50)
    print(f"Rollout 得分分析: {args.number}.jsonl")
    print("=" * 50)
    print(f"数据目录: {data_dir}")
    print(f"样本数量: {len(scores)}")

    # 提取各字段的值
    score_fields = ['score', 'shift_reward', 'zoom_reward', 'format_reward', 'valid_action_reward']
    field_names = {
        'score': 'score (总分)',
        'shift_reward': 'shift_reward (平移奖励)',
        'zoom_reward': 'zoom_reward (缩放奖励)',
        'format_reward': 'format_reward (格式奖励)',
        'valid_action_reward': 'valid_action_reward (有效动作奖励)',
    }

    for field in score_fields:
        values = [s[field] for s in scores]
        print_score_analysis(field_names[field], values, show_histogram=not args.no_histogram)

    # 计算 advantage (score - mean) / std
    all_scores = [s['score'] for s in scores]
    mean_score = np.mean(all_scores)
    std_score = np.std(all_scores)

    # 避免除以零
    if std_score > 1e-8:
        advantages = [(score - mean_score) / std_score for score in all_scores]
    else:
        advantages = [0.0 for _ in all_scores]

    # 打印 advantage 统计信息
    print_score_analysis("advantage ((score - mean) / std)", advantages, show_histogram=not args.no_histogram)

    # 打印原始得分和 advantage 列表
    print("\n" + "-" * 50)
    print("各样本得分与 advantage 详情:")
    print(f"  {'样本':<6} {'score':<8} {'advantage':<10} {'状态'}")
    print("  " + "-" * 40)

    for i, (score, adv) in enumerate(zip(all_scores, advantages)):
        # 标记高于/低于平均
        if adv > 0:
            status = "↑ 高于均值"
        elif adv < 0:
            status = "↓ 低于均值"
        else:
            status = "= 等于均值"
        print(f"  [{i+1:2d}]   {score:>6.3f}   {adv:>+7.3f}    {status}")

    # 汇总统计
    print("\n" + "-" * 50)
    print("Advantage 汇总:")
    above_mean = sum(1 for a in advantages if a > 0)
    below_mean = sum(1 for a in advantages if a < 0)
    at_mean = sum(1 for a in advantages if a == 0)
    print(f"  高于均值: {above_mean} 个样本 ({above_mean/len(advantages)*100:.1f}%)")
    print(f"  低于均值: {below_mean} 个样本 ({below_mean/len(advantages)*100:.1f}%)")
    print(f"  等于均值: {at_mean} 个样本 ({at_mean/len(advantages)*100:.1f}%)")

    return 0


if __name__ == '__main__':
    exit(main())
