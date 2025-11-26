#!/usr/bin/env python3
"""
轨迹查看器
用于查看、分析和管理已保存的标注轨迹
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

import pickle
import lzma
import os


def read_pkl_xz(file_path):
    """
    读取 .pkl.xz 文件的函数

    参数:
        file_path (str): .pkl.xz 文件的路径

    返回:
        解压缩和反序列化后的Python对象
    """

    try:
        # 使用 lzma 打开压缩文件，然后用 pickle 加载
        with lzma.open(file_path, 'rb') as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        raise Exception(f"读取文件时出错: {e}")


def list_environments(base_path: str = "data/annotate/trajectories") -> List[str]:
    """列出所有环境目录"""
    path = Path(base_path)
    if not path.exists():
        return []
    return sorted([d.name for d in path.iterdir() if d.is_dir()])


def list_trajectories(env_name: str, base_path: str = "data/annotate/trajectories") -> List[Path]:
    """列出指定环境的所有轨迹文件"""
    env_path = Path(base_path) / env_name
    if not env_path.exists():
        return []
    return sorted(env_path.glob("*.pkl.xz"))


def format_action(action) -> str:
    """格式化动作信息"""
    if isinstance(action, dict):
        action_type = action.get('action_type', 'unknown')
        return f"动作: {action_type}"
    return f"动作: {str(action)[:100]}"


def format_observation(obs) -> str:
    """格式化观察信息摘要"""
    if isinstance(obs, dict):
        keys = list(obs.keys())
        return f"观察字段: {', '.join(keys[:5])}"
    return f"观察: {str(type(obs).__name__)}"


def view_trajectory(file_path: Path):
    """查看轨迹详细步骤"""
    print(f"\n{'='*70}")
    print(f"轨迹文件: {file_path.name}")
    print(f"路径: {file_path}")
    print(f"{'='*70}\n")

    # 读取轨迹数据
    data = read_pkl_xz(str(file_path))

    # 判断数据类型
    trajectory = None
    if isinstance(data, dict):
        if 'trajectory' in data:
            trajectory = data['trajectory']
        elif 'states' in data or 'actions' in data:
            # 其他可能的键名
            trajectory = data.get('states') or data.get('actions')
    elif isinstance(data, list):
        trajectory = data

    if trajectory is None:
        print("无法识别轨迹格式")
        print(f"数据类型: {type(data)}")
        if isinstance(data, dict):
            print(f"可用键: {list(data.keys())}")
        return

    # 显示轨迹步骤
    print(f"轨迹总步数: {len(trajectory)}\n")

    for i, step in enumerate(trajectory):
        print(f"--- 步骤 {i+1} ---")

        if isinstance(step, dict):
            # StateInfo 格式
            if 'observation' in step:
                obs = step['observation']
                print(f"  {format_observation(obs)}")

            if 'info' in step:
                info = step['info']
                if isinstance(info, dict):
                    # 显示关键信息字段
                    for key in ['url', 'page_title', 'timestamp']:
                        if key in info:
                            value = str(info[key])[:60]
                            print(f"  {key}: {value}")

            # Action 格式
            if 'action_type' in step or 'action' in step:
                print(f"  {format_action(step)}")

        else:
            # 其他格式
            print(f"  {str(step)[:100]}")

        print()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="轨迹查看器 - 查看标注轨迹的步骤内容")

    parser.add_argument(
        "--path",
        type=str,
        default="data/annotate/trajectories",
        help="轨迹文件基础路径"
    )

    parser.add_argument(
        "--env",
        type=str,
        help="指定要查看的环境名称",
        default="classifieds"
    )

    parser.add_argument(
        "--file",
        type=str,
        help="指定要查看的轨迹文件路径"
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="列出所有环境和轨迹文件"
    )

    args = parser.parse_args()

    # 列出所有环境和文件
    if args.list:
        envs = list_environments(args.path)
        print(f"\n{'='*70}")
        print("所有环境和轨迹文件:")
        print(f"{'='*70}\n")

        for env in envs:
            trajectories = list_trajectories(env, args.path)
            print(f"{env} ({len(trajectories)} 个文件):")
            for traj in trajectories:
                size = traj.stat().st_size / 1024
                print(f"  - {traj.name} ({size:.1f} KB)")
            print()
        return

    # 查看指定环境的轨迹
    if args.env:
        trajectories = list_trajectories(args.env, args.path)
        if not trajectories:
            print(f"环境 '{args.env}' 中没有轨迹文件")
            return

        print(f"\n环境: {args.env}")
        print(f"找到 {len(trajectories)} 个轨迹文件\n")

        for i, traj in enumerate(trajectories, 1):
            print(f"{i}. {traj.name}")

        # 让用户选择
        try:
            choice = input(f"\n请选择要查看的轨迹 (1-{len(trajectories)}, 回车退出): ")
            if choice.strip():
                idx = int(choice) - 1
                if 0 <= idx < len(trajectories):
                    view_trajectory(trajectories[idx])
        except (ValueError, KeyboardInterrupt):
            pass
        return

    # 直接查看指定文件
    if args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"文件不存在: {file_path}")
            return
        view_trajectory(file_path)
        return

    # 默认显示概览
    envs = list_environments(args.path)
    print(f"\n{'='*70}")
    print("轨迹概览:")
    print(f"{'='*70}\n")

    total = 0
    for env in envs:
        count = len(list_trajectories(env, args.path))
        total += count
        print(f"{env}: {count} 个轨迹文件")

    print(f"\n总计: {total} 个轨迹文件")
    print(f"\n使用方法:")
    print(f"  --list          列出所有文件")
    print(f"  --env <环境>    查看指定环境")
    print(f"  --file <路径>   查看指定文件")


if __name__ == "__main__":
    main()
