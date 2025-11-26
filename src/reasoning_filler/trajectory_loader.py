"""
轨迹读取模块

负责扫描、读取和解析标注轨迹数据
"""

import json
import lzma
import pickle
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class TrajectoryInfo:
    """轨迹信息"""
    trajectory_file: Path  # 轨迹文件路径
    metadata_file: Path    # 元数据文件路径
    env_name: str          # 环境名称
    task_id: int           # 任务ID


class TrajectoryLoader:
    """轨迹加载器"""

    def __init__(self, base_dir: str = "data/annotate/trajectories"):
        """
        初始化轨迹加载器

        Args:
            base_dir: 轨迹数据基础目录
        """
        self.base_dir = Path(base_dir)
        if not self.base_dir.exists():
            raise FileNotFoundError(f"轨迹目录不存在: {base_dir}")

    def scan_trajectories(self, env_filter: Optional[str] = None) -> List[TrajectoryInfo]:
        """
        扫描所有轨迹文件

        Args:
            env_filter: 环境名称过滤，如 "classifieds"，None 表示所有环境

        Returns:
            轨迹信息列表
        """
        trajectory_infos = []

        # 遍历环境目录
        for env_dir in self.base_dir.iterdir():
            if not env_dir.is_dir():
                continue

            env_name = env_dir.name

            # 环境过滤
            if env_filter and env_name != env_filter:
                continue

            # 扫描该环境下的轨迹文件
            for traj_file in env_dir.glob("*.pkl.xz"):
                # 跳过元数据文件
                if "_metadata" in traj_file.name or "_script" in traj_file.name:
                    continue

                # 查找对应的元数据文件
                base_name = traj_file.stem  # 移除 .pkl.xz
                if base_name.endswith('.pkl'):
                    base_name = base_name[:-4]  # 移除 .pkl

                metadata_file = env_dir / f"{base_name}_metadata.json"

                if not metadata_file.exists():
                    print(f"警告: 找不到元数据文件 {metadata_file}")
                    continue

                # 从文件名提取 task_id
                # 格式: {env_name}_{task_id}_{timestamp}.pkl.xz
                parts = base_name.split('_')
                try:
                    task_id = int(parts[1])
                except (IndexError, ValueError):
                    print(f"警告: 无法从文件名提取 task_id: {traj_file.name}")
                    task_id = -1

                trajectory_infos.append(TrajectoryInfo(
                    trajectory_file=traj_file,
                    metadata_file=metadata_file,
                    env_name=env_name,
                    task_id=task_id
                ))

        # 按环境和 task_id 排序
        trajectory_infos.sort(key=lambda x: (x.env_name, x.task_id))

        return trajectory_infos

    def load_trajectory(self, traj_info: TrajectoryInfo) -> Tuple[List[Any], Dict[str, Any]]:
        """
        加载轨迹数据和元数据

        Args:
            traj_info: 轨迹信息

        Returns:
            (trajectory, metadata) 元组
            - trajectory: 轨迹数据 (Trajectory 类型，即 list[StateInfo | Action])
            - metadata: 元数据字典
        """
        # 读取轨迹数据
        trajectory = self._read_pkl_xz(traj_info.trajectory_file)

        # 读取元数据
        with open(traj_info.metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        return trajectory, metadata

    @staticmethod
    def _read_pkl_xz(file_path: Path) -> Any:
        """
        读取 .pkl.xz 文件

        Args:
            file_path: 文件路径

        Returns:
            解压缩和反序列化后的 Python 对象
        """
        try:
            with lzma.open(file_path, 'rb') as f:
                data = pickle.load(f)
            return data
        except Exception as e:
            raise Exception(f"读取文件失败 {file_path}: {e}")

    def get_statistics(self) -> Dict[str, Any]:
        """
        获取轨迹统计信息

        Returns:
            统计信息字典
        """
        all_trajectories = self.scan_trajectories()

        stats = {
            "total": len(all_trajectories),
            "by_environment": {}
        }

        for traj_info in all_trajectories:
            env_name = traj_info.env_name
            if env_name not in stats["by_environment"]:
                stats["by_environment"][env_name] = 0
            stats["by_environment"][env_name] += 1

        return stats


def main():
    """测试函数"""
    loader = TrajectoryLoader()

    print("=== 轨迹统计 ===")
    stats = loader.get_statistics()
    print(f"总数: {stats['total']}")
    print(f"各环境分布:")
    for env, count in stats['by_environment'].items():
        print(f"  {env}: {count}")

    print("\n=== 扫描轨迹 ===")
    trajectories = loader.scan_trajectories()
    print(f"找到 {len(trajectories)} 个轨迹文件")

    if trajectories:
        print("\n=== 测试加载第一个轨迹 ===")
        first_traj = trajectories[0]
        print(f"环境: {first_traj.env_name}")
        print(f"任务ID: {first_traj.task_id}")
        print(f"轨迹文件: {first_traj.trajectory_file}")

        trajectory, metadata = loader.load_trajectory(first_traj)
        print(f"\n轨迹长度: {len(trajectory)}")
        print(f"元数据:")
        print(f"  intent: {metadata.get('intent', 'N/A')}")
        print(f"  score: {metadata.get('score', 'N/A')}")
        print(f"  trajectory_length: {metadata.get('trajectory_length', 'N/A')}")


if __name__ == "__main__":
    main()
