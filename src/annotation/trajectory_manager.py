"""
轨迹管理器模块
负责保存、加载和管理标注轨迹
"""

import json
import lzma
import pickle
import os
import sys
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path

# 添加路径以便导入visualwebarena模块
sys.path.append(str(Path(__file__).parent.parent.parent / "visualwebarena"))

try:
    from browser_env import Trajectory
except ImportError:
    # 如果无法导入browser_env，使用类型别名
    from typing import List, Any
    Trajectory = List[Any]


class TrajectoryManager:
    """轨迹管理器类"""

    def __init__(self, annotate_dir: str = "data/annotate"):
        """
        初始化轨迹管理器

        Args:
            annotate_dir: 标注数据目录
        """
        self.annotate_dir = Path(annotate_dir)
        self.annotate_dir.mkdir(parents=True, exist_ok=True)

        # 为每个环境创建子目录
        self.env_dirs = {
            "classifieds": self.annotate_dir / "trajectories" / "classifieds",
            "reddit": self.annotate_dir / "trajectories" / "reddit",
            "shopping": self.annotate_dir / "trajectories" / "shopping"
        }

        for env_dir in self.env_dirs.values():
            env_dir.mkdir(parents=True, exist_ok=True)

    def save_trajectory(self, env_name: str, task_id: int, trajectory: Trajectory,
                       task_info: Dict[str, Any], score: float) -> str:
        """
        保存标注轨迹

        Args:
            env_name: 环境名称
            task_id: 任务ID
            trajectory: 轨迹数据
            task_info: 任务信息
            score: 评估分数

        Returns:
            保存的文件路径
        """
        if env_name not in self.env_dirs:
            raise ValueError(f"不支持的环境: {env_name}")

        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{env_name}_{task_id}_{timestamp}"

        # 保存轨迹数据（二进制格式，兼容原有格式）
        trajectory_file = self.env_dirs[env_name] / f"{filename}.pkl.xz"
        with lzma.open(trajectory_file, "wb") as f:
            pickle.dump(trajectory, f)

        # 保存元数据（JSON格式，便于查看和分析）
        metadata = {
            "task_id": task_id,
            "environment": env_name,
            "intent": task_info.get("intent", ""),
            "start_url": task_info.get("start_url", ""),
            "require_login": task_info.get("require_login", False),
            "images": task_info.get("images", []),
            "score": score,
            "trajectory_length": len(trajectory),
            "annotation_timestamp": datetime.now().isoformat(),
            "trajectory_file": str(trajectory_file.relative_to(self.annotate_dir)),
        }

        metadata_file = self.env_dirs[env_name] / f"{filename}_metadata.json"
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        # 额外保存Python脚本格式（便于回放和理解）
        script_file = self.env_dirs[env_name] / f"{filename}_script.py"
        self._save_as_python_script(trajectory, task_info, script_file)

        print(f"✅ 轨迹已保存:")
        print(f"   二进制文件: {trajectory_file}")
        print(f"   元数据文件: {metadata_file}")
        print(f"   Python脚本: {script_file}")

        return str(trajectory_file)

    def _save_as_python_script(self, trajectory: Trajectory,
                              task_info: Dict[str, Any], script_file: Path) -> None:
        """
        将轨迹保存为Python脚本格式

        Args:
            trajectory: 轨迹数据
            task_info: 任务信息
            script_file: 脚本文件路径
        """
        try:
            script_content = self._generate_python_script(trajectory, task_info)
            with open(script_file, "w", encoding="utf-8") as f:
                f.write(script_content)
        except Exception as e:
            print(f"警告: 生成Python脚本失败: {e}")

    def _generate_python_script(self, trajectory: Trajectory,
                               task_info: Dict[str, Any]) -> str:
        """
        生成Python脚本内容

        Args:
            trajectory: 轨迹数据
            task_info: 任务信息

        Returns:
            Python脚本内容
        """
        lines = [
            '"""',
            f'标注轨迹回放脚本',
            f'任务ID: {task_info.get("task_id", "未知")}',
            f'环境: {task_info.get("environment", "未知")}',
            f'任务描述: {task_info.get("intent", "未知")}',
            f'生成时间: {datetime.now().isoformat()}',
            '"""',
            '',
            'import re',
            'from playwright.sync_api import Playwright, sync_playwright, expect',
            '',
            '',
            'def run(playwright: Playwright) -> None:',
            '    browser = playwright.chromium.launch(headless=False)',
        ]

        # 添加登录状态（如果有）
        if task_info.get("require_login"):
            storage_state = task_info.get("storage_state", "")
            if storage_state:
                lines.append(f'    context = browser.new_context(storage_state="{storage_state}")')
            else:
                lines.append('    context = browser.new_context()')
        else:
            lines.append('    context = browser.new_context()')

        lines.extend([
            '    page = context.new_page()',
            '',
        ])

        # 添加起始URL访问
        start_url = task_info.get("start_url")
        if start_url:
            lines.append(f'    page.goto("{start_url}")')

        # 解析轨迹中的动作
        action_count = 0
        for item in trajectory:
            if isinstance(item, dict) and "action_type" in item:
                # 这是一个动作
                try:
                    action_line = self._action_to_playwright_code(item)
                    if action_line:
                        lines.append(f'    {action_line}')
                        action_count += 1
                except Exception as e:
                    lines.append(f'    # 无法转换的动作: {item.get("raw_prediction", str(item))}')

        if action_count == 0:
            lines.append('    # 没有找到有效的动作')

        lines.extend([
            '    page.close()',
            '',
            '    # ---------------------',
            '    context.close()',
            '    browser.close()',
            '',
            '',
            'with sync_playwright() as playwright:',
            '    run(playwright)',
            ''
        ])

        return '\\n'.join(lines)

    def _action_to_playwright_code(self, action: Dict[str, Any]) -> Optional[str]:
        """
        将动作转换为Playwright代码

        Args:
            action: 动作数据

        Returns:
            Playwright代码行，如果无法转换则返回None
        """
        action_type = action.get("action_type")
        raw_prediction = action.get("raw_prediction", "")

        # 这里是一个简化的转换，实际情况可能需要更复杂的逻辑
        if "click" in raw_prediction.lower():
            # 尝试提取点击目标
            if "get_by_role" in raw_prediction:
                return f'page.get_by_role("link", name="...").click()  # {raw_prediction}'
            else:
                return f'page.click("...")  # {raw_prediction}'

        elif "type" in raw_prediction.lower() or "fill" in raw_prediction.lower():
            return f'page.fill("...", "...")  # {raw_prediction}'

        elif "goto" in raw_prediction.lower():
            return f'page.goto("...")  # {raw_prediction}'

        elif "scroll" in raw_prediction.lower():
            return f'page.mouse.wheel(0, 100)  # {raw_prediction}'

        else:
            return f'# {raw_prediction}'

    def load_trajectory(self, trajectory_file: str) -> Trajectory:
        """
        加载轨迹数据

        Args:
            trajectory_file: 轨迹文件路径

        Returns:
            轨迹数据
        """
        trajectory_path = Path(trajectory_file)
        if not trajectory_path.exists():
            raise FileNotFoundError(f"轨迹文件不存在: {trajectory_file}")

        if trajectory_path.suffix == '.xz':
            # 压缩的pickle文件
            with lzma.open(trajectory_path, "rb") as f:
                return pickle.load(f)
        elif trajectory_path.suffix == '.pkl':
            # 普通pickle文件
            with open(trajectory_path, "rb") as f:
                return pickle.load(f)
        else:
            raise ValueError(f"不支持的轨迹文件格式: {trajectory_file}")

    def list_trajectories(self, env_name: Optional[str] = None,
                         limit: int = 10) -> List[Dict[str, Any]]:
        """
        列出轨迹文件

        Args:
            env_name: 环境名称，如果为None则列出所有环境
            limit: 返回数量限制

        Returns:
            轨迹信息列表
        """
        trajectories = []

        search_dirs = [self.env_dirs[env_name]] if env_name else list(self.env_dirs.values())

        for env_dir in search_dirs:
            if not env_dir.exists():
                continue

            # 查找元数据文件
            for metadata_file in env_dir.glob("*_metadata.json"):
                try:
                    with open(metadata_file, "r", encoding="utf-8") as f:
                        metadata = json.load(f)
                    trajectories.append(metadata)
                except Exception as e:
                    print(f"警告: 读取元数据文件失败 {metadata_file}: {e}")

        # 按时间排序
        trajectories.sort(key=lambda x: x.get("annotation_timestamp", ""), reverse=True)

        return trajectories[:limit]

    def get_trajectory_stats(self) -> Dict[str, Any]:
        """
        获取轨迹统计信息

        Returns:
            统计信息字典
        """
        stats = {
            "total_trajectories": 0,
            "by_environment": {},
            "average_trajectory_length": 0,
            "average_score": 0,
        }

        total_length = 0
        total_score = 0.0
        count = 0

        for env_name, env_dir in self.env_dirs.items():
            env_count = 0
            env_total_length = 0
            env_total_score = 0.0

            if env_dir.exists():
                for metadata_file in env_dir.glob("*_metadata.json"):
                    try:
                        with open(metadata_file, "r", encoding="utf-8") as f:
                            metadata = json.load(f)

                        env_count += 1
                        length = metadata.get("trajectory_length", 0)
                        score = metadata.get("score", 0.0)

                        env_total_length += length
                        env_total_score += score

                        total_length += length
                        total_score += score
                        count += 1

                    except Exception:
                        pass

            stats["by_environment"][env_name] = {
                "count": env_count,
                "average_length": env_total_length / env_count if env_count > 0 else 0,
                "average_score": env_total_score / env_count if env_count > 0 else 0,
            }

        stats["total_trajectories"] = count
        stats["average_trajectory_length"] = total_length / count if count > 0 else 0
        stats["average_score"] = total_score / count if count > 0 else 0

        return stats

    def cleanup_old_trajectories(self, keep_days: int = 30) -> int:
        """
        清理旧的轨迹文件

        Args:
            keep_days: 保留天数

        Returns:
            删除的文件数量
        """
        from datetime import timedelta

        cutoff_date = datetime.now() - timedelta(days=keep_days)
        deleted_count = 0

        for env_dir in self.env_dirs.values():
            if not env_dir.exists():
                continue

            for metadata_file in env_dir.glob("*_metadata.json"):
                try:
                    with open(metadata_file, "r", encoding="utf-8") as f:
                        metadata = json.load(f)

                    timestamp_str = metadata.get("annotation_timestamp", "")
                    if timestamp_str:
                        timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                        if timestamp < cutoff_date:
                            # 删除相关文件
                            base_name = metadata_file.stem.replace("_metadata", "")

                            # 删除元数据文件
                            metadata_file.unlink(missing_ok=True)

                            # 删除轨迹文件
                            traj_file = env_dir / f"{base_name}.pkl.xz"
                            traj_file.unlink(missing_ok=True)

                            # 删除脚本文件
                            script_file = env_dir / f"{base_name}_script.py"
                            script_file.unlink(missing_ok=True)

                            deleted_count += 1

                except Exception as e:
                    print(f"警告: 清理文件时发生错误 {metadata_file}: {e}")

        return deleted_count