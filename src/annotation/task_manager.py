"""
任务管理器模块
负责管理标注任务队列、进度跟踪和断点续标功能
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path


class TaskManager:
    """任务管理器类"""

    def __init__(self,
                 annotate_dir: str = "data/annotate",
                 progress_dir: str = "data/annotation_progress"):
        """
        初始化任务管理器

        Args:
            annotate_dir: 标注数据目录
            progress_dir: 进度数据目录
        """
        self.annotate_dir = Path(annotate_dir)
        self.progress_dir = Path(progress_dir)

        # 确保目录存在
        self.progress_dir.mkdir(parents=True, exist_ok=True)

        # 任务文件路径
        self.task_files = {
            "classifieds": self.annotate_dir / "classifieds_tasks.json",
            "reddit": self.annotate_dir / "reddit_tasks.json",
            "shopping": self.annotate_dir / "shopping_tasks.json"
        }

        # 进度文件路径
        self.progress_file = self.progress_dir / "progress.json"
        self.completed_file = self.progress_dir / "completed_tasks.json"

        # 加载数据
        self._load_tasks()
        self._load_progress()

    def _load_tasks(self) -> None:
        """加载所有任务文件"""
        self.tasks = {}
        self.task_count = {}

        for env_name, task_file in self.task_files.items():
            if task_file.exists():
                with open(task_file, 'r', encoding='utf-8') as f:
                    tasks = json.load(f)
                    self.tasks[env_name] = tasks
                    self.task_count[env_name] = len(tasks)
                    print(f"加载 {env_name} 环境任务: {len(tasks)} 个")
            else:
                self.tasks[env_name] = []
                self.task_count[env_name] = 0
                print(f"警告: 未找到 {env_name} 任务文件: {task_file}")

    def _load_progress(self) -> None:
        """加载进度数据"""
        # 加载整体进度
        if self.progress_file.exists():
            with open(self.progress_file, 'r', encoding='utf-8') as f:
                self.progress = json.load(f)
        else:
            self.progress = {
                "last_updated": None,
                "current_task_id": None,
                "current_environment": None,
                "completed_count": 0
            }

        # 加载已完成任务列表
        if self.completed_file.exists():
            with open(self.completed_file, 'r', encoding='utf-8') as f:
                self.completed_tasks = json.load(f)
        else:
            self.completed_tasks = []

        # 创建已完成任务的快速查找集合
        self.completed_task_ids = set()
        for task in self.completed_tasks:
            self.completed_task_ids.add(task["task_id"])

    def _save_progress(self) -> None:
        """保存进度数据"""
        self.progress["last_updated"] = datetime.now().isoformat()

        with open(self.progress_file, 'w', encoding='utf-8') as f:
            json.dump(self.progress, f, indent=2, ensure_ascii=False)

        with open(self.completed_file, 'w', encoding='utf-8') as f:
            json.dump(self.completed_tasks, f, indent=2, ensure_ascii=False)

    def get_next_task(self) -> Optional[Tuple[str, Dict[str, Any]]]:
        """
        获取下一个待标注的任务

        Returns:
            (环境名, 任务数据) 或 None 如果没有更多任务
        """
        # 如果有当前任务，继续当前任务
        if (self.progress["current_task_id"] and
            self.progress["current_environment"]):

            current_env = self.progress["current_environment"]
            current_id = self.progress["current_task_id"]

            # 查找当前任务
            for task in self.tasks[current_env]:
                task_id_str = f"{current_env}_{task['task_id']}"
                if task_id_str == current_id:
                    print(f"继续标注任务: {current_id}")
                    return current_env, task

        # 寻找下一个未完成的任务
        for env_name, tasks in self.tasks.items():
            for task in tasks:
                task_id_str = f"{env_name}_{task['task_id']}"
                if task_id_str not in self.completed_task_ids:
                    # 设置为当前任务
                    self.progress["current_task_id"] = task_id_str
                    self.progress["current_environment"] = env_name
                    self._save_progress()

                    print(f"开始新任务: {task_id_str}")
                    return env_name, task

        # 没有更多任务
        print("所有任务已完成！")
        return None

    def mark_task_completed(self, env_name: str, task: Dict[str, Any],
                          annotation_file: str) -> None:
        """
        标记任务为已完成

        Args:
            env_name: 环境名
            task: 任务数据
            annotation_file: 标注文件路径
        """
        task_id_str = f"{env_name}_{task['task_id']}"

        # 添加到已完成列表
        completed_task = {
            "task_id": task_id_str,
            "environment": env_name,
            "original_task_id": task['task_id'],
            "annotation_file": annotation_file,
            "completed_at": datetime.now().isoformat()
        }

        self.completed_tasks.append(completed_task)
        self.completed_task_ids.add(task_id_str)

        # 更新进度
        self.progress["completed_count"] += 1
        self.progress["current_task_id"] = None
        self.progress["current_environment"] = None

        self._save_progress()
        print(f"任务 {task_id_str} 标记为已完成")

    def skip_current_task(self) -> None:
        """跳过当前任务"""
        if self.progress["current_task_id"]:
            print(f"跳过任务: {self.progress['current_task_id']}")
            self.progress["current_task_id"] = None
            self.progress["current_environment"] = None
            self._save_progress()

    def reset_current_task(self) -> None:
        """重置当前任务（重新开始标注）"""
        if self.progress["current_task_id"]:
            print(f"重置任务: {self.progress['current_task_id']}")
            # 当前任务ID保持不变，只是重新开始标注

    def get_progress_summary(self) -> Dict[str, Any]:
        """获取进度摘要"""
        total_tasks = sum(self.task_count.values())
        completed_count = len(self.completed_tasks)
        remaining_count = total_tasks - completed_count

        # 按环境统计已完成任务
        env_completed = {}
        for env_name in self.tasks.keys():
            env_completed[env_name] = len([
                t for t in self.completed_tasks
                if t.get("environment", "") == env_name
            ])

        return {
            "总任务数": total_tasks,
            "已完成": completed_count,
            "剩余": remaining_count,
            "完成率": f"{completed_count/total_tasks*100:.1f}%" if total_tasks > 0 else "0%",
            "按环境统计": {
                env: {
                    "总数": self.task_count[env],
                    "已完成": env_completed.get(env, 0),
                    "剩余": self.task_count[env] - env_completed.get(env, 0)
                }
                for env in self.tasks.keys()
            },
            "当前任务": self.progress.get("current_task_id"),
            "最后更新": self.progress.get("last_updated")
        }

    def list_completed_tasks(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        列出最近完成的任务

        Args:
            limit: 显示数量限制

        Returns:
            最近完成的任务列表
        """
        return self.completed_tasks[-limit:] if self.completed_tasks else []