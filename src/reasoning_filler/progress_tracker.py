"""
进度跟踪模块

负责记录处理进度，支持断点重续
"""

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Set, Optional


class ProgressTracker:
    """进度跟踪器"""

    def __init__(self, progress_file: str = "data/annotate_with_reasoning/progress.json"):
        """
        初始化进度跟踪器

        Args:
            progress_file: 进度文件路径
        """
        self.progress_file = Path(progress_file)
        self.progress_data = self._load_progress()
        self._lock = threading.Lock()  # 线程安全锁

    def _load_progress(self) -> Dict[str, Any]:
        """
        加载进度数据

        Returns:
            进度数据字典
        """
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"警告: 加载进度文件失败: {e}，将创建新的进度记录")

        # 默认进度数据
        return {
            "last_updated": None,
            "processed_files": [],
            "failed_files": {},
            "statistics": {
                "total_processed": 0,
                "total_failed": 0,
                "total_actions_filled": 0,
                "validation_passed": 0,
                "validation_failed": 0,
                "passed_first_attempt": 0,
                "corrected_first_retry": 0,
                "corrected_second_retry": 0,
                "total_api_calls": 0
            }
        }

    def save_progress(self):
        """保存进度数据到文件"""
        with self._lock:
            try:
                # 更新时间戳
                self.progress_data["last_updated"] = datetime.now().isoformat()

                # 确保目录存在
                self.progress_file.parent.mkdir(parents=True, exist_ok=True)

                # 保存到文件
                with open(self.progress_file, 'w', encoding='utf-8') as f:
                    json.dump(self.progress_data, f, indent=2, ensure_ascii=False)

            except Exception as e:
                print(f"警告: 保存进度文件失败: {e}")

    def is_processed(self, file_path: str) -> bool:
        """
        检查文件是否已处理

        Args:
            file_path: 文件路径（相对路径）

        Returns:
            是否已处理
        """
        # 转换为相对路径字符串
        file_str = str(Path(file_path))
        return file_str in self.progress_data["processed_files"]

    def is_failed(self, file_path: str) -> bool:
        """
        检查文件是否处理失败

        Args:
            file_path: 文件路径（相对路径）

        Returns:
            是否处理失败
        """
        file_str = str(Path(file_path))
        return file_str in self.progress_data["failed_files"]

    def mark_processed(self, file_path: str, num_actions: int = 0):
        """
        标记文件为已处理

        Args:
            file_path: 文件路径（相对路径）
            num_actions: 处理的动作数量
        """
        with self._lock:
            file_str = str(Path(file_path))

            # 添加到已处理列表
            if file_str not in self.progress_data["processed_files"]:
                self.progress_data["processed_files"].append(file_str)

            # 从失败列表中移除（如果存在）
            if file_str in self.progress_data["failed_files"]:
                del self.progress_data["failed_files"][file_str]

            # 更新统计
            self.progress_data["statistics"]["total_processed"] += 1
            self.progress_data["statistics"]["total_actions_filled"] += num_actions

    def mark_failed(self, file_path: str, error: str):
        """
        标记文件为处理失败

        Args:
            file_path: 文件路径（相对路径）
            error: 错误信息
        """
        with self._lock:
            file_str = str(Path(file_path))

            # 添加到失败字典
            self.progress_data["failed_files"][file_str] = {
                "error": error,
                "timestamp": datetime.now().isoformat()
            }

            # 从已处理列表中移除（如果存在）
            if file_str in self.progress_data["processed_files"]:
                self.progress_data["processed_files"].remove(file_str)

            # 更新统计
            self.progress_data["statistics"]["total_failed"] += 1

    def get_processed_files(self) -> Set[str]:
        """
        获取已处理的文件集合

        Returns:
            已处理文件路径集合
        """
        return set(self.progress_data["processed_files"])

    def get_failed_files(self) -> Dict[str, Any]:
        """
        获取处理失败的文件字典

        Returns:
            失败文件字典 {文件路径: {error, timestamp}}
        """
        return self.progress_data["failed_files"].copy()

    def get_statistics(self) -> Dict[str, int]:
        """
        获取统计信息

        Returns:
            统计信息字典
        """
        return self.progress_data["statistics"].copy()

    def update_validation_stats(self, validation_stats: Dict[str, int]):
        """
        更新验证统计信息

        Args:
            validation_stats: 验证统计字典，包含以下字段：
                - validation_passed: 验证通过数
                - validation_failed: 验证失败数
                - passed_first_attempt: 第1次就通过数
                - corrected_first_retry: 第1次重试成功数
                - corrected_second_retry: 第2次重试成功数
                - total_api_calls: 总API调用次数
        """
        with self._lock:
            stats = self.progress_data["statistics"]

            # 累加验证统计
            stats["validation_passed"] = stats.get("validation_passed", 0) + validation_stats.get("validation_passed", 0)
            stats["validation_failed"] = stats.get("validation_failed", 0) + validation_stats.get("validation_failed", 0)
            stats["passed_first_attempt"] = stats.get("passed_first_attempt", 0) + validation_stats.get("passed_first_attempt", 0)
            stats["corrected_first_retry"] = stats.get("corrected_first_retry", 0) + validation_stats.get("corrected_first_retry", 0)
            stats["corrected_second_retry"] = stats.get("corrected_second_retry", 0) + validation_stats.get("corrected_second_retry", 0)
            stats["total_api_calls"] = stats.get("total_api_calls", 0) + validation_stats.get("total_api_calls", 0)

    def reset(self):
        """重置进度记录"""
        with self._lock:
            self.progress_data = {
                "last_updated": None,
                "processed_files": [],
                "failed_files": {},
                "statistics": {
                    "total_processed": 0,
                    "total_failed": 0,
                    "total_actions_filled": 0,
                    "validation_passed": 0,
                    "validation_failed": 0,
                    "passed_first_attempt": 0,
                    "corrected_first_retry": 0,
                    "corrected_second_retry": 0,
                    "total_api_calls": 0
                }
            }
        self.save_progress()

    def print_summary(self):
        """打印进度摘要"""
        stats = self.progress_data["statistics"]
        last_updated = self.progress_data.get("last_updated", "从未")

        print("\n" + "=" * 60)
        print("进度摘要")
        print("=" * 60)
        print(f"最后更新时间: {last_updated}")
        print(f"已处理文件数: {stats['total_processed']}")
        print(f"失败文件数: {stats['total_failed']}")
        print(f"已填充动作数: {stats['total_actions_filled']}")
        print(f"待处理文件数: {len(self.progress_data['processed_files'])}")

        # 显示验证统计（如果有）
        if stats.get('validation_passed', 0) > 0 or stats.get('validation_failed', 0) > 0:
            print("\n验证统计:")
            total_validated = stats.get('validation_passed', 0) + stats.get('validation_failed', 0)
            pass_rate = stats.get('validation_passed', 0) / total_validated * 100 if total_validated > 0 else 0

            print(f"  验证通过: {stats.get('validation_passed', 0)}")
            print(f"  验证失败: {stats.get('validation_failed', 0)}")
            print(f"  通过率: {pass_rate:.1f}%")

            # 详细的尝试次数分布
            if stats.get('passed_first_attempt', 0) > 0:
                print(f"\n  成功分布:")
                print(f"    第1次成功: {stats.get('passed_first_attempt', 0)}")
                print(f"    第2次成功（第1次重试）: {stats.get('corrected_first_retry', 0)}")
                print(f"    第3次成功（第2次重试）: {stats.get('corrected_second_retry', 0)}")

            # API调用统计
            if stats.get('total_api_calls', 0) > 0:
                avg_calls_per_action = stats.get('total_api_calls', 0) / stats.get('total_actions_filled', 1)
                print(f"\n  API调用统计:")
                print(f"    总调用次数: {stats.get('total_api_calls', 0)}")
                print(f"    平均每action调用次数: {avg_calls_per_action:.2f}")

        if self.progress_data["failed_files"]:
            print(f"\n失败文件列表:")
            for file_path, info in self.progress_data["failed_files"].items():
                print(f"  - {file_path}")
                print(f"    错误: {info['error']}")
                print(f"    时间: {info['timestamp']}")

        print("=" * 60 + "\n")


def main():
    """测试函数"""
    tracker = ProgressTracker("test_progress.json")

    print("=== 测试进度跟踪 ===")

    # 测试标记处理
    tracker.mark_processed("classifieds/task_1.pkl.xz", num_actions=5)
    tracker.mark_processed("reddit/task_2.pkl.xz", num_actions=3)

    # 测试标记失败
    tracker.mark_failed("shopping/task_3.pkl.xz", "API 调用超时")

    # 保存进度
    tracker.save_progress()

    # 测试查询
    print(f"\ntask_1.pkl.xz 是否已处理: {tracker.is_processed('classifieds/task_1.pkl.xz')}")
    print(f"task_3.pkl.xz 是否失败: {tracker.is_failed('shopping/task_3.pkl.xz')}")

    # 打印摘要
    tracker.print_summary()

    # 清理测试文件
    import os
    if os.path.exists("test_progress.json"):
        os.remove("test_progress.json")
        print("测试文件已清理")


if __name__ == "__main__":
    main()
