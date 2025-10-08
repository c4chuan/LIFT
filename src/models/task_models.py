"""
任务相关的数据模型

定义任务状态、任务数据结构等
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from vwa.src.envs.browser import FastCachedwActionMatchingBrowserEnv


class TaskState(Enum):
    """任务状态枚举"""
    IDLE = "idle"  # 空闲状态，等待被消费
    BUSY = "busy"  # 忙碌状态，正在执行环境交互
    PROCESSING = "processing"  # 处理状态，已被消费等待反馈
    COMPLETED = "completed"  # 已完成
    FAILED = "failed"  # 失败


@dataclass
class VWATask:
    """
    VWA任务数据类

    统一了监督学习和标准模式的任务结构
    通过ref_trajectory字段的有无来区分
    """
    task_id: int
    steps: int = 0
    trajectory: List[str] = field(default_factory=list)  # 存储summary的列表
    state_trajectory: List[Any] = field(default_factory=list)  # [state0, action0, state1, action1, ...]
    action_history: List[str] = field(default_factory=list)  # 动作历史（可读字符串）
    env: Optional[Any] = None  # FastCachedwActionMatchingBrowserEnv
    task_info: Dict[str, Any] = field(default_factory=dict)
    config_file: Dict[str, Any] = field(default_factory=dict)
    obs_info: Dict[str, Any] = field(default_factory=dict)

    # 监督学习特有字段
    ref_trajectory: Optional[List[Any]] = None  # 参考轨迹

    # 状态管理
    state: TaskState = TaskState.IDLE

    def __post_init__(self):
        """初始化后的验证"""
        if self.env is None:
            raise ValueError("env不能为None")

    def is_supervised(self) -> bool:
        """判断是否为监督学习任务"""
        return self.ref_trajectory is not None

    def get_current_action_index(self) -> int:
        """
        获取当前动作索引

        state_trajectory结构: [state_0, action_0, state_1, action_1, ...]
        最后一个元素如果是动作，则当前动作索引 = (len - 1) // 2
        """
        if len(self.state_trajectory) == 0:
            return 0
        # 如果最后一个是action（奇数位置）
        if len(self.state_trajectory) % 2 == 0:
            return (len(self.state_trajectory)) // 2 - 1
        # 如果最后一个是state（偶数位置）
        return (len(self.state_trajectory)) // 2

    def get_ref_action_at_index(self, index: int) -> Optional[Any]:
        """
        获取参考轨迹中指定索引的动作

        Args:
            index: 动作索引

        Returns:
            参考动作，如果索引越界或不是监督任务则返回None
        """
        if not self.is_supervised():
            return None

        ref_action_position = 2 * index + 1
        if ref_action_position >= len(self.ref_trajectory):
            return None

        return self.ref_trajectory[ref_action_position]

    def reset_for_new_task(self):
        """重置任务状态以便复用环境"""
        self.steps = 0
        self.trajectory.clear()
        self.state_trajectory.clear()
        self.action_history.clear()
        self.task_info.clear()
        self.obs_info.clear()
        self.state = TaskState.IDLE


@dataclass
class TaskMessage:
    """任务消息，用于在队列中传递"""
    task: VWATask
    message: List[Dict[str, Any]]  # LLM消息格式

    def __hash__(self):
        return hash(self.task.task_id)


class MessageQueueItem:
    """消息队列项"""
    def __init__(self, task: VWATask, message: List[Dict[str, Any]]):
        self.task = task
        self.message = message
        self.task_id = task.task_id
