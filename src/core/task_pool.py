"""
任务池管理模块

提供高效的任务状态管理和查询接口
使用字典实现O(1)查找，替代原有的列表查找
"""
import asyncio
from typing import Dict, List, Optional, Set
from collections import defaultdict

from src.models.task_models import VWATask, TaskState, MessageQueueItem
from src.config.environment_config import EnvironmentConfig
from vwa.src.envs.browser import FastCachedwActionMatchingBrowserEnv


class TaskPool:
    """
    任务池管理器

    职责：
    1. 管理所有任务实例的创建和生命周期
    2. 维护任务状态索引，提供O(1)查询
    3. 提供线程安全的状态转换接口
    """

    def __init__(self, config: EnvironmentConfig, task_configs: List[Dict]):
        """
        初始化任务池

        Args:
            config: 环境配置
            task_configs: 任务配置列表
        """
        self.config = config
        self.task_configs = task_configs

        # 所有任务实例的字典 {task_id: VWATask}
        self._tasks: Dict[int, VWATask] = {}

        # 状态索引 {state: set of task_ids}
        self._state_index: Dict[TaskState, Set[int]] = defaultdict(set)

        # 消息队列
        self._message_queue: List[MessageQueueItem] = []

        # 任务指针，用于循环分配任务
        self._task_pointer: int = 0

        # 用于并发控制的锁
        self._lock = asyncio.Lock()

        # 用于通知消息队列变化的事件
        self._message_available = asyncio.Event()

    @property
    def total_task_count(self) -> int:
        """总任务配置数量"""
        return len(self.task_configs)

    @property
    def active_task_count(self) -> int:
        """活跃任务实例数量"""
        return len(self._tasks)

    def get_task_by_id(self, task_id: int) -> Optional[VWATask]:
        """
        根据ID获取任务

        Args:
            task_id: 任务ID

        Returns:
            任务实例，如果不存在返回None
        """
        return self._tasks.get(task_id)

    def get_tasks_by_state(self, state: TaskState) -> List[VWATask]:
        """
        获取指定状态的所有任务

        Args:
            state: 任务状态

        Returns:
            任务列表
        """
        task_ids = self._state_index.get(state, set())
        return [self._tasks[tid] for tid in task_ids if tid in self._tasks]

    async def create_task(self) -> Optional[VWATask]:
        """
        创建新任务实例

        从task_configs中按照task_pointer循环创建任务
        自动创建环境实例

        Returns:
            新创建的任务，如果无可用配置返回None
        """
        async with self._lock:
            if self._task_pointer >= len(self.task_configs):
                return None

            # 获取任务配置
            task_config = self.task_configs[self._task_pointer]

            # 提取ref_trajectory（如果存在）
            ref_trajectory = task_config.get('ref_trajectory', None)
            config_without_ref = {k: v for k, v in task_config.items() if k != 'ref_trajectory'}

            # 创建环境实例
            env = FastCachedwActionMatchingBrowserEnv(
                headless=self.config.headless,
                slow_mo=self.config.slow_mo,
                action_set_tag=self.config.action_set_tag,
                observation_type=self.config.observation_type,
                current_viewport_only=self.config.current_viewport_only,
                viewport_size=self.config.get_viewport_size(),
                save_trace_enabled=self.config.save_trace_enabled,
                sleep_after_execution=self.config.sleep_after_execution,
            )

            # 创建任务实例
            task = VWATask(
                task_id=task_config['task_id'],
                steps=0,
                env=env,
                config_file=config_without_ref,
                ref_trajectory=ref_trajectory,
                state=TaskState.IDLE
            )

            # 添加到管理池
            self._tasks[task.task_id] = task
            self._state_index[TaskState.IDLE].add(task.task_id)

            # 更新指针
            self._task_pointer = (self._task_pointer + 1) % min(
                len(self.task_configs),
                self.config.task_pointer_limit
            )

            return task

    async def update_task_state(
        self,
        task_id: int,
        new_state: TaskState
    ) -> bool:
        """
        更新任务状态

        线程安全的状态转换

        Args:
            task_id: 任务ID
            new_state: 新状态

        Returns:
            是否更新成功
        """
        async with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return False

            old_state = task.state

            # 从旧状态索引中移除
            if task_id in self._state_index[old_state]:
                self._state_index[old_state].remove(task_id)

            # 更新状态
            task.state = new_state

            # 添加到新状态索引
            self._state_index[new_state].add(task_id)

            return True

    async def enqueue_message(
        self,
        task: VWATask,
        message: List[Dict]
    ):
        """
        将任务消息加入队列

        Args:
            task: 任务实例
            message: 消息内容
        """
        async with self._lock:
            queue_item = MessageQueueItem(task, message)
            self._message_queue.append(queue_item)

            # 通知有新消息
            self._message_available.set()

    async def dequeue_messages(
        self,
        num: int,
        mark_as_processing: bool = True
    ) -> List[MessageQueueItem]:
        """
        从队列中取出消息

        Args:
            num: 取出数量
            mark_as_processing: 是否将对应任务标记为PROCESSING状态

        Returns:
            消息列表
        """
        async with self._lock:
            # 取出指定数量的消息
            messages = self._message_queue[:num]
            self._message_queue = self._message_queue[num:]

            # 如果需要，标记任务为处理中
            if mark_as_processing:
                for msg_item in messages:
                    await self.update_task_state(
                        msg_item.task_id,
                        TaskState.PROCESSING
                    )

            # 如果队列为空，清除事件
            if len(self._message_queue) == 0:
                self._message_available.clear()

            return messages

    async def wait_for_messages(self, timeout: Optional[float] = None) -> bool:
        """
        等待消息队列中有消息

        Args:
            timeout: 超时时间（秒），None表示无限等待

        Returns:
            是否有消息可用
        """
        try:
            await asyncio.wait_for(
                self._message_available.wait(),
                timeout=timeout
            )
            return True
        except asyncio.TimeoutError:
            return False

    def get_message_queue_length(self) -> int:
        """获取消息队列长度"""
        return len(self._message_queue)

    def get_state_counts(self) -> Dict[str, int]:
        """
        获取各状态的任务数量

        Returns:
            状态计数字典
        """
        return {
            state.value: len(task_ids)
            for state, task_ids in self._state_index.items()
        }

    async def cleanup_task(self, task_id: int):
        """
        清理任务资源

        Args:
            task_id: 任务ID
        """
        async with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return

            # 关闭环境
            if task.env:
                try:
                    await task.env.aclose()
                except Exception as e:
                    print(f"关闭任务{task_id}的环境时出错: {e}")

            # 从索引中移除
            if task_id in self._state_index[task.state]:
                self._state_index[task.state].remove(task_id)

            # 从任务字典中移除
            del self._tasks[task_id]

    def __str__(self) -> str:
        """字符串表示"""
        state_counts = self.get_state_counts()
        return (
            f"TaskPool("
            f"total_configs={self.total_task_count}, "
            f"active_tasks={self.active_task_count}, "
            f"states={state_counts}, "
            f"queue_length={self.get_message_queue_length()})"
        )
