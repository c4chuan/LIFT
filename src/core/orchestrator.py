"""
环境编排器（核心协调器）

负责协调所有组件，实现环境管理的核心业务逻辑
"""
import asyncio
from typing import List, Dict, Any, Optional
from PIL import Image
from pydantic import BaseModel

from src.config.environment_config import EnvironmentConfig
from src.core.task_pool import TaskPool
from src.core.message_builder import IMessageBuilder, ActionDescriptionHelper
from src.core.reward_calculator import RewardCalculator
from src.core.action_strategy import IActionStrategy
from src.models.task_models import VWATask, TaskState
from src.env.envtools import parallel_prepare, refresh_env_login, reset_env
from src.utils.scp_tools import parallel_scp_to_remote, parallel_scp_to_remote_cmd_version
from vwa.src.envs.actions import create_none_action, create_id_based_action, ActionTypes


class ResponseWithReward(BaseModel):
    response: str
    reward_sum: float


class EnvironmentOrchestrator:
    """
    环境编排器

    职责：
    1. 协调TaskPool、MessageBuilder、RewardCalculator等组件
    2. 实现环境交互的核心流程
    3. 管理任务生命周期
    4. 处理并发和错误
    """

    def __init__(
        self,
        config: EnvironmentConfig,
        task_pool: TaskPool,
        message_builder: IMessageBuilder,
        reward_calculator: RewardCalculator,
        action_strategy: IActionStrategy
    ):
        """
        初始化环境编排器

        Args:
            config: 环境配置
            task_pool: 任务池
            message_builder: 消息构建器
            reward_calculator: 奖励计算器
            action_strategy: 动作策略
        """
        self.config = config
        self.task_pool = task_pool
        self.message_builder = message_builder
        self.reward_calculator = reward_calculator
        self.action_strategy = action_strategy

        # 用于并发控制的锁
        self._production_lock = asyncio.Lock()

    async def initialize_environments(self, num_envs: int):
        """
        初始化指定数量的环境实例

        Args:
            num_envs: 环境数量
        """
        print(f"正在初始化{num_envs}个环境...")

        tasks = []
        for i in range(num_envs):
            task = await self.task_pool.create_task()
            if task:
                tasks.append(task)

        if len(tasks) > 0:
            # 并行生产初始消息
            actions = [create_none_action() for _ in tasks]
            await self.parallel_produce(tasks, actions)

        print(f"环境初始化完成，创建了{len(tasks)}个环境")

    async def parallel_produce(
        self,
        tasks: List[VWATask],
        actions: List[Any]
    ):
        """
        并行生产消息

        这是核心方法，负责：
        1. 并行执行环境交互（reset或step）
        2. 构建消息
        3. 更新任务状态
        4. 将消息加入队列

        Args:
            tasks: 任务列表
            actions: 对应的动作列表
        """
        async with self._production_lock:
            # 1. 判断每个任务是否需要重置
            task_flags = self._get_task_reset_flags(tasks, actions)

            # 2. 准备需要reset的任务配置
            config_files_to_prepare = []
            for task, action, needs_reset in zip(tasks, actions, task_flags):
                if needs_reset:
                    config_files_to_prepare.append(task.config_file)

            # 3. 并行准备任务信息
            task_infos = []
            if len(config_files_to_prepare) > 0:
                task_infos = parallel_prepare(
                    self.config.cache_dir,
                    config_files_to_prepare,
                    max_workers=self.config.max_workers
                )

            # 4. 构建协程列表
            coros = []
            info_index = 0

            for task, action, needs_reset in zip(tasks, actions, task_flags):
                if needs_reset:
                    # 需要reset
                    task.task_info = task_infos[info_index]
                    coros.append(
                        task.env.areset(
                            options={"config_file": task_infos[info_index]['config_file']}
                        )
                    )
                    info_index += 1
                else:
                    # 执行step前，先决定使用哪个动作
                    decided_action = self.action_strategy.decide_action(task, action)

                    # 将动作添加到state_trajectory
                    task.state_trajectory.append(decided_action)

                    # 执行step
                    coros.append(task.env.astep(decided_action))

            # 5. 并行执行所有协程
            try:
                results = await asyncio.gather(*coros, return_exceptions=True)
            except Exception as e:
                print(f"并行执行环境交互时出错: {e}")
                return

            # 6. 处理结果并构建消息
            messages = []
            task_ids = []

            result_index = 0
            for task, action, needs_reset in zip(tasks, actions, task_flags):
                result = results[result_index]
                result_index += 1

                # 检查是否有异常
                if isinstance(result, Exception):
                    print(f"任务{task.task_id}执行出错: {result}")
                    continue

                # 根据是否reset处理不同的结果
                if needs_reset:
                    obs, info = result
                    state_info = {
                        "observation": obs,
                        "info": info,
                        "url": task.env.page.url
                    }
                    task.state_trajectory.append(state_info)

                    # 构建消息
                    message = self.message_builder.construct_message(task, obs, info)
                    task.obs_info = info

                else:
                    obs, action_reward, _, _, info = result

                    # 处理obs为None的情况
                    if obs.get('image') is None:
                        obs = task.state_trajectory[-1]['observation']
                        info = task.state_trajectory[-1]['info']

                    state_info = {
                        "observation": obs,
                        "info": info,
                        "url": task.env.page.url
                    }
                    task.state_trajectory.append(state_info)

                    # 获取动作描述
                    action_str = ActionDescriptionHelper.get_action_description(
                        action,
                        state_info["info"]["observation_metadata"],
                        action_set_tag=self.config.action_set_tag,
                        prompt_constructor=self.message_builder.prompt_constructor
                    )
                    task.action_history.append(action_str)

                    # 构建消息
                    message = self.message_builder.construct_message(task, obs, info)
                    task.obs_info = info

                messages.append(message)
                task_ids.append(task.task_id)

                # 更新任务状态为IDLE
                await self.task_pool.update_task_state(task.task_id, TaskState.IDLE)

                print(f"Task:{task.task_id}-Action:{action.action_type if hasattr(action, 'action_type') else 'NONE'}step完毕-加入消息队列")

            # 7. 发送图片到服务器（如果需要）
            if self.config.type == "remote":
                self._send_images_to_server(task_ids, messages)

            # 8. 将消息加入队列
            for task, message in zip(tasks, messages):
                if message:  # 确保消息有效
                    await self.task_pool.enqueue_message(task, message)

            # 9. 打印状态信息
            self._print_status()

    def _get_task_reset_flags(
        self,
        tasks: List[VWATask],
        actions: List[Any]
    ) -> List[bool]:
        """
        判断每个任务是否需要reset

        Args:
            tasks: 任务列表
            actions: 动作列表

        Returns:
            布尔值列表，True表示需要reset
        """
        flags = []
        for task, action in zip(tasks, actions):
            needs_reset = (
                action.action_type == ActionTypes.NONE or
                action.action_type == ActionTypes.STOP or
                task.steps >= self.config.max_task_steps
            )
            flags.append(needs_reset)
        return flags

    async def feed_responses(self, responses: List[ResponseWithReward]):
        """
        处理来自模型的响应

        Args:
            responses: 响应列表
        """
        # 等待所有busy任务完成
        while len(self.task_pool.get_tasks_by_state(TaskState.BUSY)) > 0:
            await asyncio.sleep(0.1)

        # 提取动作
        actions = []
        for response in responses:
            action_str = self.message_builder.extract_action(response.response)
            if action_str is None or action_str == "None":
                action = create_none_action()
                action.obs_reward = response.reward_sum
            else:
                action = create_id_based_action(action_str)
                action.obs_reward = response.reward_sum

            action.update({"raw_prediction": response})
            actions.append(action)

        # 获取PROCESSING状态的任务
        processing_tasks = self.task_pool.get_tasks_by_state(TaskState.PROCESSING)

        if len(processing_tasks) != len(responses):
            print(f"警告: 响应数量({len(responses)})与处理中任务数量({len(processing_tasks)})不匹配")
            return

        # 更新任务信息
        tasks_to_produce = []
        for task, response, action in zip(processing_tasks, responses, actions):
            task.steps += 1

            # 提取summary
            summary = self.message_builder.extract_summary(response.response)
            task.trajectory.append(summary)

            # 更新状态为BUSY
            await self.task_pool.update_task_state(task.task_id, TaskState.BUSY)

            tasks_to_produce.append(task)

        # 触发异步生产（不等待）
        asyncio.create_task(self.parallel_produce(tasks_to_produce, actions))

    def get_messages(self, num: int) -> List[Dict[str, Any]] | str:
        """
        获取消息

        Args:
            num: 消息数量

        Returns:
            消息列表，或"Finished"表示任务已完成
        """
        queue_length = self.task_pool.get_message_queue_length()
        busy_count = len(self.task_pool.get_tasks_by_state(TaskState.BUSY))

        if queue_length < num:
            if busy_count > 0:
                print(f"消息队列中的有效消息不足({queue_length}/{num})，正在生产中")
                return []
            elif self.task_pool._task_pointer >= self.task_pool.total_task_count:
                print(f"任务队列已用完")
                return "Finished"
            else:
                print(f"未知情况: queue_length={queue_length}, busy_count={busy_count}")
                return []

        # 同步方式取出消息
        messages_items = asyncio.run(
            self.task_pool.dequeue_messages(min(num, queue_length))
        )

        messages = [item.message for item in messages_items]

        print(f"取出{len(messages)}条消息")
        self._print_status()

        return messages

    def get_valid_action_rewards(self, responses: List[str]) -> List[float]:
        """
        计算动作奖励

        Args:
            responses: 响应列表

        Returns:
            奖励列表
        """
        processing_tasks = self.task_pool.get_tasks_by_state(TaskState.PROCESSING)

        if len(processing_tasks) == 0:
            print("警告: 没有处理中的任务")
            return []

        # 计算每个任务的batch size
        batch_size = len(responses) // len(processing_tasks)

        rewards = []
        for i, task in enumerate(processing_tasks):
            task_responses = responses[i * batch_size:(i + 1) * batch_size]
            task_rewards = self.reward_calculator.calculate_batch_rewards(
                task_responses,
                [task] * len(task_responses)
            )
            rewards.extend(task_rewards)

        return rewards

    def refresh_env(self):
        """刷新环境"""
        refresh_env_login()
        reset_env(self.config.env_name)

    def _send_images_to_server(
        self,
        task_ids: List[int],
        messages: List[List[Dict[str, Any]]]
    ):
        """
        发送图片到服务器

        Args:
            task_ids: 任务ID列表
            messages: 消息列表
        """
        image_paths = self._collect_image_paths(messages)

        if self.config.scp_version == "client":
            parallel_scp_to_remote(task_ids, image_paths)
        elif self.config.scp_version == "cmd":
            parallel_scp_to_remote_cmd_version(task_ids, image_paths)

    def _collect_image_paths(
        self,
        task_messages: List[List[Dict[str, Any]]]
    ) -> List[List[str]]:
        """
        从消息中收集图片路径

        Args:
            task_messages: 任务消息列表

        Returns:
            图片路径列表
        """
        image_paths = []
        for task_msg in task_messages:
            msg_image_paths = []
            for msg in task_msg:
                if msg.get('role') == 'user':
                    for content in msg.get('content', []):
                        if 'image' in content:
                            msg_image_paths.append(content['image'])
            image_paths.append(msg_image_paths)
        return image_paths

    def _print_status(self):
        """打印当前状态"""
        state_counts = self.task_pool.get_state_counts()
        queue_length = self.task_pool.get_message_queue_length()

        print(f"状态统计: {state_counts}, 消息队列: {queue_length}")
