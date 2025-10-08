"""
环境管理API路由模块

提供FastAPI端点，将API层和业务逻辑分离
"""
import asyncio
from typing import List
from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

from src.core.orchestrator import EnvironmentOrchestrator


class ResponseList(BaseModel):
    """响应列表模型"""
    responses: List[str]


def create_environment_router(orchestrator: EnvironmentOrchestrator) -> APIRouter:
    """
    创建环境管理路由

    Args:
        orchestrator: 环境编排器实例

    Returns:
        FastAPI路由器
    """
    router = APIRouter(prefix="", tags=["environment"])

    @router.get('/get_length')
    async def api_get_length():
        """获取任务总数"""
        return {"length": orchestrator.task_pool.total_task_count}

    @router.get('/get_messages')
    async def api_get_messages(num: int):
        """
        获取指定数量的消息

        Args:
            num: 请求的消息数量

        Returns:
            消息列表
        """
        # 等待直到有足够的消息或没有busy任务
        while orchestrator.task_pool.get_tasks_by_state(TaskState.BUSY):
            await asyncio.sleep(0.1)

        msgs = orchestrator.get_messages(num)
        return {'messages': msgs}

    @router.get('/get_val_messages')
    async def api_get_val_messages(num: int):
        """
        获取验证消息（不改变任务状态）

        Args:
            num: 请求的消息数量

        Returns:
            消息列表
        """
        # 等待直到有足够的消息
        queue_length = orchestrator.task_pool.get_message_queue_length()
        busy_count = len(orchestrator.task_pool.get_tasks_by_state(TaskState.BUSY))

        while queue_length < num and busy_count > 0:
            await asyncio.sleep(0.1)
            queue_length = orchestrator.task_pool.get_message_queue_length()
            busy_count = len(orchestrator.task_pool.get_tasks_by_state(TaskState.BUSY))

        # 不标记为processing，仅查看
        messages_items = await orchestrator.task_pool.dequeue_messages(
            min(num, queue_length),
            mark_as_processing=False
        )

        messages = [item.message for item in messages_items]
        return {'messages': messages}

    @router.post('/feed_responses')
    async def api_feed_responses(
        responses: List[str],
        background_tasks: BackgroundTasks
    ):
        """
        接收模型响应并触发后台处理

        Args:
            responses: 响应列表
            background_tasks: FastAPI后台任务

        Returns:
            接收状态
        """
        background_tasks.add_task(orchestrator.feed_responses, responses)
        return {'status': 'accepted'}

    @router.get('/refresh_env')
    async def api_refresh_env():
        """刷新环境"""
        try:
            orchestrator.refresh_env()
            return {'status': 'refreshed'}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"刷新环境失败: {str(e)}")

    @router.get('/get_is_last_batch')
    async def api_get_is_last_batch():
        """检查是否是最后一批任务"""
        is_last = orchestrator.task_pool._task_pointer >= orchestrator.task_pool.total_task_count
        return {'flag': is_last}

    @router.post('/get_valid_action_rewards')
    async def api_get_valid_action_rewards(responselist: ResponseList):
        """
        计算动作奖励

        Args:
            responselist: 响应列表

        Returns:
            奖励列表
        """
        try:
            rewards = orchestrator.get_valid_action_rewards(responselist.responses)
            return {'rewards': rewards}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"计算奖励失败: {str(e)}")

    @router.get('/status')
    async def api_get_status():
        """获取系统状态"""
        state_counts = orchestrator.task_pool.get_state_counts()
        queue_length = orchestrator.task_pool.get_message_queue_length()

        return {
            'task_pool': {
                'total_configs': orchestrator.task_pool.total_task_count,
                'active_tasks': orchestrator.task_pool.active_task_count,
                'task_pointer': orchestrator.task_pool._task_pointer,
                'state_counts': state_counts,
                'queue_length': queue_length
            },
            'config': {
                'max_num_envs': orchestrator.config.max_num_envs,
                'max_task_steps': orchestrator.config.max_task_steps,
                'env_name': orchestrator.config.env_name
            }
        }

    return router


# 为了兼容性，需要导入TaskState
from src.models.task_models import TaskState
