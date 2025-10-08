"""
消息构建模块

提供统一的消息构建接口，封装prompt构建逻辑
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List
from PIL import Image

from src.models.task_models import VWATask
from vwa.src.helper_functions import get_action_description


class IMessageBuilder(ABC):
    """消息构建器接口"""

    @abstractmethod
    def construct_message(
        self,
        task: VWATask,
        obs: Dict[str, Any],
        info: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        构建消息

        Args:
            task: 任务实例
            obs: 观察数据
            info: 信息字典

        Returns:
            消息列表（LLM格式）
        """
        pass

    @abstractmethod
    def extract_action(self, response: str) -> str:
        """
        从响应中提取动作

        Args:
            response: LLM响应

        Returns:
            动作字符串
        """
        pass

    @abstractmethod
    def extract_summary(self, response: str) -> str:
        """
        从响应中提取summary

        Args:
            response: LLM响应

        Returns:
            summary字符串
        """
        pass


class EnvLIFTMessageBuilder(IMessageBuilder):
    """
    基于EnvLIFTConstructor的消息构建器

    封装EnvLIFTConstructor，提供统一接口
    """

    def __init__(self, prompt_constructor):
        """
        初始化

        Args:
            prompt_constructor: EnvLIFTConstructor实例
        """
        self.prompt_constructor = prompt_constructor

    def construct_message(
        self,
        task: VWATask,
        obs: Dict[str, Any],
        info: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        构建消息

        根据任务当前状态构建适当的消息
        """
        # 提取必要的信息
        page_screenshot_img = Image.fromarray(obs['image']) if obs['image'] is not None else None

        # 构建action_history
        if len(task.action_history) == 0:
            action_history = ["None"]
        else:
            action_history = task.action_history

        # 调用prompt_constructor
        message = self.prompt_constructor.construct(
            task_id=task.task_id,
            trajectory=task.state_trajectory,
            intent=task.task_info.get('intent', ''),
            page_screenshot_img=page_screenshot_img,
            images=task.task_info.get('images', []),
            meta_data={"action_history": action_history}
        )

        return message

    def extract_action(self, response: str) -> str:
        """从响应中提取动作"""
        return self.prompt_constructor.extract_action(response)

    def extract_summary(self, response: str) -> str:
        """从响应中提取summary"""
        return self.prompt_constructor.exstract_summary(response)


class ActionDescriptionHelper:
    """
    动作描述辅助类

    提供将动作转换为可读字符串的功能
    """

    @staticmethod
    def get_action_description(
        action: Any,
        observation_metadata: Dict[str, Any],
        action_set_tag: str,
        prompt_constructor: Any
    ) -> str:
        """
        获取动作的描述字符串

        Args:
            action: 动作对象
            observation_metadata: 观察元数据
            action_set_tag: 动作集标签
            prompt_constructor: prompt构造器

        Returns:
            动作描述字符串
        """
        return get_action_description(
            action,
            observation_metadata,
            action_set_tag=action_set_tag,
            prompt_constructor=prompt_constructor
        )
