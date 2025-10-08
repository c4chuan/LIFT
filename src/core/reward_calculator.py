"""
奖励计算模块

提供动作验证和奖励计算功能
"""
import re
from typing import Dict, Any, List

from src.models.task_models import VWATask
from src.reward.reward_tools import get_action_id_answer, action_format_reward


class RewardCalculator:
    """
    奖励计算器

    职责：
    1. 验证动作的有效性
    2. 计算动作奖励
    3. 评估动作与参考轨迹的对齐程度
    """

    def __init__(self):
        """初始化奖励计算器"""
        pass

    def calculate_batch_rewards(
        self,
        responses: List[str],
        tasks: List[VWATask]
    ) -> List[float]:
        """
        批量计算动作奖励

        Args:
            responses: 响应列表
            tasks: 对应的任务列表

        Returns:
            奖励列表
        """
        if len(responses) != len(tasks):
            raise ValueError(f"响应数量({len(responses)})与任务数量({len(tasks)})不匹配")

        rewards = []
        for response, task in zip(responses, tasks):
            reward = self.calculate_single_reward(response, task)
            rewards.append(reward)

        return rewards

    def calculate_single_reward(
        self,
        response: str,
        task: VWATask
    ) -> float:
        """
        计算单个响应的奖励

        Args:
            response: LLM响应
            task: 任务实例

        Returns:
            奖励值
        """
        # 1. 检查格式是否正确
        if action_format_reward(response) == 0:
            return 0.0

        # 2. 提取动作信息
        try:
            action_info = get_action_id_answer(response)
        except Exception:
            return 0.0

        # 3. 根据动作信息计算奖励
        reward = self._calculate_reward_by_action_info(
            task,
            action_info,
            response
        )

        return reward

    def _calculate_reward_by_action_info(
        self,
        task: VWATask,
        action_info: Dict[str, Any],
        response: str
    ) -> float:
        """
        根据动作信息计算奖励

        Args:
            task: 任务实例
            action_info: 动作信息字典
            response: 原始响应

        Returns:
            奖励值
        """
        element_id = action_info.get('element_id', -1)
        answer = action_info.get('answer', None)
        url = action_info.get('url', None)

        # 情况1: 与element_id无关的动作
        if element_id == -1:
            return 0.0

        # 情况2: 基于element_id的动作
        elif element_id and int(element_id) > 0:
            # 验证element_id是否有效
            validation_reward = self._evaluate_action_validation(
                task,
                element_id=element_id
            )

            # 评估点击对齐度
            alignment_reward = self._evaluate_action_click_alignment(
                task,
                element_id,
                response
            )

            return validation_reward + alignment_reward

        # 情况3: STOP类型动作（有answer）
        elif isinstance(answer, str):
            return self._evaluate_action_validation(
                task,
                answer=answer
            )

        # 情况4: 导航类型动作（有url）
        elif isinstance(url, str):
            if "http://" in url or "https://" in url:
                return 0.0
            else:
                return -1.0

        else:
            return 0.0

    def _evaluate_action_validation(
        self,
        task: VWATask,
        element_id: Any = None,
        answer: Any = None
    ) -> float:
        """
        验证动作的有效性

        Args:
            task: 任务实例
            element_id: 元素ID（可选）
            answer: 答案（可选）

        Returns:
            奖励值（0.0表示有效但不加分，-1.0表示无效扣分）
        """
        # 验证element_id是否在有效范围内
        if element_id is not None:
            obs_nodes_info = task.obs_info.get('observation_metadata', {}).get('image', {}).get('obs_nodes_info', {})

            if element_id in obs_nodes_info:
                return 0.0  # 有效，不加分
            else:
                return -1.0  # 无效，扣分

        # 验证STOP动作的答案
        elif answer is not None:
            # TODO: 这里可以集成evaluator来验证答案正确性
            # 目前暂时返回0.0
            return 0.0

        else:
            return 0.0

    def _evaluate_action_click_alignment(
        self,
        task: VWATask,
        element_id: Any,
        response: str
    ) -> float:
        """
        评估点击动作与描述的对齐程度

        通过检查summary中是否包含element的描述信息来评估

        Args:
            task: 任务实例
            element_id: 元素ID
            response: 响应文本

        Returns:
            奖励值（0.5表示对齐，0.0表示不对齐）
        """
        # 提取summary
        summary = self._extract_summary(response)
        if not summary:
            return 0.0

        # 检查element_id是否在obs_nodes_info中
        obs_nodes_info = task.obs_info.get('observation_metadata', {}).get('image', {}).get('obs_nodes_info', {})
        if element_id not in obs_nodes_info:
            return 0.0

        # 获取element的语义信息
        obs_nodes_semantic_info = task.obs_info.get('observation_metadata', {}).get('image', {}).get('obs_nodes_semantic_info', {})
        if element_id not in obs_nodes_semantic_info:
            return 0.0

        # 提取描述信息
        id_description = self._get_element_id_description(
            obs_nodes_semantic_info[element_id]
        )

        # 检查描述中的任何组件是否在summary中
        for component in id_description.split(" "):
            if component and component in summary:
                return 0.5

        return 0.0

    @staticmethod
    def _extract_summary(response: str) -> str:
        """
        从响应中提取summary

        Args:
            response: 响应文本

        Returns:
            summary内容
        """
        pattern = r'<summary>([\s\S]*?)</summary>'
        match = re.search(pattern, response)
        if match:
            return match.group(1)
        return ""

    @staticmethod
    def _get_element_id_description(semantic_string: str) -> str:
        """
        从语义字符串中提取描述

        Args:
            semantic_string: 语义字符串

        Returns:
            描述文本
        """
        # 找到所有 [...] 的内容
        matches = re.findall(r'\[([^\]]*)\]', semantic_string)
        # 返回最后一个，否则返回空
        return matches[-1] if matches else ''
