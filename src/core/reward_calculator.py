"""
奖励计算模块

提供动作验证和奖励计算功能
"""
import re
from typing import Dict, Any, List, Optional

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

    async def calculate_batch_rewards(
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
            reward = await self.calculate_single_reward(response, task)
            rewards.append(reward)

        return rewards

    async def calculate_single_reward(
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
        reward = await self._calculate_reward_by_action_info(
            task,
            action_info,
            response
        )

        return reward

    async def _calculate_reward_by_action_info(
        self,
        task: VWATask,
        action_info: Dict[str, Any],
        response: str
    ) -> float:
        """
        根据动作信息计算奖励

        新逻辑:
        1. STOP动作: 使用 evaluator 进行评估
        2. 非STOP动作: 与参考轨迹中的动作进行对比(动作类型和element_id)

        Args:
            task: 任务实例
            action_info: 动作信息字典
            response: 原始响应

        Returns:
            奖励值
        """
        answer = action_info.get('answer', None)

        # 情况1: STOP动作 - 使用 evaluator 评估
        if answer is not None:
            return await self._evaluate_action_validation(
                task,
                answer=answer
            )

        # 情况2: 非STOP动作 - 与参考动作对比
        else:
            return self._compare_with_reference_action(
                task,
                action_info,
                response
            )

    async def _evaluate_action_validation(
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
            奖励值（0.0-1.0表示评估得分，-1.0表示无效扣分）
        """
        # 验证element_id是否在有效范围内
        if element_id is not None:
            obs_nodes_info = task.obs_info.get('observation_metadata', {}).get('image', {}).get('obs_nodes_info', {})

            if element_id in obs_nodes_info:
                return 0.0  # 有效，不加分
            else:
                return -1.0  # 无效，扣分

        # 验证STOP动作的答案 - 使用 evaluator
        elif answer is not None:
            try:
                from visualwebarena.src.evaluation.vwa_evaluators import evaluator_router
                from visualwebarena.browser_env.actions import create_stop_action

                # 构造带 answer 的 STOP action
                stop_action = create_stop_action(answer)

                # 使用现有的 state_trajectory，追加 STOP action
                trajectory = task.state_trajectory + [stop_action]

                # 调用 evaluator
                evaluator = evaluator_router(task.task_info['config_file'], captioning_fn=None)
                score = await evaluator(
                    trajectory=trajectory,
                    config_file=task.task_info['config_file'],
                    page=task.env.page
                )

                # 打印评估结果
                print(f"Task:{task.task_id}-STOP动作评估-Answer:[{answer}]-Score:{score}")

                return score  # 0.0 或 1.0
            except Exception as e:
                print(f"评估 STOP 动作时出错: {e}")
                import traceback
                traceback.print_exc()
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

    @staticmethod
    def _extract_action_type_from_response(response: str) -> Optional[str]:
        """
        从响应中提取动作类型

        Args:
            response: 响应文本

        Returns:
            动作类型字符串(如 "click", "type", "scroll" 等),如果提取失败则返回None
        """
        # 从代码块中提取动作
        pattern = r'```((.|\n)*?)```'
        match = re.search(pattern, response)
        if not match:
            return None

        action_str = match.group(1).strip()
        if not action_str:
            return None

        # 提取动作类型(第一个单词或[之前的部分)
        if "[" in action_str:
            action_type = action_str.split("[")[0].strip()
        else:
            actions = action_str.split()
            if actions:
                action_type = actions[0].strip()
            else:
                return None

        return action_type

    def _compare_with_reference_action(
        self,
        task: VWATask,
        action_info: Dict[str, Any],
        response: str
    ) -> float:
        """
        对比当前动作与参考动作

        Args:
            task: 任务实例
            action_info: 当前动作信息
            response: 响应文本

        Returns:
            奖励值: 1.0表示匹配,-1.0表示不匹配,0.0表示无法比较
        """
        # 检查是否为监督学习任务
        if not task.is_supervised():
            return 0.0

        # 获取当前动作索引
        action_index = task.get_current_action_index()

        # 获取参考动作
        ref_action = task.get_ref_action_at_index(action_index)
        if ref_action is None:
            return 0.0

        # 提取当前动作类型
        current_action_type = self._extract_action_type_from_response(response)
        if current_action_type is None:
            return 0.0

        # 获取参考动作类型(从action字典中)
        ref_action_type_id = ref_action.action_type
        if ref_action_type_id is None:
            return 0.0

        # 将参考动作类型ID转换为字符串名称
        # 根据ActionTypes枚举映射
        action_type_mapping = {
            0: "none",
            1: "scroll",
            2: "press",
            6: "click",
            7: "type",
            8: "hover",
            9: "page_focus",
            10: "new_tab",
            11: "go_back",
            12: "go_forward",
            13: "goto",
            14: "close_tab",
            17: "stop",
            18: "clear",
            19: "upload"
        }
        ref_action_type_name = action_type_mapping.get(ref_action_type_id, "unknown")

        # 对比动作类型
        if current_action_type.lower() != ref_action_type_name.lower():
            return 0.0

        # 动作类型匹配,进一步对比element_id(如果适用)
        current_element_id = action_info.get('element_id')
        ref_element_id = ref_action.element_id

        # 如果两者都有element_id,则进行对比
        if current_element_id and ref_element_id:
            if str(current_element_id) == str(ref_element_id):
                return 1.0
            else:
                return 0.0

        # 如果没有element_id参与,动作类型匹配就算成功
        return 1.0
