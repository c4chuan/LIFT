"""
动作决策策略模块

使用策略模式区分监督学习和标准学习的动作决策逻辑
"""
from abc import ABC, abstractmethod
from typing import Any

from src.models.task_models import VWATask


class IActionStrategy(ABC):
    """动作决策策略接口"""

    @abstractmethod
    def decide_action(
        self,
        task: VWATask,
        current_action: Any
    ) -> Any:
        """
        决定应该使用的动作

        Args:
            task: 任务实例
            current_action: 当前动作（来自模型或上层）

        Returns:
            实际应该执行的动作
        """
        pass


class StandardActionStrategy(IActionStrategy):
    """
    标准动作策略

    直接使用模型生成的动作，不进行任何修改
    适用于非监督学习模式
    """

    def decide_action(
        self,
        task: VWATask,
        current_action: Any
    ) -> Any:
        """
        直接返回当前动作

        Args:
            task: 任务实例
            current_action: 当前动作

        Returns:
            当前动作（不做修改）
        """
        return current_action


class SupervisedActionStrategy(IActionStrategy):
    """
    监督学习动作策略

    根据参考轨迹和当前动作的比较，决定使用哪个动作
    可以在此基础上实现更复杂的策略，如：
    - 随机替换
    - 基于相似度的替换
    - 渐进式替换等
    """

    def __init__(self, use_reference_probability: float = 0.0):
        """
        初始化监督学习策略

        Args:
            use_reference_probability: 使用参考动作的概率（0-1）
                0.0 表示总是使用当前动作
                1.0 表示总是使用参考动作
                0.5 表示50%概率使用参考动作
        """
        self.use_reference_probability = use_reference_probability

    def decide_action(
        self,
        task: VWATask,
        current_action: Any
    ) -> Any:
        """
        根据策略决定使用当前动作还是参考动作

        Args:
            task: 任务实例
            current_action: 当前动作

        Returns:
            决定后的动作
        """
        # 如果不是监督任务，直接返回当前动作
        if not task.is_supervised():
            return current_action

        # 获取当前动作索引
        action_index = task.get_current_action_index()

        # 获取参考动作
        ref_action = task.get_ref_action_at_index(action_index)

        # 如果没有参考动作，使用当前动作
        if ref_action is None:
            return current_action

        # 决定是否使用参考动作
        should_use_reference = self._should_use_reference_action(
            current_action,
            ref_action,
            task
        )

        if should_use_reference:
            return ref_action
        else:
            return current_action

    def _should_use_reference_action(
        self,
        current_action: Any,
        ref_action: Any,
        task: VWATask
    ) -> bool:
        """
        判断是否应该使用参考动作

        当前实现：总是使用当前动作（概率为0）
        可以扩展为更复杂的逻辑

        Args:
            current_action: 当前动作
            ref_action: 参考动作
            task: 任务实例

        Returns:
            True表示使用参考动作，False表示使用当前动作
        """
        if current_action.obs_reward > ref_action.obs_reward:
            return False
        else:
            return True


class AdaptiveActionStrategy(IActionStrategy):
    """
    自适应动作策略

    根据任务的历史表现动态调整使用参考动作的概率
    例如：如果模型表现好，减少使用参考动作；反之增加
    """

    def __init__(
        self,
        initial_probability: float = 0.5,
        adaptation_rate: float = 0.1
    ):
        """
        初始化自适应策略

        Args:
            initial_probability: 初始使用参考动作的概率
            adaptation_rate: 适应速率（调整概率的步长）
        """
        self.probability = initial_probability
        self.adaptation_rate = adaptation_rate
        self.performance_history = []

    def decide_action(
        self,
        task: VWATask,
        current_action: Any
    ) -> Any:
        """
        自适应地决定使用哪个动作

        Args:
            task: 任务实例
            current_action: 当前动作

        Returns:
            决定后的动作
        """
        if not task.is_supervised():
            return current_action

        action_index = task.get_current_action_index()
        ref_action = task.get_ref_action_at_index(action_index)

        if ref_action is None:
            return current_action

        # 基于当前概率决定
        import random
        should_use_reference = random.random() < self.probability

        if should_use_reference:
            return ref_action
        else:
            return current_action

    def update_performance(self, reward: float):
        """
        根据奖励更新策略

        Args:
            reward: 奖励值
        """
        self.performance_history.append(reward)

        # 如果奖励为负，增加使用参考动作的概率
        if reward < 0:
            self.probability = min(1.0, self.probability + self.adaptation_rate)
        # 如果奖励为正，减少使用参考动作的概率
        elif reward > 0:
            self.probability = max(0.0, self.probability - self.adaptation_rate)

        # 保持概率在[0, 1]范围内
        self.probability = max(0.0, min(1.0, self.probability))


# 工厂函数
def create_action_strategy(
    mode: str = "standard",
    **kwargs
) -> IActionStrategy:
    """
    创建动作策略

    Args:
        mode: 策略模式
            - "standard": 标准策略
            - "supervised": 监督学习策略
            - "adaptive": 自适应策略
        **kwargs: 传递给策略构造函数的参数

    Returns:
        动作策略实例
    """
    if mode == "standard":
        return StandardActionStrategy()
    elif mode == "supervised":
        return SupervisedActionStrategy(**kwargs)
    elif mode == "adaptive":
        return AdaptiveActionStrategy(**kwargs)
    else:
        raise ValueError(f"未知的策略模式: {mode}")
