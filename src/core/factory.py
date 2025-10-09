"""
系统工厂模块

提供便捷的系统初始化和组装功能
"""
from typing import Dict, Any, List

from src.config.environment_config import EnvironmentConfig
from src.core.task_pool import TaskPool
from src.core.message_builder import IMessageBuilder, EnvLIFTMessageBuilder
from src.core.reward_calculator import RewardCalculator
from src.core.action_strategy import IActionStrategy, create_action_strategy
from src.core.orchestrator import EnvironmentOrchestrator
from src.agentic.policy import EnvLIFTConstructor
from src.utils.llm_config import construct_llm_config
from visualwebarena.src.llms.tokenizer import Tokenizer


class EnvironmentSystemFactory:
    """
    环境系统工厂

    负责创建和组装整个系统的所有组件
    """

    @staticmethod
    def create_orchestrator(
        config: EnvironmentConfig,
        task_configs: List[Dict[str, Any]],
        action_strategy_mode: str = "standard"
    ) -> EnvironmentOrchestrator:
        """
        创建环境编排器

        这是创建整个系统的主入口

        Args:
            config: 环境配置
            task_configs: 任务配置列表
            action_strategy_mode: 动作策略模式
                - "standard": 标准模式（直接使用模型动作）
                - "supervised": 监督学习模式（可能使用参考动作）

        Returns:
            配置完成的环境编排器
        """
        # 1. 创建TaskPool
        task_pool = TaskPool(config, task_configs)

        # 2. 创建MessageBuilder
        message_builder = EnvironmentSystemFactory._create_message_builder(config)

        # 3. 创建RewardCalculator
        reward_calculator = RewardCalculator()

        # 4. 创建ActionStrategy
        action_strategy = create_action_strategy(
            mode=action_strategy_mode,
            use_reference_probability=1.0  # 默认使用参考动作
        )

        # 5. 创建Orchestrator
        orchestrator = EnvironmentOrchestrator(
            config=config,
            task_pool=task_pool,
            message_builder=message_builder,
            reward_calculator=reward_calculator,
            action_strategy=action_strategy
        )

        return orchestrator

    @staticmethod
    def _create_message_builder(config: EnvironmentConfig) -> IMessageBuilder:
        """
        创建消息构建器

        Args:
            config: 环境配置

        Returns:
            消息构建器实例
        """
        # 创建EnvLIFTConstructor
        prompt_constructor = EnvLIFTConstructor(
            instruction_path=config.instruction_path,
            save_dir=config.results_dir,
            lm_config=construct_llm_config(),
            tokenizer=Tokenizer(provider="openai", model_name="gpt-4o")
        )

        # 封装为MessageBuilder
        message_builder = EnvLIFTMessageBuilder(prompt_constructor)

        return message_builder

    @staticmethod
    def create_from_dict(
        config_dict: Dict[str, Any],
        task_configs: List[Dict[str, Any]]
    ) -> EnvironmentOrchestrator:
        """
        从字典配置创建系统

        Args:
            config_dict: 配置字典
            task_configs: 任务配置列表

        Returns:
            环境编排器
        """
        # 创建配置对象
        config = EnvironmentConfig(**config_dict)

        # 判断是否为监督学习模式
        action_strategy_mode = "supervised" if config.is_supervised_mode() else "standard"

        # 创建orchestrator
        orchestrator = EnvironmentSystemFactory.create_orchestrator(
            config,
            task_configs,
            action_strategy_mode
        )

        return orchestrator
