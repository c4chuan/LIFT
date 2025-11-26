"""
Action Validator Module

This module provides functionality to:
1. Extract action from generated reasoning text
2. Validate extracted action against ground truth
3. Support strict matching for quality control
"""

import re
from typing import Tuple, Optional


class ActionValidator:
    """验证器：从reasoning中提取action并验证其正确性"""

    def __init__(self, strict_matching: bool = True):
        """
        初始化验证器

        Args:
            strict_matching: 是否使用严格匹配模式（完全字符串匹配）
        """
        self.strict_matching = strict_matching

    def extract_action_from_reasoning(self, reasoning: str) -> Optional[str]:
        """
        从reasoning文本中提取<action>标签内的动作

        Args:
            reasoning: 完整的reasoning文本

        Returns:
            提取出的action字符串，如果没找到则返回None

        Examples:
            Input: "... <action>click [7]</action> ..."
            Output: "click [7]"
        """
        if not reasoning:
            return None

        # 匹配<action>标签内的内容
        action_pattern = r'<action>\s*(.*?)\s*</action>'
        match = re.search(action_pattern, reasoning, re.DOTALL | re.IGNORECASE)

        if match:
            action_text = match.group(1).strip()
            # 移除多余的空白字符，统一为单个空格
            action_text = ' '.join(action_text.split())
            return action_text

        return None

    def validate_action(
        self,
        extracted_action: Optional[str],
        ground_truth: str
    ) -> Tuple[bool, str]:
        """
        验证提取的action是否与ground truth匹配

        Args:
            extracted_action: 从reasoning中提取的action
            ground_truth: 标注的正确action

        Returns:
            (is_valid, message): 验证结果和消息

        Examples:
            validate_action("click [7]", "click [7]") -> (True, "Action matches")
            validate_action("click [12]", "click [7]") -> (False, "Action mismatch: ...")
        """
        # 标准化ground truth
        ground_truth = ground_truth.strip()
        ground_truth = ' '.join(ground_truth.split())

        # 检查是否成功提取
        if extracted_action is None:
            return False, "No action tag found in reasoning"

        # 严格匹配模式：完全字符串匹配
        if self.strict_matching:
            if extracted_action == ground_truth:
                return True, "Action matches perfectly"
            else:
                return False, f"Action mismatch: expected '{ground_truth}', got '{extracted_action}'"

        # 非严格模式（未来可扩展）
        else:
            # 可以在这里添加更宽松的匹配逻辑
            # 例如：忽略大小写、部分匹配等
            if extracted_action.lower() == ground_truth.lower():
                return True, "Action matches (case-insensitive)"
            else:
                return False, f"Action mismatch: expected '{ground_truth}', got '{extracted_action}'"

    def extract_and_validate(
        self,
        reasoning: str,
        ground_truth: str
    ) -> Tuple[bool, Optional[str], str]:
        """
        一站式提取并验证action

        Args:
            reasoning: 完整的reasoning文本
            ground_truth: 标注的正确action

        Returns:
            (is_valid, extracted_action, message):
                - is_valid: 验证是否通过
                - extracted_action: 提取出的action（可能为None）
                - message: 验证消息

        Examples:
            reasoning = "Let's observe... <action>click [7]</action>"
            extract_and_validate(reasoning, "click [7]")
            -> (True, "click [7]", "Action matches perfectly")
        """
        # 提取action
        extracted_action = self.extract_action_from_reasoning(reasoning)

        # 验证action
        is_valid, message = self.validate_action(extracted_action, ground_truth)

        return is_valid, extracted_action, message

    def get_validation_stats(self, validation_results: list) -> dict:
        """
        统计验证结果

        Args:
            validation_results: 验证结果列表，每个元素为(is_valid, extracted, message)

        Returns:
            统计信息字典
        """
        total = len(validation_results)
        passed = sum(1 for result in validation_results if result[0])
        failed = total - passed

        return {
            'total': total,
            'passed': passed,
            'failed': failed,
            'pass_rate': passed / total if total > 0 else 0.0
        }


# 便捷函数
def quick_validate(reasoning: str, ground_truth: str) -> bool:
    """
    快速验证函数（便捷接口）

    Args:
        reasoning: 完整的reasoning文本
        ground_truth: 标注的正确action

    Returns:
        是否验证通过
    """
    validator = ActionValidator(strict_matching=True)
    is_valid, _, _ = validator.extract_and_validate(reasoning, ground_truth)
    return is_valid
