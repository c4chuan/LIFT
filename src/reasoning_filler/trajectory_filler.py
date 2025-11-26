"""
轨迹填充模块

负责遍历轨迹、调用 API 并填充推理内容
"""

import json
import lzma
import pickle
import shutil
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from PIL import Image

from .prompt_builder import PromptBuilder,map_url_to_real
from .action_validator import ActionValidator
from .prompt_visualizer import PromptVisualizer, PromptRecord, ActionPromptHistory
from visualwebarena.src.envs.actions import Action,ActionTypes
from itertools import chain
from browser_env.constants import (
    ASCII_CHARSET,
    FREQ_UNICODE_CHARSET,
    MAX_ANSWER_LENGTH,
    MAX_ELEMENT_ID,
    MAX_ELEMENT_INDEX_IN_VIEWPORT,
    MAX_PAGE_NUMBER,
    MAX_VANILLA_STR_LENGTH,
    PLAYWRIGHT_ACTIONS,
    PLAYWRIGHT_LOCATORS,
    ROLES,
    SPECIAL_KEY_MAPPINGS,
    SPECIAL_KEYS,
    SPECIAL_LOCATORS,
    TEXT_MAX_LENGTH,
    TYPING_MAX_LENGTH,
    URL_MAX_LENGTH,
    RolesType,
)
_key2id: dict[str, int] = {
    key: i
    for i, key in enumerate(
        chain(SPECIAL_KEYS, ASCII_CHARSET, FREQ_UNICODE_CHARSET, ["\n"])
    )
}
_id2key: list[str] = sorted(_key2id, key=_key2id.get)  # type: ignore[arg-type]
class TrajectoryFiller:
    """轨迹填充器"""

    def __init__(
        self,
        reasoning_caller,
        prompt_builder: PromptBuilder,
        action_validator: ActionValidator = None,
        output_dir: str = "data/annotate_with_reasoning",
        enable_validation: bool = True,
        max_retry_attempts: int = 2,
        gpt4o_config: Optional[Dict[str, Any]] = None,
        visualization_config: Optional[Dict[str, Any]] = None
    ):
        """
        初始化轨迹填充器

        Args:
            reasoning_caller: 推理生成 API 调用器（QwenCaller 或 GPT4oCaller）
            prompt_builder: Prompt 构建器
            action_validator: Action 验证器（如果为None则创建默认的）
            output_dir: 输出目录
            enable_validation: 是否启用验证
            max_retry_attempts: 最大重试次数（总共会有 1+max_retry_attempts 次尝试）
            gpt4o_config: GPT-4o配置字典（包含api_key, base_url等）
            visualization_config: Prompt可视化配置字典
        """
        self.reasoning_caller = reasoning_caller
        self.prompt_builder = prompt_builder
        self.action_validator = action_validator if action_validator else ActionValidator(strict_matching=True)
        self.output_dir = Path(output_dir)
        self.enable_validation = enable_validation
        self.max_retry_attempts = max_retry_attempts

        # 初始化 GPT-4o 反馈生成器（如果配置存在且启用）
        self.gpt4o_feedback = None
        if gpt4o_config and gpt4o_config.get("enabled", False):
            try:
                from src.utils.gpt4o_feedback import GPT4oFeedbackGenerator
                self.gpt4o_feedback = GPT4oFeedbackGenerator(
                    api_key=gpt4o_config.get("api_key"),
                    base_url=gpt4o_config.get("base_url"),
                    model=gpt4o_config.get("model_name", "gpt-4o"),
                    max_retries=gpt4o_config.get("max_retries", 3),
                    timeout=gpt4o_config.get("timeout", 60)
                )
                print("✓ GPT-4o feedback generator initialized")
            except Exception as e:
                print(f"Warning: Failed to initialize GPT-4o feedback generator: {e}")
                print("Will use fallback correction method")

        # 初始化 Prompt 可视化器（如果配置存在且启用）
        self.enable_visualization = False
        self.visualizer = None
        if visualization_config and visualization_config.get("enable_prompt_visualization", False):
            try:
                viz_output_dir = visualization_config.get("output_dir", "data/prompt_visualizations")
                self.visualizer = PromptVisualizer(output_dir=viz_output_dir)
                self.enable_visualization = True
                print(f"✓ Prompt visualizer initialized (output: {viz_output_dir})")
            except Exception as e:
                print(f"Warning: Failed to initialize visualizer: {e}")
                self.enable_visualization = False

        # 验证统计信息
        self.validation_stats = {
            'total_actions': 0,
            'validation_passed': 0,
            'validation_failed': 0,
            'passed_first_attempt': 0,
            'corrected_first_retry': 0,
            'corrected_second_retry': 0,
            'total_api_calls': 0,
            'gpt4o_feedback_used': 0,
            'gpt4o_api_calls': 0
        }

    def fill_trajectory(
        self,
        trajectory: List[Any],
        metadata: Dict[str, Any],
        verbose: bool = True
    ) -> Tuple[List[Any], int]:
        """
        填充轨迹的推理内容

        Args:
            trajectory: 原始轨迹数据 (list[StateInfo | Action])
            metadata: 元数据
            verbose: 是否打印详细信息

        Returns:
            (填充后的轨迹, 填充的动作数) 元组
        """
        intent = metadata.get("intent", "")
        input_images = self._load_input_images(metadata.get("images", []))

        filled_trajectory = []
        action_history = []
        history_reasonings = []  # 历史推理对，用于提供上下文
        num_filled = 0

        # 可视化相关：记录每个 action 的 prompt/response 历史
        visualization_histories = []  # List[ActionPromptHistory]
        action_index = 0  # 当前 action 的索引（不包括 StateInfo）

        # 遍历轨迹
        for i, item in enumerate(trajectory):
            # 判断是 StateInfo 还是 Action
            if self._is_state_info(item):
                # StateInfo 直接保存
                filled_trajectory.append(item)
                current_state = item

            elif self._is_action(item):
                # Action 需要填充 raw_prediction
                self.validation_stats['total_actions'] += 1

                # 创建可视化历史记录（如果启用）
                current_action_history = None
                if self.enable_visualization:
                    current_action_history = ActionPromptHistory(action_index=action_index)

                try:
                    # 提取当前状态的截图和 URL
                    screenshot = self.prompt_builder.extract_screenshot_from_state(current_state)
                    current_url = self.prompt_builder.extract_url_from_state(current_state)

                    # 提取 observation text
                    observation_text = self.prompt_builder.extract_observation_text_from_state(current_state)

                    # 获取 ground truth 动作
                    gt_action = map_url_to_real(self._action_to_string(item))

                    # 构建 prompt（包含历史推理上下文）
                    messages = self.prompt_builder.build_reasoning_prompt(
                        intent=intent,
                        current_url=current_url,
                        current_screenshot=screenshot,
                        previous_actions=action_history,
                        ground_truth_action=gt_action,
                        input_images=input_images,
                        history_reasonings=history_reasonings,
                        current_observation_text=observation_text
                    )

                    # 第1次尝试：生成推理
                    if verbose:
                        print(f"  - 步骤 {i // 2 + 1}: 正在生成推理（第1次尝试）...")

                    reasoning = self.reasoning_caller.call(messages)
                    self.validation_stats['total_api_calls'] += 1

                    # 验证生成的action（如果启用验证）
                    is_valid = True
                    extracted_action = None
                    validation_msg = ""
                    retry_count = 0
                    gpt4o_feedback_text = None  # 用于记录 GPT-4o 反馈

                    if self.enable_validation:
                        is_valid, extracted_action, validation_msg = self.action_validator.extract_and_validate(
                            reasoning, gt_action
                        )

                        if verbose:
                            if is_valid:
                                print(f"    ✓ 验证通过 (action: {extracted_action})")
                            else:
                                print(f"    ✗ 验证失败: {validation_msg}")

                        # 记录第一次尝试（如果启用可视化）
                        if current_action_history is not None:
                            record = PromptRecord(
                                attempt_number=1,
                                messages=messages,
                                response=reasoning,
                                extracted_action=extracted_action if extracted_action else "[NO ACTION FOUND]",
                                validation_passed=is_valid,
                                ground_truth_action=gt_action,
                                gpt4o_feedback=None
                            )
                            current_action_history.add_record(record)

                        # 重试循环：如果验证失败且还有重试机会
                        while not is_valid and retry_count < self.max_retry_attempts:
                            retry_count += 1

                            if verbose:
                                print(f"  - 步骤 {i // 2 + 1}: 尝试纠正（第{retry_count + 1}次尝试）...")

                            # 如果启用了 GPT-4o 反馈
                            gpt4o_feedback_text = None  # 重置 GPT-4o 反馈
                            if self.gpt4o_feedback is not None:
                                try:
                                    if verbose:
                                        print(f"    → 调用 GPT-4o 分析错误...")

                                    # 调用 GPT-4o 分析错误
                                    gpt4o_feedback_text = self.gpt4o_feedback.analyze_error(
                                        intent=intent,
                                        current_url=current_url,
                                        observation_text=observation_text,
                                        screenshot=screenshot,
                                        generated_reasoning=reasoning,
                                        generated_action=extracted_action if extracted_action else "[NO ACTION FOUND]",
                                        ground_truth_action=gt_action,
                                        retry_count=retry_count
                                    )
                                    self.validation_stats['gpt4o_api_calls'] += 1
                                    self.validation_stats['gpt4o_feedback_used'] += 1

                                    if verbose:
                                        print(f"    ✓ GPT-4o 反馈已生成 ({len(gpt4o_feedback_text)} 字符)")

                                    # 构建基于 GPT-4o 反馈的纠正 prompt（每次都包含正确答案）
                                    correction_messages = self.prompt_builder.build_feedback_based_correction_prompt(
                                        original_messages=messages,
                                        gpt4o_feedback=gpt4o_feedback_text,
                                        ground_truth_action=gt_action,
                                        retry_count=retry_count
                                    )

                                except Exception as e:
                                    print(f"    ✗ GPT-4o 调用失败: {e}")
                                    print(f"    → 回退到标准纠正方式")
                                    # 回退到原有的直接纠正方式
                                    correction_messages = self.prompt_builder.build_correction_prompt(
                                        original_messages=messages,
                                        generated_action=extracted_action,
                                        ground_truth_action=gt_action,
                                        retry_count=retry_count
                                    )
                            else:
                                # 未启用 GPT-4o，使用原有的直接纠正方式
                                correction_messages = self.prompt_builder.build_correction_prompt(
                                    original_messages=messages,
                                    generated_action=extracted_action,
                                    ground_truth_action=gt_action,
                                    retry_count=retry_count
                                )

                            # 重新生成reasoning
                            reasoning = self.reasoning_caller.call(correction_messages)
                            self.validation_stats['total_api_calls'] += 1

                            # 再次验证
                            is_valid, extracted_action, validation_msg = self.action_validator.extract_and_validate(
                                reasoning, gt_action
                            )

                            if verbose:
                                if is_valid:
                                    print(f"    ✓ 纠正成功！(action: {extracted_action})")
                                else:
                                    print(f"    ✗ 仍然不匹配: {validation_msg}")

                            # 记录重试尝试（如果启用可视化）
                            if current_action_history is not None:
                                record = PromptRecord(
                                    attempt_number=retry_count + 1,
                                    messages=correction_messages,
                                    response=reasoning,
                                    extracted_action=extracted_action if extracted_action else "[NO ACTION FOUND]",
                                    validation_passed=is_valid,
                                    ground_truth_action=gt_action,
                                    gpt4o_feedback=gpt4o_feedback_text
                                )
                                current_action_history.add_record(record)

                    # 处理最终结果
                    if is_valid:
                        # 验证通过：保存reasoning
                        item.raw_prediction = reasoning
                        num_filled += 1
                        self.validation_stats['validation_passed'] += 1

                        # 记录成功的尝试次数
                        if retry_count == 0:
                            self.validation_stats['passed_first_attempt'] += 1
                        elif retry_count == 1:
                            self.validation_stats['corrected_first_retry'] += 1
                        elif retry_count == 2:
                            self.validation_stats['corrected_second_retry'] += 1

                        # 添加验证信息到action元数据
                        if hasattr(item, 'metadata'):
                            item.metadata['validation_info'] = {
                                'passed': True,
                                'retry_count': retry_count,
                                'attempts': retry_count + 1,
                                'extracted_action': extracted_action
                            }

                        # 将成功的推理添加到历史中，供后续步骤参考
                        history_reasonings.append({
                            "screenshot": screenshot,
                            "action": gt_action,
                            "reasoning": reasoning,
                            "previous_actions": "\n".join([f"- {a}" for a in action_history[-5:]])
                        })

                        # 只保留最近的5个历史推理，避免 prompt 过长
                        if len(history_reasonings) > 5:
                            history_reasonings = history_reasonings[-5:]

                        if verbose:
                            print(f"    ✓ 推理已保存 ({len(reasoning)} 字符, {retry_count + 1}次尝试)")

                    else:
                        # 验证失败且重试用尽：清空raw_prediction
                        item.raw_prediction = ""
                        self.validation_stats['validation_failed'] += 1

                        # 添加失败信息到action元数据
                        if hasattr(item, 'metadata'):
                            item.metadata['validation_info'] = {
                                'passed': False,
                                'failure_reason': validation_msg,
                                'last_generated_action': extracted_action,
                                'attempts': retry_count + 1
                            }

                        # 不加入history_reasonings（避免污染后续步骤）

                        if verbose:
                            print(f"    ✗ 验证失败，不保存 (尝试了{retry_count + 1}次)")

                except Exception as e:
                    print(f"  ✗ 步骤 {i // 2 + 1} 处理失败: {e}")
                    # 保留原有的 raw_prediction 或设为空
                    if not hasattr(item, "raw_prediction") or not item.raw_prediction:
                        item.raw_prediction = ""
                    self.validation_stats['validation_failed'] += 1

                # 保存填充后的 Action
                filled_trajectory.append(item)

                # 更新历史
                action_str = self._action_to_string(item)
                action_history.append(action_str)

                # 保存可视化历史（如果启用）
                if current_action_history is not None:
                    visualization_histories.append(current_action_history)

                # 递增 action 索引
                action_index += 1

            else:
                # 其他类型，直接保存
                filled_trajectory.append(item)

        # 生成可视化 HTML（如果启用且有记录）
        if self.enable_visualization and len(visualization_histories) > 0:
            try:
                # 从 metadata 获取必要信息（这些信息将在 save_filled_trajectory 中提供）
                # 这里我们临时保存，稍后在 save_filled_trajectory 中正式生成
                self._pending_visualization = {
                    'intent': intent,
                    'histories': visualization_histories
                }
            except Exception as e:
                print(f"Warning: Failed to prepare visualization data: {e}")

        return filled_trajectory, num_filled

    def save_filled_trajectory(
        self,
        trajectory: List[Any],
        metadata: Dict[str, Any],
        env_name: str,
        original_filename: str
    ) -> Path:
        """
        保存填充后的轨迹

        Args:
            trajectory: 填充后的轨迹
            metadata: 元数据
            env_name: 环境名称
            original_filename: 原始文件名

        Returns:
            保存的文件路径
        """
        # 创建环境目录
        env_dir = self.output_dir / env_name
        env_dir.mkdir(parents=True, exist_ok=True)

        # 保存轨迹文件
        traj_file = env_dir / original_filename
        with lzma.open(traj_file, 'wb') as f:
            pickle.dump(trajectory, f)

        # 保存元数据文件
        base_name = original_filename.replace('.pkl.xz', '')
        if base_name.endswith('.pkl'):
            base_name = base_name[:-4]

        metadata_file = env_dir / f"{base_name}_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        # 生成可视化 HTML（如果启用且有待处理的可视化数据）
        if self.enable_visualization and hasattr(self, '_pending_visualization'):
            try:
                viz_data = self._pending_visualization
                html_path = self.visualizer.generate_html(
                    env_name=env_name,
                    trajectory_name=original_filename,
                    intent=viz_data['intent'],
                    action_histories=viz_data['histories']
                )
                print(f"  ✓ 可视化文件已生成: {html_path}")
                # 清除待处理的可视化数据
                delattr(self, '_pending_visualization')
            except Exception as e:
                print(f"  ✗ 生成可视化文件失败: {e}")

        return traj_file

    @staticmethod
    def _is_state_info(item: Any) -> bool:
        """判断是否为 StateInfo"""
        return isinstance(item, dict) and "observation" in item and "info" in item

    @staticmethod
    def _is_action(item: Any) -> bool:
        """判断是否为 Action"""
        return isinstance(item, Action)

    @staticmethod
    def _action_to_string(action: Action) -> str:
        """
        将 Action 转换为字符串描述

        Args:
            action: Action dataclass

        Returns:
            动作字符串
        """
        element_id = action.element_id
        match action.action_type:
            case ActionTypes.CLICK:
                # [ID=X] xxxxx
                action_str = f"click [{element_id}]"
            case ActionTypes.CLEAR:
                action_str = f"clear [{element_id}]"
            case ActionTypes.TYPE:
                text = "".join([_id2key[i] for i in action.text]) if action.text else ""
                action_str = (
                    f"type [{element_id}] [{text}]"
                )
            case ActionTypes.HOVER:
                action_str = f"hover [{element_id}]"
            case ActionTypes.SCROLL:
                action_str = f"scroll [{action.direction}]"
            case ActionTypes.KEY_PRESS:
                action_str = f"press [{action.key_comb}]"
            case ActionTypes.GOTO_URL:
                action_str = f"goto [{action.url}]"
            case ActionTypes.NEW_TAB:
                action_str = "new_tab"
            case ActionTypes.PAGE_CLOSE:
                action_str = "close_tab"
            case ActionTypes.GO_BACK:
                action_str = "go_back"
            case ActionTypes.GO_FORWARD:
                action_str = "go_forward"
            case ActionTypes.PAGE_FOCUS:
                action_str = f"page_focus [{action.page_number}]"
            case ActionTypes.STOP:
                action_str = f"stop {action.answer}"
            case ActionTypes.NONE:
                action_str = "none"
        return action_str

    @staticmethod
    def _load_input_images(image_infos: List[Dict[str, Any]]) -> List[Image.Image]:
        """
        加载任务相关的输入图片

        Args:
            image_infos: 图片信息列表

        Returns:
            PIL Image 列表
        """
        images = []

        if not image_infos:
            return images

        for img_info in image_infos:
            try:
                img_type = img_info.get("type", "url")
                img_path = img_info.get("path", "")

                if img_type == "url":
                    # 从 URL 加载
                    import requests
                    response = requests.get(img_path, stream=True)
                    image = Image.open(response.raw)
                    images.append(image)

                elif img_type == "file":
                    # 从本地文件加载
                    image = Image.open(img_path)
                    images.append(image)

            except Exception as e:
                print(f"警告: 加载图片失败 {img_info}: {e}")

        return images


def main():
    """测试函数"""
    print("=== TrajectoryFiller 测试 ===")
    print("此模块需要配合其他模块使用，请运行 main.py 进行完整测试。")


if __name__ == "__main__":
    main()
