"""
环境控制器模块
负责管理浏览器环境的生命周期和状态
"""

import asyncio
import json
import os
import subprocess
import tempfile
import sys
from typing import Dict, Any, Tuple, Optional, List
from pathlib import Path
from PIL import Image

# 尝试导入cv2和numpy，如果失败则使用替代方案
try:
    import cv2
    import numpy as np
    HAS_CV2 = True
except ImportError:
    print("[INFO] cv2或numpy未安装，将使用PIL作为替代方案")
    HAS_CV2 = False
    # 创建numpy的简单替代
    class np:
        @staticmethod
        def ndarray(*args, **kwargs):
            pass

# 添加路径以便导入visualwebarena模块
sys.path.append(str(Path(__file__).parent.parent.parent / "visualwebarena"))

from visualwebarena.src.envs.actions import Action
from browser_env import StateInfo, Trajectory
from browser_env.auto_login import get_site_comb_from_filepath
from src.envs.browser import FastCachedwActionMatchingBrowserEnv, early_stop
from src.helper_functions import RenderHelper
from src.evaluation.vwa_evaluators import evaluator_router
from src.evaluation import image_utils


class EnvironmentController:
    """环境控制器类"""

    def __init__(self,
                 render: bool = True,
                 slow_mo: int = 0,
                 observation_type: str = "image_som",
                 viewport_width: int = 1280,
                 viewport_height: int = 2048,
                 save_trace_enabled: bool = False,
                 sleep_after_execution: float = 2.5,
                 max_steps: int = 30,
                 action_set_tag: str = "id_accessibility_tree",
                 result_dir: str = "data/annotation_results"):
        """
        初始化环境控制器

        Args:
            render: 是否渲染浏览器界面
            slow_mo: 浏览器操作延迟（毫秒）
            observation_type: 观察类型
            viewport_width: 视窗宽度
            viewport_height: 视窗高度
            save_trace_enabled: 是否保存轨迹
            sleep_after_execution: 执行后等待时间
            max_steps: 最大步数
            action_set_tag: 动作集标签
            result_dir: 结果目录
        """
        self.render = render
        self.slow_mo = slow_mo
        self.observation_type = observation_type
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height
        self.save_trace_enabled = save_trace_enabled
        self.sleep_after_execution = sleep_after_execution
        self.max_steps = max_steps
        self.action_set_tag = action_set_tag
        self.result_dir = Path(result_dir)

        # 确保结果目录存在
        self.result_dir.mkdir(parents=True, exist_ok=True)

        self.env = None
        self.render_helper = None
        self.current_config_file = None
        self.current_trajectory = []
        self.current_step = 0
        self.current_task_id = None
        self.screenshot_paths = []  # 存储所有截图路径

        # 早停阈值
        self.early_stop_thresholds = {
            "parsing_failure": 3,
            "repeating_action": 5,
        }

    async def initialize_environment(self, config_file: str) -> Tuple[StateInfo, Dict[str, Any]]:
        """
        初始化环境

        Args:
            config_file: 任务配置文件路径

        Returns:
            (初始状态信息, 任务信息)
        """
        # 预处理配置文件（处理自动登录）
        processed_config_file = await self._preprocess_config(config_file)
        self.current_config_file = processed_config_file

        # 创建浏览器环境
        self.env = FastCachedwActionMatchingBrowserEnv(
            headless=not self.render,
            slow_mo=self.slow_mo,
            action_set_tag=self.action_set_tag,
            observation_type=self.observation_type,
            current_viewport_only=True,  # 固定为True以提高性能
            viewport_size={
                "width": self.viewport_width,
                "height": self.viewport_height,
            },
            save_trace_enabled=self.save_trace_enabled,
            sleep_after_execution=self.sleep_after_execution,
            captioning_fn=None,  # 暂时不使用描述功能
        )

        # 创建渲染助手
        self.render_helper = RenderHelper(
            processed_config_file, str(self.result_dir), self.action_set_tag
        )

        # 重置环境
        obs, info = await self.env.areset(options={"config_file": processed_config_file})
        state_info = {"observation": obs, "info": info, "url": self.env.page.url}

        # 初始化轨迹
        self.current_trajectory = [state_info]
        self.current_step = 0

        # 加载任务信息
        task_info = await self._load_task_info(processed_config_file)
        self.current_task_id = task_info["task_id"]

        # 创建任务特定的截图目录
        self.task_screenshot_dir = self.result_dir / f"task_{self.current_task_id}"
        self.task_screenshot_dir.mkdir(parents=True, exist_ok=True)

        # 重置截图路径列表
        self.screenshot_paths = []

        # 保存初始状态截图
        initial_screenshot_path = self._save_som_screenshot(state_info, is_initial=True)
        if initial_screenshot_path:
            self.screenshot_paths.append(initial_screenshot_path)

        return state_info, task_info

    async def _preprocess_config(self, config_file: str) -> str:
        """
        预处理配置文件，处理自动登录等

        Args:
            config_file: 原始配置文件路径

        Returns:
            处理后的配置文件路径
        """
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)

        # 处理自动登录
        if config.get("storage_state"):
            cookie_file_name = os.path.basename(config["storage_state"])
            comb = get_site_comb_from_filepath(cookie_file_name)
            temp_dir = tempfile.mkdtemp()

            # 运行自动登录脚本
            subprocess.run([
                "python",
                "-m",
                "browser_env.auto_login",
                "--auth_folder",
                temp_dir,
                "--site_list",
                *comb,
            ], check=True)

            # 更新配置中的登录状态文件路径
            config["storage_state"] = f"{temp_dir}/{cookie_file_name}"

            # 检查文件是否存在
            if not os.path.exists(config["storage_state"]):
                raise FileNotFoundError(f"登录状态文件不存在: {config['storage_state']}")

            # 保存处理后的配置文件
            processed_config_file = f"{temp_dir}/{os.path.basename(config_file)}"
            with open(processed_config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)

            return processed_config_file

        return config_file

    async def _load_task_info(self, config_file: str) -> Dict[str, Any]:
        """
        加载任务信息

        Args:
            config_file: 配置文件路径

        Returns:
            任务信息字典
        """
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)

        # 加载任务图片（如果有）
        images = []
        image_paths = config.get("image")
        if image_paths:
            if isinstance(image_paths, str):
                image_paths = [image_paths]

            for image_path in image_paths:
                try:
                    if image_path.startswith("http"):
                        # 网络图片，这里只记录路径，不实际加载
                        images.append({"type": "url", "path": image_path})
                    else:
                        # 本地图片
                        if os.path.exists(image_path):
                            images.append({"type": "local", "path": image_path})
                        else:
                            images.append({"type": "missing", "path": image_path})
                except Exception as e:
                    print(f"警告: 加载图片失败 {image_path}: {e}")

        return {
            "config_file": config_file,
            "task_id": config["task_id"],
            "intent": config["intent"],
            "images": images,
            "start_url": config.get("start_url"),
            "require_login": config.get("require_login", False),
        }

    async def execute_action(self, action: Action) -> Tuple[bool, Optional[str], StateInfo]:
        """
        执行动作

        Args:
            action: 要执行的动作

        Returns:
            (是否成功, 错误信息, 新状态信息)
        """
        try:
            self.current_step += 1

            # 检查早停条件
            early_stop_flag, stop_info = early_stop(
                self.current_trajectory, self.max_steps, self.early_stop_thresholds
            )

            if early_stop_flag:
                return False, f"早停: {stop_info}", self._get_current_state()

            # 修正动作ID（如果需要）
            corrected_action = self.env.maybe_update_action_id(action)

            # 添加动作到轨迹
            self.current_trajectory.append(corrected_action)

            # 执行动作
            obs, success, terminated, _, info = await self.env.astep(corrected_action)

            # 创建新的状态信息
            new_state_info = {
                "observation": obs,
                "info": info,
                "url": self.env.page.url
            }

            # 添加状态到轨迹
            self.current_trajectory.append(new_state_info)

            # 保存SOM截图
            screenshot_path = self._save_som_screenshot(new_state_info, is_initial=False)
            if screenshot_path:
                self.screenshot_paths.append(screenshot_path)

            # 渲染当前状态（生成HTML可视化）
            self._render_current_state(corrected_action, new_state_info)

            error_message = None
            if not success:
                error_message = info.get('fail_error', '动作执行失败')

            return success, error_message, new_state_info

        except Exception as e:
            error_message = f"执行动作时发生异常: {str(e)}"
            return False, error_message, self._get_current_state()

    def _get_current_state(self) -> StateInfo:
        """获取当前状态信息"""
        if self.current_trajectory and len(self.current_trajectory) > 0:
            # 返回最后一个状态信息
            for item in reversed(self.current_trajectory):
                if isinstance(item, dict) and "observation" in item:
                    return item

        # 如果没有找到状态信息，返回空状态
        return {"observation": None, "info": {}, "url": ""}

    def _save_som_screenshot(self, state_info: StateInfo, is_initial: bool = False) -> Optional[str]:
        """
        保存SOM截图并返回文件路径

        Args:
            state_info: 状态信息
            is_initial: 是否是初始状态

        Returns:
            截图文件路径，如果保存失败则返回None
        """
        try:
            # 检查是否有图像观察数据
            observation = state_info.get("observation")
            if not observation or "image" not in observation:
                print("警告: 状态信息中没有图像数据")
                return None

            img_array = observation["image"]
            if img_array is None:
                print("警告: 图像数组为空")
                return None

            # 确保任务截图目录存在
            if not hasattr(self, 'task_screenshot_dir') or not self.task_screenshot_dir:
                print("警告: 任务截图目录未初始化")
                return None

            # 生成文件名
            if is_initial:
                filename = "step_0_initial_obs.png"
            else:
                filename = f"step_{self.current_step}_obs.png"

            file_path = self.task_screenshot_dir / filename

            # 转换图像数组格式并保存
            if HAS_CV2 and hasattr(img_array, 'shape') and isinstance(img_array, type(np.array([]))):
                # 使用OpenCV保存（如果可用）
                try:
                    # 如果是RGB格式，转换为BGR格式供OpenCV使用
                    if len(img_array.shape) == 3 and img_array.shape[-1] == 3:  # RGB
                        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                    else:
                        img_bgr = img_array

                    # 使用OpenCV保存图像
                    success = cv2.imwrite(str(file_path), cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
                    if not success:
                        print(f"警告: OpenCV保存截图失败: {file_path}")
                        return None
                except Exception as e:
                    print(f"警告: OpenCV保存失败，尝试使用PIL: {e}")
                    # 降级到PIL
                    return self._save_with_pil(img_array, file_path)
            else:
                # 使用PIL保存
                return self._save_with_pil(img_array, file_path)

            print(f"成功保存SOM截图: {file_path}")
            return str(file_path)

        except Exception as e:
            print(f"错误: 保存SOM截图时发生异常: {e}")
            return None

    def _save_with_pil(self, img_array, file_path):
        """
        使用PIL保存图像的辅助方法

        Args:
            img_array: 图像数组或PIL Image对象
            file_path: 保存路径

        Returns:
            文件路径或None
        """
        try:
            if isinstance(img_array, Image.Image):
                # 直接保存PIL Image
                img_array.save(file_path)
            elif hasattr(img_array, 'shape'):
                # 尝试转换numpy数组为PIL Image
                try:
                    pil_image = Image.fromarray(img_array)
                    pil_image.save(file_path)
                except Exception as e:
                    print(f"警告: 无法转换数组为PIL图像: {e}")
                    return None
            elif hasattr(img_array, 'save'):
                # 如果对象有save方法，尝试直接调用
                img_array.save(file_path)
            else:
                print(f"警告: 不支持的图像类型: {type(img_array)}")
                return None

            print(f"成功保存SOM截图（使用PIL）: {file_path}")
            return str(file_path)

        except Exception as e:
            print(f"警告: PIL保存截图失败: {e}")
            return None

    def _render_current_state(self, action: Action, state_info: StateInfo) -> None:
        """
        渲染当前状态

        Args:
            action: 刚执行的动作
            state_info: 当前状态信息
        """
        if self.render_helper:
            try:
                # 使用RenderHelper生成截图
                meta_data = {"action_history": [f"Step {self.current_step}"]}
                self.render_helper.render(
                    action,
                    state_info,
                    meta_data,
                    render_screenshot=True,
                    all_candidates=None
                )
            except Exception as e:
                print(f"警告: 渲染失败: {e}")

    def get_current_trajectory(self) -> Trajectory:
        """获取当前轨迹"""
        return self.current_trajectory.copy()

    def get_current_step_count(self) -> int:
        """获取当前步数"""
        return self.current_step

    async def evaluate_trajectory(self) -> float:
        """
        评估当前轨迹

        Returns:
            评估分数 (0.0 - 1.0)
        """
        if not self.current_config_file or not self.env:
            return 0.0

        try:
            # 创建评估器
            evaluator = evaluator_router(self.current_config_file)

            # 执行评估
            score = await evaluator(
                trajectory=self.current_trajectory,
                config_file=self.current_config_file,
                page=self.env.page
            )

            return float(score)

        except Exception as e:
            print(f"评估失败: {e}")
            return 0.0

    async def reset_environment(self) -> Tuple[StateInfo, Dict[str, Any]]:
        """
        重置环境到初始状态

        Returns:
            (初始状态信息, 任务信息)
        """
        if not self.current_config_file:
            raise RuntimeError("没有当前配置文件，无法重置")

        # 重置环境
        obs, info = await self.env.areset(options={"config_file": self.current_config_file})
        state_info = {"observation": obs, "info": info, "url": self.env.page.url}

        # 重置轨迹和步数
        self.current_trajectory = [state_info]
        self.current_step = 0

        # 重置截图路径列表
        self.screenshot_paths = []

        # 保存重置后的初始状态截图
        initial_screenshot_path = self._save_som_screenshot(state_info, is_initial=True)
        if initial_screenshot_path:
            self.screenshot_paths.append(initial_screenshot_path)

        # 重新加载任务信息
        task_info = await self._load_task_info(self.current_config_file)

        return state_info, task_info

    async def close_environment(self) -> None:
        """关闭环境"""
        # 在关闭环境前保存轨迹数据
        if self.current_task_id and self.current_trajectory:
            self.save_trajectory_data()

        if self.env:
            await self.env.aclose()
            self.env = None

        if self.render_helper:
            self.render_helper.close()
            self.render_helper = None

    def get_latest_screenshot_path(self) -> Optional[str]:
        """
        获取最新截图的路径

        Returns:
            截图文件路径，如果没有则返回None
        """
        if self.screenshot_paths:
            return self.screenshot_paths[-1]
        return None

    def get_screenshot_path(self, step: int) -> Optional[str]:
        """
        获取指定步骤的截图路径

        Args:
            step: 步骤索引（0表示初始状态）

        Returns:
            截图文件路径，如果不存在则返回None
        """
        if not self.current_task_id:
            return None

        # 构建截图文件路径
        if step == 0:
            filename = "step_0_initial_obs.png"
        else:
            filename = f"step_{step}_obs.png"

        file_path = self.task_screenshot_dir / filename
        if file_path.exists():
            return str(file_path)
        return None

    def get_all_screenshots(self) -> List[str]:
        """
        获取当前轨迹的所有截图路径

        Returns:
            截图文件路径列表
        """
        return self.screenshot_paths.copy()

    def get_current_screenshot_path(self) -> Optional[str]:
        """
        获取当前最新截图路径

        Returns:
            当前最新截图文件路径，如果没有则返回None
        """
        return self.get_latest_screenshot_path()

    def get_screenshot_count(self) -> int:
        """
        获取截图总数

        Returns:
            截图总数
        """
        return len(self.screenshot_paths)

    def get_task_screenshot_directory(self) -> Optional[str]:
        """
        获取当前任务的截图目录路径

        Returns:
            截图目录路径，如果未初始化则返回None
        """
        if hasattr(self, 'task_screenshot_dir') and self.task_screenshot_dir:
            return str(self.task_screenshot_dir)
        return None

    def get_screenshot_info(self) -> Dict[str, Any]:
        """
        获取截图相关的详细信息

        Returns:
            包含截图信息的字典
        """
        return {
            "task_id": self.current_task_id,
            "current_step": self.current_step,
            "screenshot_directory": self.get_task_screenshot_directory(),
            "total_screenshots": self.get_screenshot_count(),
            "screenshot_paths": self.get_all_screenshots(),
            "latest_screenshot": self.get_latest_screenshot_path(),
        }

    def save_trajectory_data(self) -> Optional[str]:
        """
        保存轨迹数据到JSON文件

        Returns:
            保存的JSON文件路径，如果保存失败则返回None
        """
        try:
            if not hasattr(self, 'task_screenshot_dir') or not self.task_screenshot_dir:
                print("警告: 任务截图目录未初始化，无法保存轨迹数据")
                return None

            # 构建轨迹数据
            trajectory_data = {
                "task_id": self.current_task_id,
                "current_step": self.current_step,
                "timestamp": str(self._get_current_timestamp()),
                "trajectory": self._serialize_trajectory(),
                "screenshot_paths": self.screenshot_paths,
                "config_file": self.current_config_file,
            }

            # 保存到JSON文件
            trajectory_file = self.task_screenshot_dir / "trajectory.json"
            with open(trajectory_file, 'w', encoding='utf-8') as f:
                json.dump(trajectory_data, f, ensure_ascii=False, indent=2, default=str)

            print(f"成功保存轨迹数据: {trajectory_file}")
            return str(trajectory_file)

        except Exception as e:
            print(f"错误: 保存轨迹数据时发生异常: {e}")
            return None

    def _get_current_timestamp(self):
        """获取当前时间戳"""
        from datetime import datetime
        return datetime.now()

    def _serialize_trajectory(self) -> List[Dict[str, Any]]:
        """
        序列化轨迹数据，移除不可序列化的对象

        Returns:
            可序列化的轨迹数据列表
        """
        serialized_trajectory = []

        for item in self.current_trajectory:
            if isinstance(item, dict):
                # 这是一个状态信息
                serialized_item = {
                    "type": "state",
                    "url": item.get("url", ""),
                    "observation": {
                        "text": item.get("observation", {}).get("text", ""),
                        # 不保存图像数据到JSON中，因为已经保存为PNG文件
                        "image": None
                    },
                    "info": self._serialize_info(item.get("info", {}))
                }
            else:
                # 这是一个动作
                try:
                    serialized_item = {
                        "type": "action",
                        "action_type": getattr(item, 'action_type', str(type(item))),
                        "raw_prediction": getattr(item, 'raw_prediction', str(item)),
                        "parsed_prediction": getattr(item, 'parsed_prediction', None),
                    }
                except Exception:
                    serialized_item = {
                        "type": "action",
                        "data": str(item)
                    }

            serialized_trajectory.append(serialized_item)

        return serialized_trajectory

    def _serialize_info(self, info: Dict[str, Any]) -> Dict[str, Any]:
        """
        序列化info对象，移除不可序列化的部分

        Args:
            info: 原始info字典

        Returns:
            可序列化的info字典
        """
        serialized_info = {}

        for key, value in info.items():
            if key == "page":
                # 页面对象只保存URL
                try:
                    serialized_info[key] = {"url": getattr(value, 'url', str(value))}
                except Exception:
                    serialized_info[key] = {"url": str(value)}
            elif isinstance(value, (str, int, float, bool, type(None))):
                serialized_info[key] = value
            elif isinstance(value, (list, dict)):
                try:
                    # 尝试序列化，如果失败就转为字符串
                    json.dumps(value)
                    serialized_info[key] = value
                except Exception:
                    serialized_info[key] = str(value)
            else:
                serialized_info[key] = str(value)

        return serialized_info