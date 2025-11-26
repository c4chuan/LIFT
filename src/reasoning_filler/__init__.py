"""
轨迹推理填充系统

该模块用于为标注的轨迹数据补充推理过程。
通过调用 Qwen 模型，给定任务目标、历史信息和 ground truth 动作，
生成从观察到动作的推理过程，并填充到 Action 的 raw_prediction 字段中。

主要模块：
- trajectory_loader: 轨迹数据读取
- prompt_builder: Prompt 构造
- qwen_caller: Qwen API 调用
- trajectory_filler: 轨迹填充与保存
- progress_tracker: 断点重续管理
- main: 主程序入口
"""

__version__ = "1.0.0"
__author__ = "LIFT Team"
