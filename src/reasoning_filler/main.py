"""
轨迹推理填充主程序

整合所有模块，提供批量处理功能
"""

import argparse
import os
import sys
import time
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Dict, Any

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.reasoning_filler.trajectory_loader import TrajectoryLoader, TrajectoryInfo
from src.reasoning_filler.prompt_builder import PromptBuilder
from src.reasoning_filler.qwen_caller import QwenCaller
from src.reasoning_filler.gpt4o_caller import GPT4oCaller
from src.reasoning_filler.trajectory_filler import TrajectoryFiller
from src.reasoning_filler.progress_tracker import ProgressTracker


class ReasoningFillerApp:
    """轨迹推理填充应用"""

    def __init__(
        self,
        api_key: str,
        model_name: str = "qwen-vl-plus",
        input_dir: str = "data/annotate/trajectories",
        output_dir: str = "data/annotate_with_reasoning",
        progress_file: str = "data/annotate_with_reasoning/progress.json",
        prompt_style: str = "LIFT",
        enable_validation: bool = True,
        max_retry_attempts: int = 2,
        gpt4o_config: Optional[Dict[str, Any]] = None,
        visualization_config: Optional[Dict[str, Any]] = None,
        reasoning_config: Optional[Dict[str, Any]] = None
    ):
        """
        初始化应用

        Args:
            api_key: DashScope API Key (用于 Qwen)
            model_name: Qwen 模型名称
            input_dir: 输入目录
            output_dir: 输出目录
            progress_file: 进度文件路径
            prompt_style: Prompt 风格
            enable_validation: 是否启用action验证
            max_retry_attempts: 最大重试次数
            gpt4o_config: GPT-4o 配置字典
            visualization_config: Prompt可视化配置字典
            reasoning_config: 推理生成器配置字典
        """
        print("=== 初始化轨迹推理填充系统 ===")

        # 保存配置参数（用于创建独立的 filler 实例）
        self.api_key = api_key
        self.model_name = model_name
        self.output_dir = output_dir
        self.enable_validation = enable_validation
        self.max_retry_attempts = max_retry_attempts
        self.gpt4o_config = gpt4o_config
        self.visualization_config = visualization_config
        self.reasoning_config = reasoning_config

        # 初始化各模块
        print("初始化模块...")
        self.loader = TrajectoryLoader(input_dir)
        self.prompt_builder = PromptBuilder(prompt_style=prompt_style)

        # 打印推理生成器信息
        reasoning_generator = reasoning_config.get('generator', 'qwen') if reasoning_config else 'qwen'
        if reasoning_generator == 'gpt4o':
            print(f"  推理生成器: GPT-4o ({gpt4o_config.get('model_name', 'gpt-4o')})")
        else:
            print(f"  推理生成器: Qwen ({model_name})")

        # 创建共享的 filler（用于单线程模式）
        self.filler = self._create_filler()

        self.tracker = ProgressTracker(progress_file=progress_file)

        print("✓ 模块初始化完成")
        if enable_validation:
            print(f"  验证模式: 启用（最多{max_retry_attempts + 1}次尝试）")
            if gpt4o_config and gpt4o_config.get("enabled", False):
                print(f"  GPT-4o 反馈: 已启用")
        else:
            print("  验证模式: 禁用")
        print()

    def _create_filler(self) -> TrajectoryFiller:
        """
        创建新的 TrajectoryFiller 实例

        用于并行处理时为每个轨迹创建独立的 filler，避免线程竞态条件

        Returns:
            TrajectoryFiller 实例
        """
        # 根据配置选择推理生成器
        reasoning_generator = self.reasoning_config.get('generator', 'qwen') if self.reasoning_config else 'qwen'

        if reasoning_generator == 'gpt4o':
            # 使用 GPT-4o 生成推理
            if not self.gpt4o_config or not self.gpt4o_config.get('api_key'):
                raise ValueError("使用 GPT-4o 生成推理需要配置 gpt4o.api_key")

            reasoning_caller = GPT4oCaller(
                api_key=self.gpt4o_config.get('api_key'),
                base_url=self.gpt4o_config.get('base_url'),
                model_name=self.gpt4o_config.get('model_name', 'gpt-4o'),
                max_retries=self.gpt4o_config.get('max_retries', 3),
                timeout=self.gpt4o_config.get('timeout', 60)
            )
        else:
            # 使用 Qwen 生成推理（默认）
            reasoning_caller = QwenCaller(api_key=self.api_key, model_name=self.model_name)

        return TrajectoryFiller(
            reasoning_caller=reasoning_caller,
            prompt_builder=self.prompt_builder,
            output_dir=self.output_dir,
            enable_validation=self.enable_validation,
            max_retry_attempts=self.max_retry_attempts,
            gpt4o_config=self.gpt4o_config,
            visualization_config=self.visualization_config
        )

    def process_single_trajectory(
        self,
        traj_info: TrajectoryInfo,
        skip_if_processed: bool = True,
        use_independent_filler: bool = False
    ) -> bool:
        """
        处理单个轨迹

        Args:
            traj_info: 轨迹信息
            skip_if_processed: 如果已处理，是否跳过
            use_independent_filler: 是否使用独立的 filler 实例（用于并行处理）

        Returns:
            是否处理成功
        """
        file_rel_path = f"{traj_info.env_name}/{traj_info.trajectory_file.name}"

        # 检查是否已处理
        if skip_if_processed and self.tracker.is_processed(file_rel_path):
            print(f"⊙ 跳过已处理: {file_rel_path}")
            return True

        # 检查是否之前失败过
        if self.tracker.is_failed(file_rel_path):
            print(f"⚠ 之前处理失败: {file_rel_path}")
            retry = input("是否重试？(y/n): ")
            if retry.lower() != 'y':
                return False

        print(f"\n>>> 处理轨迹: {file_rel_path}")

        # 为并行处理创建独立的 filler 实例，避免线程竞态条件
        filler = self._create_filler() if use_independent_filler else self.filler

        try:
            # 加载轨迹
            print("  [1/3] 加载轨迹数据...")
            trajectory, metadata = self.loader.load_trajectory(traj_info)
            print(f"  ✓ 轨迹长度: {len(trajectory)}")

            # 填充推理
            print("  [2/3] 生成推理内容...")
            filled_trajectory, num_filled = filler.fill_trajectory(
                trajectory, metadata, verbose=True
            )
            print(f"  ✓ 已填充 {num_filled} 个动作")

            # 保存结果
            print("  [3/3] 保存结果...")
            saved_path = filler.save_filled_trajectory(
                filled_trajectory,
                metadata,
                traj_info.env_name,
                traj_info.trajectory_file.name
            )
            print(f"  ✓ 已保存到: {saved_path}")

            # 更新验证统计
            self.tracker.update_validation_stats(filler.validation_stats)

            # 标记为已处理
            self.tracker.mark_processed(file_rel_path, num_filled)
            self.tracker.save_progress()

            print(f"✓ 处理成功!\n")
            return True

        except KeyboardInterrupt:
            print("\n用户中断")
            raise

        except Exception as e:
            print(f"✗ 处理失败: {e}\n")
            self.tracker.mark_failed(file_rel_path, str(e))
            self.tracker.save_progress()
            return False

    def process_batch(
        self,
        env_filter: Optional[str] = None,
        max_count: Optional[int] = None,
        skip_if_processed: bool = True,
        max_workers: int = 5
    ):
        """
        批量处理轨迹（支持并行）

        Args:
            env_filter: 环境过滤
            max_count: 最大处理数量
            skip_if_processed: 是否跳过已处理的
            max_workers: 并行工作线程数（设为 1 则顺序执行）
        """
        print("=== 开始批量处理 ===\n")

        # 扫描轨迹
        print("扫描轨迹文件...")
        all_trajectories = self.loader.scan_trajectories(env_filter=env_filter)
        print(f"找到 {len(all_trajectories)} 个轨迹文件\n")

        # 限制数量
        if max_count:
            all_trajectories = all_trajectories[:max_count]
            print(f"处理前 {max_count} 个轨迹\n")

        # 过滤已处理的轨迹
        trajectories_to_process = []
        skipped_count = 0

        for traj_info in all_trajectories:
            file_rel_path = f"{traj_info.env_name}/{traj_info.trajectory_file.name}"
            if skip_if_processed and self.tracker.is_processed(file_rel_path):
                skipped_count += 1
            else:
                trajectories_to_process.append(traj_info)

        if skipped_count > 0:
            print(f"跳过 {skipped_count} 个已处理的轨迹\n")

        total = len(all_trajectories)
        to_process = len(trajectories_to_process)

        if to_process == 0:
            print("没有需要处理的轨迹！\n")
            return

        print(f"将并行处理 {to_process} 个轨迹（并发数: {max_workers}）\n")

        # 统计
        success_count = 0
        failed_count = 0
        completed_count = 0
        start_time = time.time()

        # 并行处理
        if max_workers == 1:
            # 顺序执行模式
            print("使用顺序执行模式\n")
            for idx, traj_info in enumerate(trajectories_to_process, 1):
                print(f"[{completed_count + skipped_count + 1}/{total}] 进度: {(completed_count + skipped_count + 1) / total * 100:.1f}%")

                success = self.process_single_trajectory(traj_info, skip_if_processed=False)
                completed_count += 1

                if success:
                    success_count += 1
                else:
                    failed_count += 1

                # 每 5 个文件保存一次进度
                if completed_count % 5 == 0:
                    print("💾 保存进度...")
                    self.tracker.save_progress()
        else:
            # 并行执行模式
            print("使用并行执行模式\n")
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # 提交所有任务（为每个任务创建独立的 filler 实例）
                future_to_traj = {
                    executor.submit(self.process_single_trajectory, traj_info, False, True): traj_info
                    for traj_info in trajectories_to_process
                }

                # 处理完成的任务
                for future in as_completed(future_to_traj):
                    traj_info = future_to_traj[future]
                    completed_count += 1

                    try:
                        success = future.result()
                        if success:
                            success_count += 1
                            print(f"✓ [{completed_count + skipped_count}/{total}] 成功: {traj_info.env_name}/{traj_info.trajectory_file.name}")
                        else:
                            failed_count += 1
                            print(f"✗ [{completed_count + skipped_count}/{total}] 失败: {traj_info.env_name}/{traj_info.trajectory_file.name}")
                    except Exception as e:
                        failed_count += 1
                        print(f"✗ [{completed_count + skipped_count}/{total}] 异常: {traj_info.env_name}/{traj_info.trajectory_file.name} - {e}")

                    # 每 5 个文件保存一次进度
                    if completed_count % 5 == 0:
                        print("💾 保存进度...")
                        self.tracker.save_progress()

        # 最终保存
        self.tracker.save_progress()

        # 统计报告
        elapsed_time = time.time() - start_time
        self.print_final_report(total, success_count, failed_count, skipped_count, elapsed_time)

    def print_final_report(
        self,
        total: int,
        success: int,
        failed: int,
        skipped: int,
        elapsed_time: float
    ):
        """打印最终报告"""
        print("\n" + "=" * 70)
        print("处理完成!")
        print("=" * 70)
        print(f"总文件数: {total}")
        print(f"成功: {success}")
        print(f"失败: {failed}")
        print(f"跳过: {skipped}")
        print(f"耗时: {elapsed_time:.2f} 秒")
        print("=" * 70)

        # 打印进度摘要
        self.tracker.print_summary()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="轨迹推理填充工具")

    parser.add_argument(
        "--api_key",
        type=str,
        default='sk-667dbe0bc8f74a0ba832a2b0611f2d08',
        help="DashScope API Key（或设置环境变量 DASHSCOPE_API_KEY）"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="qwen-vl-max",
        help="Qwen 模型名称"
    )

    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/annotate/trajectories",
        help="输入目录"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/annotate_with_reasoning",
        help="输出目录"
    )

    parser.add_argument(
        "--env",
        type=str,
        default=None,
        help="环境过滤（如 classifieds）"
    )

    parser.add_argument(
        "--max_count",
        type=int,
        help="最大处理数量"
    )

    parser.add_argument(
        "--prompt_style",
        type=str,
        default="LIFT",
        choices=["LIFT", "ORIGINAL"],
        help="Prompt 风格"
    )

    parser.add_argument(
        "--reset_progress",
        action="store_true",
        help="重置进度记录"
    )

    parser.add_argument(
        "--max_workers",
        type=int,
        default=5,
        help="并行工作线程数（默认 5，设为 1 则顺序执行）"
    )

    parser.add_argument(
        "--enable_validation",
        action="store_true",
        default=True,
        help="启用action验证（默认启用）"
    )

    parser.add_argument(
        "--disable_validation",
        action="store_true",
        help="禁用action验证"
    )

    parser.add_argument(
        "--max_retry_attempts",
        type=int,
        default=2,
        help="最大重试次数（默认2次，总共3次尝试）"
    )

    args = parser.parse_args()

    # 处理验证开关
    enable_validation = not args.disable_validation if args.disable_validation else args.enable_validation

    # 获取 API key
    api_key = args.api_key or os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        print("错误: 请提供 API Key（--api_key 或环境变量 DASHSCOPE_API_KEY）")
        sys.exit(1)

    # 读取 config.yaml（用于推理生成器、GPT-4o 和可视化配置）
    config_path = Path(__file__).parent / "config.yaml"
    reasoning_config = None
    gpt4o_config = None
    visualization_config = None
    if config_path.exists():
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                reasoning_config = config.get('reasoning', {})
                gpt4o_config = config.get('gpt4o', {})
                visualization_config = config.get('visualization', {})

                # 打印配置信息
                generator = reasoning_config.get('generator', 'qwen') if reasoning_config else 'qwen'
                print(f"✓ 推理生成器: {generator}")

                if gpt4o_config.get('enabled', False):
                    print(f"✓ GPT-4o 错误分析: 已启用")

                if visualization_config.get('enable_prompt_visualization', False):
                    print(f"✓ Prompt 可视化: 已启用")
        except Exception as e:
            print(f"Warning: 读取 config.yaml 失败: {e}")

    # 初始化应用
    app = ReasoningFillerApp(
        api_key=api_key,
        model_name=args.model,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        prompt_style=args.prompt_style,
        enable_validation=enable_validation,
        max_retry_attempts=args.max_retry_attempts,
        gpt4o_config=gpt4o_config,
        visualization_config=visualization_config,
        reasoning_config=reasoning_config
    )

    # 重置进度（如果需要）
    if args.reset_progress:
        print("重置进度记录...")
        app.tracker.reset()
        print("✓ 进度已重置\n")

    # 批量处理
    try:
        app.process_batch(
            env_filter=args.env,
            max_count=args.max_count,
            skip_if_processed=True,
            max_workers=args.max_workers
        )
    except KeyboardInterrupt:
        print("\n\n用户中断，保存进度...")
        app.tracker.save_progress()
        print("进度已保存，下次运行将从中断处继续")


if __name__ == "__main__":
    main()
