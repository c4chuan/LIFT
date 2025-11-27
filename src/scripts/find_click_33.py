#!/usr/bin/env python3
"""
脚本功能：在指定数据目录下查找包含特定字符串（或正则表达式）的文件
输出：文件编号列表及每个文件中的匹配数量
支持过滤 valid_action_reward == 0.3 的条目
支持正则表达式匹配模式
"""

import json
import re
from pathlib import Path
from collections import defaultdict


def find_click_33(data_dir: str = "sup_rollout_data_dir_1113_1",
                  filter_valid_action: bool = True,
                  search_string: str = "click [33]",
                  use_regex: bool = False,
                  reward_threshold: float = 0.3):
    """
    扫描数据目录，查找包含指定字符串（或正则表达式）的文件

    Args:
        data_dir: 数据目录名称（相对于项目根目录）
        filter_valid_action: 是否过滤 valid_action_reward 的条目（默认 True）
        search_string: 要搜索的字符串或正则表达式（默认 "click [33]"）
        use_regex: 是否使用正则表达式匹配（默认 False）
        reward_threshold: valid_action_reward 的阈值（默认 0.3）
    """
    # 获取项目根目录（假设脚本在 src/scripts/ 下）
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    target_dir = project_root / data_dir

    # 检查目录是否存在
    if not target_dir.exists():
        print(f"错误：目录不存在 - {target_dir}")
        return

    print(f"扫描目录: {data_dir}")
    match_mode = "正则表达式" if use_regex else "字符串包含"
    if filter_valid_action:
        print(f"过滤条件: output 包含 '{search_string}' ({match_mode}) 且 valid_action_reward == {reward_threshold}\n")
    else:
        print(f"过滤条件: output 包含 '{search_string}' ({match_mode})\n")

    # 统计结果
    file_matches = defaultdict(int)  # 文件名 -> 匹配数量
    total_click_33 = 0  # 总共多少条匹配
    total_files = 0
    error_files = []

    # 遍历目标目录中的所有 JSONL 文件
    jsonl_files = sorted(target_dir.glob("*.jsonl"))

    for jsonl_file in jsonl_files:
        # 提取文件名（不含扩展名）作为标识符
        file_identifier = jsonl_file.stem

        total_files += 1

        try:
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        data = json.loads(line)
                        output = data.get('output', '')

                        # 检查 output 中是否包含搜索字符串
                        matched = False
                        if use_regex:
                            # 使用正则表达式匹配
                            if re.search(search_string, output):
                                matched = True
                        else:
                            # 使用字符串包含检查
                            if search_string in output:
                                matched = True

                        if matched:
                            total_click_33 += 1

                            # 如果需要过滤 valid_action_reward
                            if filter_valid_action:
                                valid_action_reward = data.get('valid_action_reward', None)
                                if valid_action_reward == reward_threshold:
                                    file_matches[file_identifier] += 1
                            else:
                                file_matches[file_identifier] += 1

                    except json.JSONDecodeError as e:
                        # 跳过无效的 JSON 行
                        continue

        except Exception as e:
            error_files.append((file_identifier, str(e)))

    print(f"总文件数: {total_files}\n")

    # 输出结果
    if file_matches:
        sorted_files = sorted(file_matches.items())

        if filter_valid_action:
            print(f"找到 {len(sorted_files)} 个文件包含 '{search_string}' 且 valid_action_reward == {reward_threshold}:\n")
        else:
            print(f"找到 {len(sorted_files)} 个文件包含 '{search_string}':\n")

        total_matches = 0
        for file_name, count in sorted_files:
            print(f"文件 {file_name}: {count} 条匹配")
            total_matches += count

        if filter_valid_action:
            percentage = (total_matches / total_click_33 * 100) if total_click_33 > 0 else 0
            print(f"\n总计：{len(sorted_files)} 个文件，共 {total_matches} 条符合条件的记录")
            print(f"（总共 {total_click_33} 条 '{search_string}'，其中 {total_matches} 条 valid_action_reward == {reward_threshold}，占比 {percentage:.1f}%）")
        else:
            print(f"\n总计：{len(sorted_files)} 个文件，共 {total_matches} 条包含 '{search_string}' 的记录")

        # 输出简洁的文件名列表
        file_names = [str(file_name) for file_name, _ in sorted_files]
        print(f"\n文件列表: {', '.join(file_names)}")
    else:
        if filter_valid_action:
            print(f"未找到符合条件的文件（总共 {total_click_33} 条 '{search_string}'，但没有 valid_action_reward == {reward_threshold} 的）")
        else:
            print(f"未找到包含 '{search_string}' 的文件")

    # 输出错误信息（如果有）
    if error_files:
        print(f"\n警告：以下文件处理时出错：")
        for file_name, error in error_files:
            print(f"  文件 {file_name}: {error}")


if __name__ == "__main__":
    import sys
    import argparse

    # 命令行参数解析
    parser = argparse.ArgumentParser(
        description='查找包含指定字符串（或正则表达式）的文件，可选过滤 valid_action_reward'
    )
    parser.add_argument(
        'data_dir',
        nargs='?',
        default='rollout_data/sup_rollout_wo_obs_1126_d',
        help='数据目录名称（默认: sup_rollout_wo_obs_1126_d）'
    )
    parser.add_argument(
        '--search-string',
        default='click [33]',
        help='要搜索的字符串或正则表达式（默认: "click [33]"）'
    )
    parser.add_argument(
        '--regex',
        action='store_true',
        help='使用正则表达式模式匹配'
    )
    parser.add_argument(
        '--reward-threshold',
        type=float,
        default=1,
        help='valid_action_reward 的阈值（默认: 0.3）'
    )
    parser.add_argument(
        '--no-filter',
        action='store_true',
        help='不过滤 valid_action_reward，显示所有包含搜索字符串的记录'
    )

    args = parser.parse_args()

    # 默认启用过滤，除非指定 --no-filter
    filter_valid_action = not args.no_filter

    find_click_33(
        data_dir=args.data_dir,
        filter_valid_action=filter_valid_action,
        search_string=args.search_string,
        use_regex=args.regex,
        reward_threshold=args.reward_threshold
    )
