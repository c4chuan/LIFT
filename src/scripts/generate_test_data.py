"""
参数化的测试数据生成器
通过命令行参数或配置文件替换网站占位符并生成测试数据
"""
import json
import os
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Any


def load_config(config_path: str) -> Dict[str, Dict[str, str]]:
    """加载配置文件"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"错误: 配置文件 {config_path} 不存在")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"错误: 配置文件格式无效: {e}")
        sys.exit(1)


def get_input_paths(dataset: str, input_dir: str) -> List[str]:
    """根据数据集类型获取输入文件路径"""
    if dataset == "webarena":
        return [os.path.join(input_dir, "wa/test_webarena.raw.json")]
    elif dataset == "visualwebarena":
        return [
            os.path.join(input_dir, "vwa/test_classifieds.raw.json"),
            os.path.join(input_dir, "vwa/test_shopping.raw.json"),
            os.path.join(input_dir, "vwa/test_reddit.raw.json"),
        ]
    elif dataset == "annotate":
        return [
            os.path.join(input_dir, "annotate/shopping_tasks.json"),
            os.path.join(input_dir, "annotate/reddit_tasks.json"),
            os.path.join(input_dir, "annotate/classifieds_tasks.json"),
        ]
    else:
        raise ValueError(f"不支持的数据集: {dataset}")


def get_replace_map(dataset: str, config: Dict[str, Dict[str, str]]) -> Dict[str, str]:
    """根据数据集类型获取替换映射"""
    if dataset not in config:
        raise ValueError(f"配置文件中缺少数据集 '{dataset}' 的配置")

    dataset_config = config[dataset]

    if dataset == "webarena":
        return {
            "__REDDIT__": dataset_config.get("REDDIT", ""),
            "__SHOPPING__": dataset_config.get("SHOPPING", ""),
            "__SHOPPING_ADMIN__": dataset_config.get("SHOPPING_ADMIN", ""),
            "__GITLAB__": dataset_config.get("GITLAB", ""),
            "__WIKIPEDIA__": dataset_config.get("WIKIPEDIA", ""),
            "__MAP__": dataset_config.get("MAP", ""),
            "__HOMEPAGE__": dataset_config.get("HOMEPAGE", ""),
        }
    elif dataset == "visualwebarena":
        return {
            "__REDDIT__": dataset_config.get("REDDIT", ""),
            "__SHOPPING__": dataset_config.get("SHOPPING", ""),
            "__WIKIPEDIA__": dataset_config.get("WIKIPEDIA", ""),
            "__CLASSIFIEDS__": dataset_config.get("CLASSIFIEDS", ""),
            "__HOMEPAGE__": dataset_config.get("HOMEPAGE", ""),
        }
    elif dataset == "annotate":
        return {
            "__REDDIT__": dataset_config.get("REDDIT", ""),
            "__SHOPPING__": dataset_config.get("SHOPPING", ""),
            "__CLASSIFIEDS__": dataset_config.get("CLASSIFIEDS", ""),
        }
    else:
        raise ValueError(f"不支持的数据集: {dataset}")


def process_file(inp_path: str, replace_map: Dict[str, str], verbose: bool = False, is_annotate: bool = False) -> None:
    """处理单个输入文件"""
    if not os.path.exists(inp_path):
        print(f"警告: 输入文件 {inp_path} 不存在，跳过")
        return

    if verbose:
        print(f"处理文件: {inp_path}")

    try:
        # 读取原始文件
        with open(inp_path, "r", encoding='utf-8') as f:
            raw = f.read()

        # 执行替换
        replaced_count = 0
        for placeholder, replacement in replace_map.items():
            if replacement:  # 只有在有值的情况下才替换
                count_before = raw.count(placeholder)
                raw = raw.replace(placeholder, replacement)
                if count_before > 0:
                    replaced_count += count_before
                    if verbose:
                        print(f"  替换 {placeholder} -> {replacement} ({count_before}次)")

        if verbose:
            print(f"  总共替换了 {replaced_count} 个占位符")

        # 对于 annotate 数据集，直接替换原文件
        if is_annotate:
            with open(inp_path, "w", encoding='utf-8') as f:
                f.write(raw)
            if verbose:
                print(f"  已更新文件: {inp_path}")
        else:
            # 原有逻辑：创建输出目录和生成单独文件
            output_dir = inp_path.replace('.raw.json', '')
            os.makedirs(output_dir, exist_ok=True)

            # 写入处理后的文件
            output_path = inp_path.replace(".raw", "")
            with open(output_path, "w", encoding='utf-8') as f:
                f.write(raw)

            if verbose:
                print(f"  生成文件: {output_path}")

            # 解析JSON并生成单独的文件
            try:
                data = json.loads(raw)
                for idx, item in enumerate(data):
                    item_path = os.path.join(output_dir, f"{idx}.json")
                    with open(item_path, "w", encoding='utf-8') as f:
                        json.dump(item, f, indent=2, ensure_ascii=False)

                if verbose:
                    print(f"  生成 {len(data)} 个单独的JSON文件到 {output_dir}")

            except json.JSONDecodeError as e:
                print(f"警告: 无法解析JSON文件 {output_path}: {e}")

    except Exception as e:
        print(f"错误: 处理文件 {inp_path} 时出现异常: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="参数化的测试数据生成器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python generate_test_data.py --config config.json --dataset annotate
  python generate_test_data.py --config config.json --dataset visualwebarena --input-dir config_files
  python generate_test_data.py --config config.json --dataset webarena --input-dir config_files --verbose

配置文件格式:
{
  "webarena": {
    "REDDIT": "http://reddit.example.com",
    "SHOPPING": "http://shopping.example.com",
    "SHOPPING_ADMIN": "http://shopping-admin.example.com",
    "GITLAB": "http://gitlab.example.com",
    "WIKIPEDIA": "http://wikipedia.example.com",
    "MAP": "http://map.example.com",
    "HOMEPAGE": "http://homepage.example.com"
  },
  "visualwebarena": {
    "REDDIT": "http://reddit.example.com",
    "SHOPPING": "http://shopping.example.com",
    "WIKIPEDIA": "http://wikipedia.example.com",
    "CLASSIFIEDS": "http://classifieds.example.com",
    "HOMEPAGE": "http://homepage.example.com"
  },
  "annotate": {
    "REDDIT": "http://127.0.0.1:9999",
    "SHOPPING": "http://127.0.0.1:7770",
    "CLASSIFIEDS": "http://127.0.0.1:9980"
  }
}
        """
    )

    parser.add_argument(
        "--config", "-c",
        required=True,
        help="包含网站域名映射的JSON配置文件路径"
    )

    parser.add_argument(
        "--dataset", "-d",
        choices=["webarena", "visualwebarena", "annotate"],
        required=True,
        help="数据集类型"
    )

    parser.add_argument(
        "--input-dir", "-i",
        default="data",
        help="输入配置文件目录 (默认: data)"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="启用详细输出模式"
    )

    args = parser.parse_args()

    try:
        # 加载配置
        config = load_config(args.config)

        # 显示数据集信息
        print(f"数据集: {args.dataset}")

        # 获取替换映射
        replace_map = get_replace_map(args.dataset, config)

        if args.verbose:
            print("网站映射:")
            for key, value in replace_map.items():
                print(f"  {key}: {value}")

        # 获取输入文件路径
        inp_paths = get_input_paths(args.dataset, args.input_dir)

        # 处理每个文件
        is_annotate = args.dataset == "annotate"
        for inp_path in inp_paths:
            process_file(inp_path, replace_map, args.verbose, is_annotate)

        print(f"完成! 处理了 {len(inp_paths)} 个文件")

    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()