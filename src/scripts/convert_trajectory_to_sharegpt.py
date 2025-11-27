#!/usr/bin/env python3
"""
将轨迹数据转换为ShareGPT格式的微调数据

使用方法:
    python src/scripts/convert_trajectory_to_sharegpt.py \
        --data_dir data/annotate_with_reasoning \
        --environments classifieds \
        --output_dir LLaMA-Factory/data \
        --output_filename trajectory_finetune_data.json \
        --include_system_message
"""

import os
import sys
import json
import argparse
import pickle
import lzma
from pathlib import Path
from typing import List, Dict, Any, Union
import numpy as np
from PIL import Image
import cv2
from vwa.src.helper_functions import get_action_description
# 添加项目根目录到sys.path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 添加visualwebarena路径，以便pickle能找到browser_env模块
vwa_path = project_root / "visualwebarena"
if vwa_path.exists():
    sys.path.insert(0, str(vwa_path))

import re


# 创建mock模块以解决导入问题
class MockModule:
    """Mock模块，用于处理缺失的导入"""
    def __init__(self, name):
        self.name = name

    def __getattr__(self, attr):
        return MockModule(f"{self.name}.{attr}")

    def __call__(self, *args, **kwargs):
        return MockModule(f"{self.name}()")

    def __iter__(self):
        return iter([])

    def __len__(self):
        return 0

    def __getitem__(self, key):
        return MockModule(f"{self.name}[{key}]")

    def __mro_entries__(self, bases):
        # 当MockModule被用作基类时，返回object作为实际基类
        return (object,)


# 设置必要的环境变量
os.environ.setdefault('DATASET', 'visualwebarena')
os.environ.setdefault('SHOPPING', 'NA')
os.environ.setdefault('SHOPPING_ADMIN', 'NA')
os.environ.setdefault('REDDIT', 'NA')
os.environ.setdefault('WIKIPEDIA', 'NA')
os.environ.setdefault('MAP', 'NA')
os.environ.setdefault('HOMEPAGE', 'NA')
os.environ.setdefault('CLASSIFIEDS', 'NA')
os.environ.setdefault('CLASSIFIEDS_RESET_TOKEN', 'NA')

# 在导入前先mock掉可能缺失的模块及其子模块
# 注意：不要mock requests和tqdm，因为它们会影响其他库的导入（如huggingface_hub）
missing_modules = [
    'openai', 'anthropic', 'vertexai'
]
for module_name in missing_modules:
    if module_name not in sys.modules:
        sys.modules[module_name] = MockModule(module_name)

# 单独处理requests - 如果不存在，创建一个更完善的mock
try:
    import requests
except ImportError:
    # 创建一个requests mock，包含exceptions子模块
    class RequestsMock(MockModule):
        pass

    requests_mock = RequestsMock('requests')

    # 为requests.exceptions创建真实的异常类
    class RequestsExceptions:
        class HTTPError(Exception):
            pass
        class ConnectionError(Exception):
            pass
        class Timeout(Exception):
            pass
        class RequestException(Exception):
            pass

    requests_mock.exceptions = RequestsExceptions()
    sys.modules['requests'] = requests_mock
    sys.modules['requests.exceptions'] = RequestsExceptions()

# 修复anyio.abc的问题
try:
    import anyio
    if not hasattr(anyio, 'abc'):
        anyio.abc = MockModule('anyio.abc')
except:
    pass

# 预导入browser_env模块，确保pickle能正确反序列化
try:
    from browser_env import actions
    from browser_env.actions import ActionTypes, Action
    from browser_env.utils import StateInfo
    print("成功导入browser_env模块")
except Exception as e:
    print(f"警告: 导入browser_env失败: {e}")
    # 继续执行，使用RestrictedUnpickler处理

# 导入EnvLIFTConstructor及其依赖
try:
    from src.agentic.policy import EnvLIFTConstructor,SFTDataConstructor
    from llms import lm_config
    from src.llms.tokenizer import Tokenizer
    print("成功导入EnvLIFTConstructor及其依赖")
except Exception as e:
    print(f"警告: 导入EnvLIFTConstructor失败: {e}")
    import traceback
    traceback.print_exc()
    EnvLIFTConstructor = None


class RestrictedUnpickler(pickle.Unpickler):
    """自定义Unpickler，处理缺失的模块导入"""

    def find_class(self, module, name):
        # 尝试正常导入
        try:
            return super().find_class(module, name)
        except (ModuleNotFoundError, AttributeError, TypeError) as e:
            # 如果导入失败，返回一个简单的类
            # print(f"警告: 无法导入 {module}.{name}，使用占位符")
            # 创建一个简单的占位类
            return type(name, (), {})


def load_trajectory_files(classifieds_path, task_ids=None):
    """
    读取classifieds目录下的pkl.xz文件（简化版本）

    Args:
        classifieds_path (str): 存放轨迹文件的目录路径
        task_ids (list, optional): 指定要读取的任务编号列表。如果为None，则读取所有任务

    Returns:
        list: 包含轨迹数据的字典列表
    """
    if not os.path.exists(classifieds_path):
        print(f"警告: 目录 {classifieds_path} 不存在")
        return []

    # 如果没有指定task_ids，则提取所有任务编号
    if task_ids is None:
        task_ids_set = set()
        for filename in os.listdir(classifieds_path):
            # 匹配 classifieds_11_20250930_154359.pkl.xz 格式
            match = re.match(r'(\w+)_(\d+)_\d+_\d+\.pkl\.xz$', filename)
            if match:
                task_ids_set.add(int(match.group(2)))
        task_ids = sorted(list(task_ids_set))

    # 为每个task_id找到对应的pkl.xz文件（取最新的）
    trajectory_data = []
    for task_id in task_ids:
        # 查找该task_id的所有pkl.xz文件
        pattern = re.compile(rf'\w+_{task_id}_\d+_\d+\.pkl\.xz$')
        matching_files = []

        for filename in os.listdir(classifieds_path):
            if pattern.match(filename):
                matching_files.append(filename)

        if not matching_files:
            continue

        # 取最新的文件（按文件名排序，最后一个是最新的）
        latest_file = sorted(matching_files)[-1]
        file_path = os.path.join(classifieds_path, latest_file)

        # 读取pkl.xz文件
        try:
            with lzma.open(file_path, 'rb') as f:
                # 直接使用标准pickle.load，因为我们已经导入了browser_env模块
                trajectory = pickle.load(f)

            trajectory_data.append({
                'task_id': task_id,
                'trajectory': trajectory,
                'file_path': file_path
            })

        except Exception as e:
            print(f"错误: 无法读取task_id={task_id}的文件 {latest_file}: {str(e)}")
            # import traceback
            # traceback.print_exc()
            continue

    return trajectory_data


def save_image(image_array: np.ndarray, save_path: str) -> None:
    """
    保存numpy图像数组为PNG文件

    Args:
        image_array: numpy数组，形状为(H, W, 3)，RGB格式
        save_path: 保存路径
    """
    # 确保目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 使用PIL保存（不需要RGB转BGR）
    img = Image.fromarray(image_array.astype(np.uint8))
    img.save(save_path, 'PNG')


def create_env_lift_constructor(output_dir: str, environment: str, instruction_path: str = None, use_raw_prediction: bool = True) -> Any:
    """
    创建EnvLIFTConstructor实例

    Args:
        output_dir: 输出目录，用于保存图像
        environment: 环境名称（如'classifieds'）
        instruction_path: 配置文件路径（相对于项目根目录或绝对路径），默认为lift.json
        use_raw_prediction: 是否使用raw_prediction作为动作文本，默认为True

    Returns:
        EnvLIFTConstructor实例，如果导入失败则返回None
    """
    if EnvLIFTConstructor is None:
        print("警告: EnvLIFTConstructor未成功导入，将使用fallback方法")
        return None

    try:
        # 设置instruction_path - 配置文件路径
        if instruction_path:
            instruction_path = Path(instruction_path)
            if not instruction_path.is_absolute():
                instruction_path = project_root / instruction_path
        else:
            instruction_path = project_root / "visualwebarena" / "src" / "prompts" / "vwa" / "jsons" / "lift.json"

        if not instruction_path.exists():
            print(f"警告: lift.json配置文件不存在: {instruction_path}")
            return None

        # 创建dummy的LMConfig（不实际调用LLM，仅用于格式化）
        lm_cfg = lm_config.LMConfig(
            provider="openai",
            model="gpt-4o",
            mode="chat",
            gen_config={
                "temperature": 0.7,
                "max_obs_length": 2000
            }
        )

        # 创建Tokenizer（用于处理文本长度）
        tokenizer = Tokenizer(provider="openai", model_name="gpt-4o")

        # 为该环境创建专门的图片保存目录
        env_image_dir = os.path.join(output_dir, f"{environment}_trajectory_images")
        os.makedirs(env_image_dir, exist_ok=True)

        # 初始化EnvLIFTConstructor
        constructor = SFTDataConstructor(
            instruction_path=str(instruction_path),
            lm_config=lm_cfg,
            tokenizer=tokenizer,
            save_dir=env_image_dir,
            use_raw_prediction=use_raw_prediction
        )

        print("✓ EnvLIFTConstructor初始化成功")
        return constructor

    except Exception as e:
        print(f"警告: 初始化EnvLIFTConstructor失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def convert_envlift_to_sharegpt(envlift_messages: List[Dict], output_dir: str, environment: str) -> Dict[str, Any]:
    """
    将EnvLIFTConstructor的输出格式转换为ShareGPT格式

    EnvLIFT格式:
        [
            {
                "role": "system",
                "content": [{"text": "..."}]
            },
            {
                "role": "user",
                "content": [
                    {"text": "..."},
                    {"image": "path/to/image.png"}
                ]
            },
            ...
        ]

    ShareGPT格式:
        {
            "messages": [
                {"role": "system", "content": "..."},
                {"role": "user", "content": "<image>\n..."},
                ...
            ],
            "images": ["data/...", "data/...", ...]
        }

    Args:
        envlift_messages: EnvLIFTConstructor的输出消息列表
        output_dir: 输出目录，用于计算相对路径
        environment: 环境名称，用于构建正确的图片路径

    Returns:
        ShareGPT格式的字典
    """
    sharegpt_messages = []
    images = []

    for msg in envlift_messages:
        role = msg["role"]
        content_list = msg["content"]

        # 合并content并提取图像
        merged_content_parts = []

        for item in content_list:
            if "text" in item:
                merged_content_parts.append(item["text"])
            elif "image" in item:
                # 提取图像路径
                image_path = item["image"]

                # 转换为相对路径（相对于LLaMA-Factory工作目录）
                # image_path格式: output_dir/{environment}_trajectory_images/{task_id}/xxx.png
                # 需要转换为: data/{environment}_trajectory_images/{task_id}/xxx.png
                try:
                    # 获取相对于output_dir的路径
                    rel_path = os.path.relpath(image_path, output_dir)
                    # 添加data/前缀
                    image_relative = f"data/{rel_path}"
                    images.append(image_relative)
                except:
                    # 如果转换失败，直接使用原路径
                    images.append(image_path)

                # 添加<image>占位符
                merged_content_parts.append("<image>")

        # 合并为单个字符串
        merged_content = "\n".join(merged_content_parts).strip()

        sharegpt_messages.append({
            "role": role,
            "content": merged_content
        })

    return {
        "messages": sharegpt_messages,
        "images": images
    }


def trajectory_to_sharegpt(
    trajectory: List[Union[Dict, Any]],
    metadata: Dict[str, Any],
    environment: str,
    output_dir: str,
    include_system_message: bool = True,
    constructor: Any = None
) -> Dict[str, Any]:
    """
    将单个轨迹转换为ShareGPT格式

    Args:
        trajectory: 轨迹数据，格式为[StateInfo, Action, StateInfo, Action, ...]
        metadata: 元数据字典
        environment: 环境名称（如'classifieds'）
        output_dir: 输出目录
        include_system_message: 是否包含system消息
        constructor: EnvLIFTConstructor实例（如果为None，使用fallback方法）

    Returns:
        ShareGPT格式的字典
    """
    task_id = metadata['task_id']
    intent = metadata['intent']

    # 如果提供了constructor，使用EnvLIFTConstructor来构造消息
    if constructor is not None:
        try:
            adjusted_trajectory = list(trajectory)

            # 构造action_history
            action_history = ["None"]  # 初始状态的previous action是None

            for i in range(1, len(adjusted_trajectory)):
                item = adjusted_trajectory[i]
                if not isinstance(item, dict):  # 这是一个Action
                    # 使用format_previous_action来格式化
                    action_str = item.metadata['validation_info']['extracted_action']
                    action_history.append(action_str)

            # 提取intent图像（如果metadata中有的话）
            intent_images = metadata.get('intent_images', [])
            if isinstance(intent_images, list) and len(intent_images) > 0:
                # 转换为PIL Image对象
                intent_image_pil = [Image.fromarray(img.astype(np.uint8)) if isinstance(img, np.ndarray) else img
                                   for img in intent_images]
            else:
                intent_image_pil = []

            # 调用EnvLIFTConstructor的construct方法
            envlift_messages = constructor.construct(
                task_id=task_id,
                trajectory=adjusted_trajectory,
                intent=intent,
                page_screenshot_img=None,  # 未使用
                images=intent_image_pil,
                meta_data={"action_history": action_history},
            )

            # 转换为ShareGPT格式
            sharegpt_data = convert_envlift_to_sharegpt(envlift_messages, output_dir, environment)

            print(f"  使用EnvLIFTConstructor成功构造消息")
            return sharegpt_data

        except Exception as e:
            print(f"  警告: 使用EnvLIFTConstructor失败，使用fallback方法: {e}")
            import traceback
            traceback.print_exc()
            # 继续使用fallback方法

    # Fallback方法：EnvLIFTConstructor不可用时返回None
    print(f"  错误: EnvLIFTConstructor不可用，无法构造消息")
    return None


def convert_trajectories(
    data_dir: str,
    environments: List[str],
    output_dir: str,
    output_filename: str,
    include_system_message: bool = True,
    instruction_path: str = None,
    use_raw_prediction: bool = True
) -> None:
    """
    转换所有轨迹数据

    Args:
        data_dir: 轨迹数据根目录
        environments: 环境列表
        output_dir: 输出目录
        output_filename: 输出文件名
        include_system_message: 是否包含system消息
        instruction_path: 配置文件路径（相对于项目根目录或绝对路径）
        use_raw_prediction: 是否使用raw_prediction作为动作文本，默认为True
    """
    all_sharegpt_data = []
    total_trajectories = 0
    successful_conversions = 0
    total_images = 0

    for environment in environments:
        print(f"\n处理环境: {environment}")
        env_path = os.path.join(data_dir, environment)

        if not os.path.exists(env_path):
            print(f"警告: 环境目录不存在: {env_path}")
            continue

        # 为当前环境创建EnvLIFTConstructor实例
        print(f"为环境 {environment} 初始化EnvLIFTConstructor...")
        constructor = create_env_lift_constructor(output_dir, environment, instruction_path, use_raw_prediction)
        if constructor is None:
            print(f"错误: 无法为环境 {environment} 创建EnvLIFTConstructor，跳过该环境")
            continue

        # 读取轨迹文件
        print(f"读取轨迹文件...")
        trajectory_data = load_trajectory_files(env_path, task_ids=None)
        total_trajectories += len(trajectory_data)

        print(f"找到 {len(trajectory_data)} 个轨迹文件")

        # 转换每个轨迹
        for item in trajectory_data:
            task_id = item['task_id']
            trajectory = item['trajectory']
            file_path = item['file_path']

            # 读取对应的metadata
            metadata_path = file_path.replace('.pkl.xz', '_metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
            else:
                print(f"警告: 未找到metadata文件: {metadata_path}")
                metadata = {
                    'task_id': task_id,
                    'intent': 'unknown',
                    'environment': environment
                }


            # 转换为ShareGPT格式
            for index in range(2, len(trajectory)+2,2):
                try:
                    sharegpt_item = trajectory_to_sharegpt(
                        trajectory=trajectory[:index],
                        metadata=metadata,
                        environment=environment,
                        output_dir=output_dir,
                        include_system_message=include_system_message,
                        constructor=constructor
                    )

                    if sharegpt_item:
                        all_sharegpt_data.append(sharegpt_item)
                        successful_conversions += 1
                        total_images += len(sharegpt_item['images'])
                        print(f"✓ 成功转换 task_id={task_id}, 消息数={len(sharegpt_item['messages'])}, 图像数={len(sharegpt_item['images'])}")
                    else:
                        print(f"✗ 跳过 task_id={task_id}")

                except Exception as e:
                    print(f"✗ 错误: 处理 task_id={task_id} 时出错: {e}")
                    import traceback
                    traceback.print_exc()

    # 保存结果
    output_path = os.path.join(output_dir, output_filename)
    os.makedirs(output_dir, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_sharegpt_data, f, ensure_ascii=False, indent=2)

    # 打印统计信息
    print("\n" + "="*60)
    print("转换完成!")
    print("="*60)
    print(f"总轨迹数: {total_trajectories}")
    print(f"成功转换: {successful_conversions}")
    print(f"失败数量: {total_trajectories - successful_conversions}")
    print(f"生成对话数: {len(all_sharegpt_data)}")
    print(f"保存图像数: {total_images}")
    print(f"输出文件: {output_path}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="将轨迹数据转换为ShareGPT格式的微调数据"
    )

    parser.add_argument(
        '--data_dir',
        type=str,
        default='data/annotate_with_reasoning',
        help='轨迹数据根目录'
    )

    parser.add_argument(
        '--environments',
        type=str,
        nargs='+',
        default=['classifieds'],
        help='要处理的环境列表（可多选），如: classifieds reddit'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default='LLaMA-Factory/data',
        help='输出目录'
    )

    parser.add_argument(
        '--output_filename',
        type=str,
        default='trajectory_finetune_data_d.json',
        help='输出文件名'
    )

    parser.add_argument(
        '--include_system_message',
        action='store_true',
        help='是否包含system消息'
    )

    parser.add_argument(
        '--no_system_message',
        action='store_true',
        help='不包含system消息'
    )

    parser.add_argument(
        '--instruction_path',
        type=str,
        default='visualwebarena/src/prompts/vwa/jsons/lift_d.json',
        help='配置文件路径（相对于项目根目录或绝对路径），默认为visualwebarena/src/prompts/vwa/jsons/lift.json'
    )

    parser.add_argument(
        '--use_action_strs',
        default=True,
        # action='store_true',
        help='使用all_prev_action_strs作为动作文本（默认使用raw_prediction）'
    )

    args = parser.parse_args()

    # 处理system message flag
    include_system = args.include_system_message and not args.no_system_message
    # 处理use_raw_prediction flag（use_action_strs为True时，use_raw_prediction为False）
    use_raw_prediction = not args.use_action_strs

    print("="*60)
    print("轨迹数据转换为ShareGPT格式")
    print("="*60)
    print(f"数据目录: {args.data_dir}")
    print(f"环境列表: {args.environments}")
    print(f"输出目录: {args.output_dir}")
    print(f"输出文件: {args.output_filename}")
    print(f"包含system消息: {include_system}")
    print(f"配置文件: {args.instruction_path or '默认(lift.json)'}")
    print(f"动作文本来源: {'raw_prediction' if use_raw_prediction else 'action_strs'}")
    print("="*60)

    # 执行转换
    convert_trajectories(
        data_dir=args.data_dir,
        environments=args.environments,
        output_dir=args.output_dir,
        output_filename=args.output_filename,
        include_system_message=include_system,
        instruction_path=args.instruction_path,
        use_raw_prediction=use_raw_prediction
    )


if __name__ == '__main__':
    main()
