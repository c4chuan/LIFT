import os
import json
import re
import pickle
import lzma
from datasets import Dataset

def supervise_dataset_construct(initial_configs,config_path = "/data/wangzhenchuan/Projects/LIFT/vwa/configs/visualwebarena/test_classifieds_v2", data_path = None):
    """
    读取 config_path 目录下所有 .json 文件，
    并通过 load_dataset 直接构造一个 Hugging Face Dataset.
    Args:
        config_path (str): 存放 0.json,1.json,… 的文件夹路径
        data_path (str, optional): 如果你的 JSON 里有跟 data_path 组合使用的字段，可以在这里传入（本例不使用）
    Returns:
        datasets.Dataset: 合并后的数据集（train split）
    """
    files = sorted(
        [f for f in os.listdir(config_path) if f.endswith(".json")],
        key=lambda x: int(os.path.splitext(x)[0])
    )
    trajectory_files = None
    if 'annotate_path' in initial_configs:
        annotation_path = initial_configs.annotate_path
        classifieds_path = annotation_path + '/' + 'classifieds'

        trajectory_files = load_trajectory_files(classifieds_path)

    records = []
    # 是否有标注文件存在
    if trajectory_files is not None:
        for trajectory_file in trajectory_files:
            fname = str(trajectory_file['task_id'])+ '.json'
            full = os.path.join(config_path, fname)
            with open(full, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            cfg['ref_trajectory'] = trajectory_file['trajectory']
            records.append(cfg)
    else:
        for fname in files:
            full = os.path.join(config_path, fname)
            with open(full, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            records.append(cfg)

    # 直接从 list[dict] 构造，无视不同文件间的 schema 差异
    # mydataset = Dataset.from_list(records)

    return records
def dataset_construct(config_path = "/data/wangzhenchuan/Projects/LIFT/vwa/configs/visualwebarena/test_classifieds_v2", data_path = None):
    """
    读取 config_path 目录下所有 .json 文件，
    并通过 load_dataset 直接构造一个 Hugging Face Dataset.
    Args:
        config_path (str): 存放 0.json,1.json,… 的文件夹路径
        data_path (str, optional): 如果你的 JSON 里有跟 data_path 组合使用的字段，可以在这里传入（本例不使用）
    Returns:
        datasets.Dataset: 合并后的数据集（train split）
    """
    files = sorted(
        [f for f in os.listdir(config_path) if f.endswith(".json")],
        key=lambda x: int(os.path.splitext(x)[0])
    )

    records = []
    for fname in files:
        full = os.path.join(config_path, fname)
        with open(full, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        records.append(cfg)

    # 直接从 list[dict] 构造，无视不同文件间的 schema 差异
    mydataset = Dataset.from_list(records)

    # 保存到data_path
    if not data_path == None:
        mydataset.save_to_disk(data_path)
    return mydataset

def load_trajectory_files(classifieds_path, task_ids=None):
    """
    读取classifieds目录下的pkl.xz文件

    Args:
        classifieds_path (str): 存放轨迹文件的目录路径，如 'data/annotate/trajectories/classifieds'
        task_ids (list, optional): 指定要读取的任务编号列表。如果为None，则读取所有任务

    Returns:
        list: 包含轨迹数据的字典列表，每个元素格式为:
              {'task_id': int, 'trajectory': object, 'file_path': str}
    """
    if not os.path.exists(classifieds_path):
        print(f"警告: 目录 {classifieds_path} 不存在")
        return []

    # 如果没有指定task_ids，则提取所有任务编号
    if task_ids is None:
        task_ids_set = set()
        for filename in os.listdir(classifieds_path):
            match = re.match(r'classifieds_(\d+)_.*\.pkl\.xz$', filename)
            if match:
                task_ids_set.add(int(match.group(1)))
        task_ids = sorted(list(task_ids_set))

    # 为每个task_id找到对应的pkl.xz文件（取最新的）
    trajectory_data = []
    for task_id in task_ids:
        # 查找该task_id的所有pkl.xz文件
        pattern = re.compile(rf'classifieds_{task_id}_(\d+)_(\d+)\.pkl\.xz$')
        matching_files = []

        for filename in os.listdir(classifieds_path):
            if pattern.match(filename):
                matching_files.append(filename)

        if not matching_files:
            print(f"警告: 未找到task_id={task_id}的pkl.xz文件")
            continue

        # 取最新的文件（按文件名排序，最后一个是最新的）
        latest_file = sorted(matching_files)[-1]
        file_path = os.path.join(classifieds_path, latest_file)

        # 读取pkl.xz文件
        try:
            with lzma.open(file_path, 'rb') as f:
                trajectory = pickle.load(f)

            trajectory_data.append({
                'task_id': task_id,
                'trajectory': trajectory,
                'file_path': file_path
            })
            print(f"成功读取: task_id={task_id}, 文件={latest_file}")

        except Exception as e:
            print(f"错误: 无法读取task_id={task_id}的文件 {latest_file}: {str(e)}")
            continue

    return trajectory_data