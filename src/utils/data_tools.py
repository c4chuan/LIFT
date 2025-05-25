import os
import json
from datasets import Dataset

def dataset_construct(config_path = "../vwa/configs/visualwebarena/test_classifieds_v2", data_path = None):
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