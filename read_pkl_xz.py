import pickle
import lzma
import os


def read_pkl_xz(file_path):
    """
    读取 .pkl.xz 文件的函数

    参数:
        file_path (str): .pkl.xz 文件的路径

    返回:
        解压缩和反序列化后的Python对象
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")

    if not file_path.endswith('.pkl.xz'):
        raise ValueError("文件必须是 .pkl.xz 格式")

    try:
        # 使用 lzma 打开压缩文件，然后用 pickle 加载
        with lzma.open(file_path, 'rb') as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        raise Exception(f"读取文件时出错: {e}")


def read_target_file():
    """
    读取指定的文件
    """
    file_path = r"D:\localrepository\LIFT\data\annotate\trajectories\classifieds\classifieds_4_20250926_163653.pkl.xz"
    return read_pkl_xz(file_path)


if __name__ == "__main__":
    # 示例用法
    try:
        data = read_target_file()
        print(f"成功读取文件！数据类型: {type(data)}")

        # 如果是字典，显示键
        if isinstance(data, dict):
            print(f"字典包含 {len(data)} 个键:")
            for key in list(data.keys())[:10]:  # 只显示前10个键
                print(f"  - {key}")
            if len(data) > 10:
                print(f"  ... 还有 {len(data) - 10} 个键")

        # 如果是列表，显示长度和前几个元素的类型
        elif isinstance(data, list):
            print(f"列表包含 {len(data)} 个元素")
            if len(data) > 0:
                print(f"第一个元素类型: {type(data[0])}")
                if len(data) > 1:
                    print(f"第二个元素类型: {type(data[1])}")

        # 其他类型
        else:
            print(f"数据内容: {str(data)[:200]}...")  # 只显示前200个字符

    except Exception as e:
        print(f"错误: {e}")