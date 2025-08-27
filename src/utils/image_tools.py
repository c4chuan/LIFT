from PIL import Image
import numpy as np
import cv2

def pil_to_cv2_and_save(pil_image: Image.Image, save_path: str) -> None:
    """
    将 PIL Image 转换为 NumPy 数组，并使用 OpenCV 保存到本地文件。

    参数:
    pil_image (Image.Image): 待转换的 PIL Image 对象。
    save_path (str): 保存文件的路径，例如 'output.jpg' 或 'output.png'。
    """
    # 1. 将 PIL Image 转换为 NumPy 数组
    # PIL 图像的颜色通道顺序是 RGB，而 OpenCV 是 BGR。
    # 这里需要进行通道转换，pil_image.mode 获取图像模式，例如 'RGB'。
    # 如果是 RGBA 图像，转换时要先去除 alpha 通道，因为 cv2.imwrite 不支持 alpha 通道。
    if pil_image.mode == 'RGB':
        # 将 PIL.Image 转换为 NumPy 数组，通道顺序为 RGB
        # np.array(pil_image) 的默认通道顺序就是 RGB
        img_np = np.array(pil_image)
        # 将 RGB 转换为 BGR，因为 cv2.imwrite 默认保存为 BGR 格式
        img_cv2 = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    elif pil_image.mode == 'RGBA':
        # PIL.Image 转 NumPy 数组，通道顺序为 RGBA
        img_np = np.array(pil_image)
        # 提取 RGB 三个通道，并转换为 BGR
        img_rgb = img_np[:, :, :3]
        img_cv2 = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    else:
        # 如果是灰度图或其他模式，直接转换，cv2.imwrite 支持灰度图
        img_cv2 = np.array(pil_image)

    # 2. 使用 cv2.imwrite 保存图像
    try:
        cv2.imwrite(save_path, img_cv2)
        print(f"图像已成功保存到: {save_path}")
    except Exception as e:
        print(f"保存图像时出错: {e}")

def cv2_save(image_array: np.ndarray, save_path: str) -> None:
    """
    将 NumPy 数组（ndarray）直接保存为图像文件。

    这个函数假定输入的 NumPy 数组是 OpenCV 兼容的格式，
    即通道顺序为 BGR（对于彩色图像）。

    参数:
    image_array (np.ndarray): 待保存的 NumPy 数组。
    save_path (str): 保存文件的路径，例如 'output.jpg' 或 'output.png'。
    """
    try:
        cv2.imwrite(save_path, image_array)
        print(f"NumPy 数组已成功保存到: {save_path}")
    except Exception as e:
        print(f"保存图像时出错: {e}")

if __name__ == '__main__':
    # 示例用法
    # 1. 创建一个示例 PIL Image 对象
    # 创建一个 100x100 的白色 RGB 图像
    pil_img = Image.new('RGB', (100, 100), color='white')

    # 2. 调用函数并保存
    pil_to_cv2_and_save(pil_img, 'saved_image.png')

    # 3. 验证是否成功保存
    # 尝试加载刚刚保存的图像
    try:
        loaded_img = cv2.imread('saved_image.png')
        if loaded_img is not None:
            print("已成功验证保存的图像。")
            print(f"加载的图像尺寸: {loaded_img.shape}")
        else:
            print("无法加载保存的图像，可能保存失败。")
    except Exception as e:
        print(f"加载图像时出错: {e}")