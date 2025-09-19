import torch
import numpy as np
import cv2
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt


def plot_1d_tensor(tensor):
    """
    可视化一个一维 tensor

    参数：
        tensor: torch.Tensor，一维张量
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("输入必须是一个 torch.Tensor")
    if tensor.ndim != 1:
        raise ValueError("输入必须是一维张量")

    # 转换为 numpy 方便绘图
    values = tensor.cpu().numpy()
    indices = range(len(values))

    plt.figure(figsize=(8, 4))
    plt.plot(indices, values, marker='o')
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.title("1D Tensor Visualization")
    plt.grid(True)
    plt.show()


# 示例
t = torch.tensor([1, 3, 2, 4, 6, 5], dtype=torch.float32)
plot_1d_tensor(t)
def visualize_tensor_distribution(tensor, save_path=None):
    """
    输入：
        tensor: 一维 PyTorch tensor
        save_path: (可选) 保存图片的路径，若为 None 则直接显示图形
    功能：
        将 tensor 从小到大排序，按值域等分成 100 段，统计每段数量并可视化。
        可选择保存图片。
    """
    if tensor.ndim != 1:
        raise ValueError("输入必须为一维 tensor")

    # 转为 CPU float 类型，方便处理
    data = tensor.detach().cpu().float()

    # 从小到大排序
    sorted_data, _ = torch.sort(data)

    # 得到最小值和最大值
    min_val = sorted_data[0].item()
    max_val = sorted_data[-1].item()

    # 统计每个区间的数量（等值域切分）
    counts = torch.histc(data, bins=100, min=min_val, max=max_val)

    # 绘制柱状图
    plt.figure(figsize=(12, 6))
    plt.bar(range(100), counts.numpy(), width=1.0, edgecolor='black')
    plt.xlabel("Bins")
    plt.ylabel("Count")
    plt.title("Tensor Value Distribution (100 Segments)")

    # 保存或显示
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"图片已保存到: {save_path}")
    else:
        plt.show()

    return counts
def show_mask_on_image(img, mask):
    """
    参数:
      img: (h, w, 3)的ndarray图片，数值范围假定为0-255
      mask: (h, w)的ndarray，注意力图（建议值在0到1之间，如果不是则归一化）

    功能:
      - 对mask归一化，使其值在[0, 1]之间
      - 在原图上叠加深蓝色阴影：mask值为0处不影响原图，mask值较大的地方叠加较深的蓝色阴影
      - 同时生成mask对应的热力图（利用matplotlib的jet colormap）

    返回:
      overlay_img: 原图与阴影叠加后的图片
      heatmap: 基于mask生成的热力图（RGB格式）
    """
    # 将输入图像转换为float32方便运算
    img_float = img.astype(np.float32)

    # 归一化mask到[0,1]
    mask_min, mask_max = mask.min(), mask.max()
    if mask_max - mask_min > 1e-8:
        mask_norm = (mask - mask_min) / (mask_max - mask_min)
    else:
        mask_norm = mask.copy().astype(np.float32)

    # 生成深蓝色阴影（这里选用RGB=(0, 0, 139)）
    blue_shadow = np.array([0, 0, 139], dtype=np.float32)

    # 调整叠加程度因子, 例如最大叠加比例为0.6
    max_alpha = 0.6
    # 将mask_norm乘上最大叠加比例，作为每个像素的透明度
    alpha_mask = mask_norm[..., np.newaxis] * max_alpha

    # 叠加深蓝色：采用线性混合
    overlay_img = (1 - alpha_mask) * img_float + alpha_mask * blue_shadow
    # 保证结果在0-255范围，并转换回uint8
    overlay_img = np.clip(overlay_img, 0, 255).astype(np.uint8)

    # 生成热力图：使用matplotlib的jet colormap
    cmap = plt.get_cmap('jet')
    # cmap接受的输入要求在0-1之间，所以这里用mask_norm
    heatmap = cmap(mask_norm)[:, :, :3]  # 取RGB部分，忽略alpha通道
    # 若需要将热力图转换为0-255的uint8图像，可以乘以255
    heatmap = (heatmap * 255).astype(np.uint8)

    return overlay_img, heatmap