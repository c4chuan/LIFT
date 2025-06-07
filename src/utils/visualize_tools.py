import torch
import numpy as np
import cv2
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

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