import torch
def min_max_normalize(x):
    x_min, x_max = x.min(), x.max()
    return (x - x_min) / (x_max - x_min)


def stretch_middle(x, gamma=2.0):
    x_min, x_max = x.min(), x.max()
    x_norm = (x - x_min) / (x_max - x_min)
    y = x_norm ** gamma
    return y * (x_max - x_min) + x_min

def z_score(x):
    x_mean = x.mean()
    x_std = x.std()
    return (x - x_mean) / x_std

def softmax_tensor(x):
    # 减去最大值防止数值溢出
    x_exp = torch.exp(x - torch.max(x))
    return x_exp / torch.sum(x_exp)

def sigmmoid(x):
    return 1 / (1 + torch.exp(-x))


def median_contrast_transform(x, scale_below, scale_above):
    """
    对张量进行中位数对比度变换。
    - 中位数以下的值变得更小。
    - 中位数以上的值变得更大。

    参数:
    x (torch.Tensor): 输入张量。
    scale_below (float): 应用于中位数以下值的缩放因子 (必须 > 1)。
    scale_above (float): 应用于中位数以上值的缩放因子 (必须 > 1)。
    """
    if not (scale_below > 1 and scale_above > 1):
        raise ValueError("缩放因子 scale_below 和 scale_above 都必须大于 1")

    # 1. 计算中位数
    median = torch.median(x)

    # 2. 创建一个与x相同形状的输出张量
    y = x.clone()

    # 3. 识别中位数以上和以下的元素
    mask_below = x < median
    mask_above = x > median

    # 4. 应用变换
    # 对于低于中位数的值：使其与中位数的距离拉大，从而变得更小
    y[mask_below] = median + scale_below * (x[mask_below] - median)

    # 对于高于中位数的值：使其与中位数的距离拉大，从而变得更大
    y[mask_above] = median + scale_above * (x[mask_above] - median)

    return y
def quantile_invert(t: torch.Tensor) -> torch.Tensor:
    """
    基于分布分位数反转一个 PyTorch tensor
    大值变成小的百分位，小值变成大的百分位
    """
    device = t.device
    shape = t.shape
    flat = t.flatten()

    # 排序并计算分位数 (0~1)
    ranks = torch.argsort(torch.argsort(flat))  # 排名，从0到N-1
    quantiles = ranks.float() / (len(flat) - 1)  # 转成分位数

    # 分位数反转
    inv_quantiles = 1 - quantiles

    # reshape 回原来形状
    return inv_quantiles.view(shape).to(device)

if __name__ == '__main__':
    x = torch.tensor([0.0004, 0.0005, 0.0005, 0.0005, 0.0005,
                      0.0005, 0.0005, 0.0005, 0.0005, 0.0005,
                      0.001, 0.001, 0.001, 0.001, 0.001,
                      0.002,0.004])

    # 先对x进行min-max归一化
    x = min_max_normalize(x)
    print(x)
    y = quantile_invert(x)
    print(y)
    # print(median_contrast_transform(x, scale_below=2.0, scale_above=2.0))
