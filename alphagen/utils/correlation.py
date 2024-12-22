import torch
from torch import Tensor

from alphagen.utils.pytorch_utils import masked_mean_std


def _mask_either_nan(x: Tensor, y: Tensor, fill_with: float = torch.nan):
    x = x.clone()                       # [days, stocks]
    y = y.clone()                       # [days, stocks]
    nan_mask = x.isnan() | y.isnan()
    x[nan_mask] = fill_with
    y[nan_mask] = fill_with
    n = (~nan_mask).sum(dim=1)
    return x, y, n, nan_mask


# def _rank_data(x: Tensor, nan_mask: Tensor) -> Tensor:
#     rank = x.argsort().argsort().float()            # [d, s]
#     eq = x[:, None] == x[:, :, None]                # [d, s, s]
#     eq = eq / eq.sum(dim=2, keepdim=True)           # [d, s, s]
#     rank = (eq @ rank[:, :, None]).squeeze(dim=2)
#     rank[nan_mask] = 0
#     return rank                                     # [d, s]


def _rank_data(x: torch.Tensor, nan_mask: torch.Tensor, batch_size: int = 32) -> torch.Tensor:
    """
    计算每个元素的排名，并处理 NaN 值。

    参数：
    - x: 输入张量，形状为 [d, s]，其中 d 为样本数，s 为每个样本的特征维度。
    - nan_mask: 一个布尔张量，形状为 [d, s]，标记哪些元素是 NaN。
    - batch_size: 处理时的批次大小，默认值为 32。

    返回：
    - 返回一个形状为 [d, s] 的张量，表示每个元素的排名。
    """
    # 获取张量的维度
    d, s = x.shape

    # 计算排序的排名
    rank = x.argsort().argsort().float()  # [d, s]

    # 初始化结果张量
    result = torch.zeros_like(rank)  # [d, s]

    # 按批次处理数据
    for batch_start in range(0, s, batch_size):
        batch_end = min(batch_start + batch_size, s)

        # 选择当前批次的数据
        x_batch = x[:, batch_start:batch_end]  # 形状 [d, batch_size]
        rank_batch = rank[:, batch_start:batch_end]  # 形状 [d, batch_size]

        # 逐列计算相等关系并累加排名
        for i in range(batch_end - batch_start):
            # 计算当前批次列与其他列的相等关系
            eq = (x_batch == x_batch[:, i][:, None]).float()  # [d, batch_size]

            # 对相等关系进行归一化
            eq /= eq.sum(dim=1, keepdim=True)  # [d, batch_size]，每列归一化

            # 计算加权排名：eq 与 rank 相乘
            result[:, batch_start + i] = torch.sum(eq * rank_batch[:, i][:, None], dim=1)  # 逐列加权

    # 使用 nan_mask 将无效值设为 0
    result[nan_mask] = 0

    return result  # [d, s]


def _batch_pearsonr_given_mask(
    x: Tensor, y: Tensor,
    n: Tensor, mask: Tensor
) -> Tensor:
    x_mean, x_std = masked_mean_std(x, n, mask)
    y_mean, y_std = masked_mean_std(y, n, mask)
    cov = (x * y).sum(dim=1) / n - x_mean * y_mean
    stdmul = x_std * y_std
    stdmul[(x_std < 1e-3) | (y_std < 1e-3)] = 1
    corrs = cov / stdmul
    return corrs


def batch_spearmanr(x: Tensor, y: Tensor) -> Tensor:
    x, y, n, nan_mask = _mask_either_nan(x, y)
    rx = _rank_data(x, nan_mask)
    ry = _rank_data(y, nan_mask)
    return _batch_pearsonr_given_mask(rx, ry, n, nan_mask)


def batch_pearsonr(x: Tensor, y: Tensor) -> Tensor:
    return _batch_pearsonr_given_mask(*_mask_either_nan(x, y, fill_with=0.))
