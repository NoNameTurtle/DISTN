import torch
import torch.nn as nn
from sklearn.metrics import r2_score
import numpy as np


def BCE_with_logit_torch(pred, true, mask_value=None, mask=None, weights=None, pos_weight=None):
    if mask_value is not None:
        mask = torch.gt(true, mask_value)
    if mask is not None:  # mask_value (priority) or mask is specified
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
        weights = torch.masked_select(weights, mask) if weights is not None else None
    bce_loss = nn.BCEWithLogitsLoss(weight=weights, pos_weight=pos_weight)
    return bce_loss(pred, true)


def MAE_torch(pred, true, mask_value=None, mask=None, weights=None):
    if weights is None:
        weights = torch.ones_like(true)

    if mask_value is not None:
        mask = torch.gt(true, mask_value)
    if mask is not None:  # mask_value (priority) or mask is specified
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
        weights = torch.masked_select(weights, mask)
    return torch.mean(torch.abs(true - pred) * weights)


def MAPE_torch(pred, true, mask_value=None, mask=None, weights=None):
    if weights is None:
        weights = torch.ones_like(true)

    if mask_value is not None:
        mask = torch.gt(true, mask_value) if mask is None else torch.gt(true, mask_value) & mask
    if mask is not None:  # mask_value (priority) or mask is specified
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
        weights = torch.masked_select(weights, mask)
    return torch.mean(torch.abs(torch.div((true - pred), true)) * weights)


def SMAPE_torch(pred, true, mask_value=None, mask=None, weights=None):
    if weights is None:
        weights = torch.ones_like(true)

    if mask_value is not None:
        mask = torch.gt(true, mask_value)
    if mask is not None:  # mask_value (priority) or mask is specified
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
        weights = torch.masked_select(weights, mask)
    return torch.mean(torch.div((true - pred).abs(), (true.abs() + pred.abs()) / 2) * weights)


def MSE_torch(pred, true, mask_value=None, mask=None, weights=None):
    if weights is None:
        weights = torch.ones_like(true)

    if mask_value is not None:
        mask = torch.gt(true, mask_value)
    if mask is not None:  # mask_value (priority) or mask is specified
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
        weights = torch.masked_select(weights, mask)

    return torch.mean(((pred - true) ** 2) * weights)


def RMSE_torch(pred, true, mask_value=None, mask=None, weights=None):
    if weights is None:
        weights = torch.ones_like(true)

    if mask_value is not None:
        mask = torch.gt(true, mask_value)
    if mask is not None:  # mask_value (priority) or mask is specified
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
        weights = torch.masked_select(weights, mask)

    return torch.sqrt(torch.mean(((pred - true) ** 2) * weights))


def Gumbel_GEVL_torch(pred, true, gamma=1., mask_value=None, mask=None, weights=None):
    if weights is None:
        weights = torch.ones_like(true)

    if mask_value is not None:
        mask = torch.gt(true, mask_value)
    if mask is not None:  # mask_value (priority) or mask is specified
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
        weights = torch.masked_select(weights, mask)

    return torch.mean(neg_log_gumbel_kernel(pred, true, gamma) * weights)


def Frechet_GEVL_torch(pred, true, alpha=13., s=1.7, mask_value=None, mask=None, weights=None):
    if weights is None:
        weights = torch.ones_like(true)

    if mask_value is not None:
        mask = torch.gt(true, mask_value)
    if mask is not None:  # mask_value (priority) or mask is specified
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
        weights = torch.masked_select(weights, mask)

    return torch.mean(neg_log_frechet_kernel(pred, true, s, alpha) * weights)


def r2_score_torch(pred, true, mask_value=None, mask=None):
    if mask_value is not None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    elif mask is not None:
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    if true.numel() == 0:
        return np.nan
    else:
        return r2_score(true.cpu().numpy(), pred.cpu().numpy())


def neg_log_gumbel_kernel(pred, true, gamma=1.):
    square = (pred - true) ** 2
    return ((1 - torch.exp(-square)) ** gamma) * square


def neg_log_frechet_kernel(pred, true, s=1.7, alpha=13.):
    delta = torch.abs(pred - true)
    coef = s * (alpha / (1 + alpha)) ** (1 / alpha)
    first = (-1 - alpha) * (-(delta + coef) / s) ** (-alpha)
    second = torch.log((delta + coef) / s)
    return first + second


def absolution_error_ratio_score_torch(true, pred, grand_mean, mask_value=None, mask=None):
    """
    Absolution Error Ratio Score (AERS) = 1 - MAE / grand_mean，关于该指标的说明：
    - grand mean是KPI在所有小区所有scenario下的均值，作为一般情况下KPI水平的基准
    - 用grand mean主要是考虑到告警情况下存在大量实际值为0的情况，无法使用MAPE等比例值进行模型评价
    - 用1去减是为了让分数越高代表模型越好
    - R2是从方差的角度衡量模型的拟合程度，而AERS是从绝对误差的角度衡量模型的预测精度，更符合对于一般来说对于误差的认知
    :param true: KPI实际值
    :param pred: KPI预测值
    :param grand_mean: 作为ratio分母的KPI总体均值
    :param mask_value:
    :param mask:
    :return:
    """
    if mask_value is not None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    elif mask is not None:
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return 1 - (true - pred).abs().mean() / grand_mean


def metrics(pred, true):
    mae = MAE_torch(pred, true).item()
    mse = MSE_torch(pred, true).item()
    rmse = RMSE_torch(pred, true).item()
    smape = SMAPE_torch(pred, true).item()
    r2 = r2_score_torch(pred.reshape(-1), true.reshape(-1))
    return mae, mse, rmse, smape, r2
