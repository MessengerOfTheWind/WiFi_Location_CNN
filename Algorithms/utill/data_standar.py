import numpy as np

def min_max_scale_np(array, axis=None, eps=1e-12):
    """
    NumPy 实现，可指定 axis：
    axis=None → 整体归一化
    axis=0    → 按列归一化
    axis=1    → 按行归一化
    """
    array = np.asarray(array, dtype=float)
    vmin = array.min(axis=axis, keepdims=True)
    vmax = array.max(axis=axis, keepdims=True)
    scaled = (array - vmin) / (vmax - vmin + eps)  # eps 防止除 0
    return scaled


def normalize_rssi(data):
    # data: shape (N, 23, 23)
    min_val = data.min()
    max_val = data.max()
    data_norm = (data - min_val) / (max_val - min_val + 1e-12)
    return data_norm, min_val, max_val


def normalize_test_or_valid_data(min_val, max_val, valid_data):
    data_norm = (valid_data - min_val) / (max_val - min_val + 1e-12)
    return data_norm


def normalize_coords(total_coords, coords):
    # coords: shape (N, 2)
    min_vals = total_coords.min(axis=0)
    max_vals = total_coords.max(axis=0)
    coords_norm = (coords - min_vals) / (max_vals - min_vals + 1e-12)
    return coords_norm, min_vals, max_vals


def denormalize_rssi(normalized_data, min_val, max_val):
    """
    反归一化 RSSI 数据
    参数：
        normalized_data: 归一化后的 RSSI 数据 (N, 23, 23)
        min_val: 训练时记录的最小值
        max_val: 训练时记录的最大值
    返回：
        原始尺度的 RSSI 数据
    """
    return normalized_data * (max_val - min_val) + min_val


def denormalize_coords(normalized_coords, min_vals, max_vals):
    """
    反归一化坐标数据（每列分别恢复）
    参数：
        normalized_coords: 归一化后的坐标数组 (N, 2)
        min_vals: 每列的最小值 (2,)
        max_vals: 每列的最大值 (2,)
    返回：
        恢复后的真实坐标 (N, 2)
    """
    return normalized_coords * (max_vals - min_vals) + min_vals
# def denormalize_coords(normalized_coords, min_vals, max_vals):
#     if normalized_coords.ndim == 1:
#         # 说明只预测了一个值，比如经度或纬度
#         normalized_coords = normalized_coords.reshape(-1, 1)
#     if min_vals.ndim == 1:
#         min_vals = min_vals.reshape(1, -1)
#     if max_vals.ndim == 1:
#         max_vals = max_vals.reshape(1, -1)
#     return normalized_coords * (max_vals - min_vals) + min_vals



def mean_euclidean_distance(y_pred, y_true):
    """
    计算预测坐标与真实坐标之间的平均欧拉距离
    参数：
        y_pred: 预测坐标，形状为 (N, 2)
        y_true: 真实坐标，形状为 (N, 2)
    返回：
        平均欧拉距离（float）
    """
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    distances = np.linalg.norm(y_pred - y_true, axis=1)  # 每行一个点
    return np.mean(distances)

