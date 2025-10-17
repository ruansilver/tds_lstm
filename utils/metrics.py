"""
评估指标计算
"""

import numpy as np
from typing import Dict, List
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging

logger = logging.getLogger(__name__)

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    计算回归任务的评估指标
    
    Args:
        y_true: 真实值，形状为 (n_samples, seq_len, n_features) 或 (n_samples, n_features)
        y_pred: 预测值，形状与y_true相同
        
    Returns:
        包含各种指标的字典
    """
    # 展平为二维数组进行计算
    if y_true.ndim > 2:
        y_true_flat = y_true.reshape(-1, y_true.shape[-1])
        y_pred_flat = y_pred.reshape(-1, y_pred.shape[-1])
    else:
        y_true_flat = y_true
        y_pred_flat = y_pred
    
    # 基本回归指标
    mse = mean_squared_error(y_true_flat, y_pred_flat)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true_flat, y_pred_flat)
    
    # R²分数
    r2 = r2_score(y_true_flat, y_pred_flat)
    
    # 平均绝对百分比误差 (MAPE)
    mape = calculate_mape(y_true_flat, y_pred_flat)
    
    # 归一化均方根误差 (NRMSE)
    nrmse = rmse / (np.max(y_true_flat) - np.min(y_true_flat)) if np.max(y_true_flat) != np.min(y_true_flat) else 0
    
    # 相关系数
    correlation = calculate_correlation(y_true_flat, y_pred_flat)
    
    # 最大绝对误差
    max_error = np.max(np.abs(y_true_flat - y_pred_flat))
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mape,
        'nrmse': nrmse,
        'correlation': correlation,
        'max_error': max_error
    }

def calculate_per_joint_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, List[float]]:
    """
    计算每个关节的评估指标
    
    Args:
        y_true: 真实值，形状为 (n_samples, seq_len, n_joints)
        y_pred: 预测值，形状与y_true相同
        
    Returns:
        包含每个关节指标的字典
    """
    if y_true.ndim > 2:
        y_true_flat = y_true.reshape(-1, y_true.shape[-1])
        y_pred_flat = y_pred.reshape(-1, y_pred.shape[-1])
    else:
        y_true_flat = y_true
        y_pred_flat = y_pred
    
    n_joints = y_true_flat.shape[1]
    
    per_joint_metrics = {
        'mse': [],
        'rmse': [],
        'mae': [],
        'r2': [],
        'mape': [],
        'correlation': []
    }
    
    for joint_idx in range(n_joints):
        y_true_joint = y_true_flat[:, joint_idx]
        y_pred_joint = y_pred_flat[:, joint_idx]
        
        # 计算该关节的指标
        mse = mean_squared_error(y_true_joint, y_pred_joint)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true_joint, y_pred_joint)
        r2 = r2_score(y_true_joint, y_pred_joint)
        mape = calculate_mape(y_true_joint.reshape(-1, 1), y_pred_joint.reshape(-1, 1))
        correlation = np.corrcoef(y_true_joint, y_pred_joint)[0, 1] if len(np.unique(y_true_joint)) > 1 else 0
        
        per_joint_metrics['mse'].append(mse)
        per_joint_metrics['rmse'].append(rmse)
        per_joint_metrics['mae'].append(mae)
        per_joint_metrics['r2'].append(r2)
        per_joint_metrics['mape'].append(mape)
        per_joint_metrics['correlation'].append(correlation)
    
    return per_joint_metrics

def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8) -> float:
    """
    计算平均绝对百分比误差 (MAPE)
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        epsilon: 避免除零的小数
        
    Returns:
        MAPE值
    """
    # 避免除零
    y_true_safe = np.where(np.abs(y_true) < epsilon, epsilon, y_true)
    mape = np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100
    return mape

def calculate_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    计算预测值与真实值的相关系数
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        
    Returns:
        相关系数
    """
    # 展平数组
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    # 计算相关系数
    if len(np.unique(y_true_flat)) > 1:
        correlation = np.corrcoef(y_true_flat, y_pred_flat)[0, 1]
        # 处理NaN值
        if np.isnan(correlation):
            correlation = 0.0
    else:
        correlation = 0.0
    
    return correlation

def calculate_sequence_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, np.ndarray]:
    """
    计算序列级别的指标（每个时间步的指标）
    
    Args:
        y_true: 真实值，形状为 (n_samples, seq_len, n_features)
        y_pred: 预测值，形状与y_true相同
        
    Returns:
        包含每个时间步指标的字典
    """
    seq_len = y_true.shape[1]
    
    sequence_metrics = {
        'mse_per_timestep': [],
        'mae_per_timestep': [],
        'correlation_per_timestep': []
    }
    
    for t in range(seq_len):
        # 提取该时间步的数据
        y_true_t = y_true[:, t, :]
        y_pred_t = y_pred[:, t, :]
        
        # 计算指标
        mse_t = mean_squared_error(y_true_t, y_pred_t)
        mae_t = mean_absolute_error(y_true_t, y_pred_t)
        corr_t = calculate_correlation(y_true_t, y_pred_t)
        
        sequence_metrics['mse_per_timestep'].append(mse_t)
        sequence_metrics['mae_per_timestep'].append(mae_t)
        sequence_metrics['correlation_per_timestep'].append(corr_t)
    
    # 转换为numpy数组
    for key in sequence_metrics:
        sequence_metrics[key] = np.array(sequence_metrics[key])
    
    return sequence_metrics

def calculate_prediction_confidence(
    predictions: np.ndarray, 
    monte_carlo_samples: List[np.ndarray] = None
) -> Dict[str, np.ndarray]:
    """
    计算预测的置信度（如果有蒙特卡洛样本）
    
    Args:
        predictions: 主要预测结果
        monte_carlo_samples: 蒙特卡洛采样结果列表
        
    Returns:
        置信度指标
    """
    if monte_carlo_samples is None:
        # 如果没有蒙特卡洛样本，返回零方差
        return {
            'prediction_std': np.zeros_like(predictions),
            'prediction_var': np.zeros_like(predictions),
            'confidence_intervals': np.zeros((*predictions.shape, 2))
        }
    
    # 将蒙特卡洛样本堆叠
    mc_array = np.stack(monte_carlo_samples, axis=0)  # (n_samples, batch, seq, features)
    
    # 计算统计量
    prediction_std = np.std(mc_array, axis=0)
    prediction_var = np.var(mc_array, axis=0)
    
    # 计算95%置信区间
    confidence_intervals = np.percentile(mc_array, [2.5, 97.5], axis=0)
    confidence_intervals = np.transpose(confidence_intervals, (1, 2, 3, 0))  # (batch, seq, features, 2)
    
    return {
        'prediction_std': prediction_std,
        'prediction_var': prediction_var,
        'confidence_intervals': confidence_intervals
    }

def print_metrics_summary(metrics: Dict[str, float], title: str = "评估指标"):
    """
    打印指标摘要
    
    Args:
        metrics: 指标字典
        title: 标题
    """
    logger.info(f"\n{title}")
    logger.info("-" * 50)
    for metric_name, metric_value in metrics.items():
        if isinstance(metric_value, (int, float)):
            logger.info(f"{metric_name:15s}: {metric_value:.6f}")
        else:
            logger.info(f"{metric_name:15s}: {metric_value}")
    logger.info("-" * 50)


# ========== 对齐metrics - 角度导数相关 ==========

def calculate_angular_derivatives(
    predictions: np.ndarray,
    sample_rate: float = 2000.0
) -> Dict[str, float]:
    """
    计算角度的导数指标：角速度、角加速度、角加速度变化率（jerk）
    
    Args:
        predictions: 预测的关节角度 (batch, time, joints) 或 (time, joints)
        sample_rate: 采样率（Hz）
        
    Returns:
        包含导数指标的字典
    """
    if predictions.ndim == 2:
        # (time, joints) -> (1, time, joints)
        predictions = predictions[np.newaxis, :]
    
    # 计算导数（使用diff）
    vel = np.diff(predictions, axis=1)  # 角速度
    acc = np.diff(vel, axis=1)  # 角加速度  
    jerk = np.diff(acc, axis=1)  # 角加速度变化率
    
    # 转换单位：从 (radians/sample) 到 (radians/second)
    # 乘以采样率
    vel_rps = vel * sample_rate
    acc_rps2 = acc * sample_rate
    jerk_rps3 = jerk * sample_rate
    
    # 计算平均绝对值
    metrics = {
        'angular_velocity': np.abs(vel_rps).mean(),
        'angular_acceleration': np.abs(acc_rps2).mean(),
        'angular_jerk': np.abs(jerk_rps3).mean()
    }
    
    return metrics


def calculate_angle_mae(
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray = None
) -> float:
    """
    计算关节角度的平均绝对误差（考虑mask）
    
    Args:
        pred: 预测角度 (batch, time, joints) 或 (time, joints)
        target: 目标角度，形状与pred相同
        mask: 可选的mask (batch, time) 或 (time,)，True表示有效
        
    Returns:
        MAE值
    """
    if mask is not None:
        # 扩展mask以匹配关节维度
        if mask.ndim == 1 and pred.ndim == 2:
            # (time,) -> (time, 1)
            mask = mask[:, np.newaxis]
        elif mask.ndim == 2 and pred.ndim == 3:
            # (batch, time) -> (batch, time, 1)
            mask = mask[:, :, np.newaxis]
        
        # 应用mask
        valid_pred = pred[mask]
        valid_target = target[mask]
        
        if len(valid_pred) == 0:
            return 0.0
        
        return np.abs(valid_pred - valid_target).mean()
    else:
        return np.abs(pred - target).mean()


def calculate_per_finger_mae(
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray = None
) -> Dict[str, float]:
    """
    计算每个手指的MAE
    
    假设关节顺序为：
    - Thumb: 0-3
    - Index: 4-7
    - Middle: 8-11
    - Ring: 12-15
    - Pinky: 16-19
    
    Args:
        pred: 预测角度 (batch, time, 20) 或 (time, 20)
        target: 目标角度
        mask: 可选的mask
        
    Returns:
        每个手指的MAE字典
    """
    # 定义手指的关节索引
    finger_indices = {
        'thumb': list(range(0, 4)),
        'index': list(range(4, 8)),
        'middle': list(range(8, 12)),
        'ring': list(range(12, 16)),
        'pinky': list(range(16, 20))
    }
    
    per_finger_mae = {}
    
    for finger_name, indices in finger_indices.items():
        # 提取该手指的关节
        pred_finger = pred[..., indices]
        target_finger = target[..., indices]
        
        # 计算该手指的MAE
        mae = calculate_angle_mae(pred_finger, target_finger, mask)
        per_finger_mae[f'mae_{finger_name}'] = mae
    
    return per_finger_mae


def calculate_pd_groups_mae(
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray = None
) -> Dict[str, float]:
    """
    计算近端(proximal)/中端(mid)/远端(distal)关节组的MAE
    
    关节分组（基于emg2pose的定义）：
    - Proximal (近端): 每个手指的第一个关节
    - Mid (中端): 每个手指的第二个关节
    - Distal (远端): 每个手指的第三个关节
    
    Args:
        pred: 预测角度
        target: 目标角度
        mask: 可选的mask
        
    Returns:
        每个组的MAE字典
    """
    # 简化的分组（根据emg2pose的JOINTS定义）
    pd_indices = {
        'proximal': [0, 1, 4, 5, 8, 9, 12, 13, 16, 17],  # MCP关节
        'mid': [2, 6, 10, 14, 18],  # PIP关节
        'distal': [3, 7, 11, 15, 19]  # DIP/IP关节
    }
    
    pd_mae = {}
    
    for group_name, indices in pd_indices.items():
        # 提取该组的关节
        pred_group = pred[..., indices]
        target_group = target[..., indices]
        
        # 计算该组的MAE
        mae = calculate_angle_mae(pred_group, target_group, mask)
        pd_mae[f'mae_{group_name}'] = mae
    
    return pd_mae


def calculate_comprehensive_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray = None,
    sample_rate: float = 2000.0,
    include_derivatives: bool = True,
    include_per_finger: bool = True,
    include_pd_groups: bool = True
) -> Dict[str, float]:
    """
    计算综合指标（对齐版本）
    
    Args:
        pred: 预测角度
        target: 目标角度
        mask: 可选的mask
        sample_rate: 采样率
        include_derivatives: 是否包含导数指标
        include_per_finger: 是否包含每个手指的MAE
        include_pd_groups: 是否包含PD组的MAE
        
    Returns:
        综合指标字典
    """
    metrics = {}
    
    # 基本MAE
    metrics['mae'] = calculate_angle_mae(pred, target, mask)
    
    # 导数指标（角速度、角加速度、jerk）
    if include_derivatives:
        derivative_metrics = calculate_angular_derivatives(pred, sample_rate)
        metrics.update(derivative_metrics)
    
    # 每个手指的MAE
    if include_per_finger and pred.shape[-1] == 20:
        finger_metrics = calculate_per_finger_mae(pred, target, mask)
        metrics.update(finger_metrics)
    
    # PD组的MAE
    if include_pd_groups and pred.shape[-1] == 20:
        pd_metrics = calculate_pd_groups_mae(pred, target, mask)
        metrics.update(pd_metrics)
    
    return metrics