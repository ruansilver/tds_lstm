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
