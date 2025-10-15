"""
可视化工具
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

# 设置中文字体和样式
plt.style.use('default')
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 支持中文显示
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
sns.set_palette("husl")

def plot_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    joint_indices: Optional[List[int]] = None,
    sample_indices: Optional[List[int]] = None,
    save_path: Optional[str] = None,
    title: str = "预测vs真实值对比",
    figsize: tuple = (15, 10)
):
    """
    绘制预测值与真实值的对比图
    
    Args:
        y_true: 真实值，形状为 (n_samples, seq_len, n_joints)
        y_pred: 预测值，形状与y_true相同
        joint_indices: 要显示的关节索引列表，默认显示前4个
        sample_indices: 要显示的样本索引列表，默认显示前2个
        save_path: 保存路径
        title: 图表标题
        figsize: 图像大小
    """
    if joint_indices is None:
        joint_indices = list(range(min(4, y_true.shape[-1])))
    
    if sample_indices is None:
        sample_indices = list(range(min(2, y_true.shape[0])))
    
    n_joints = len(joint_indices)
    n_samples = len(sample_indices)
    
    fig, axes = plt.subplots(n_samples, n_joints, figsize=figsize)
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    if n_joints == 1:
        axes = axes.reshape(-1, 1)
    
    fig.suptitle(title, fontsize=16)
    
    for i, sample_idx in enumerate(sample_indices):
        for j, joint_idx in enumerate(joint_indices):
            ax = axes[i, j]
            
            # 绘制真实值和预测值
            time_steps = range(y_true.shape[1])
            ax.plot(time_steps, y_true[sample_idx, :, joint_idx], 
                   'b-', label='真实值', linewidth=2, alpha=0.8)
            ax.plot(time_steps, y_pred[sample_idx, :, joint_idx], 
                   'r--', label='预测值', linewidth=2, alpha=0.8)
            
            ax.set_title(f'样本 {sample_idx}, 关节 {joint_idx}')
            ax.set_xlabel('时间步')
            ax.set_ylabel('关节角度')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"预测对比图已保存到: {save_path}")
    
    plt.show()

def plot_training_history(
    train_history: Dict[str, List[float]],
    val_history: Dict[str, List[float]],
    save_path: Optional[str] = None,
    figsize: tuple = (15, 5)
):
    """
    绘制训练历史
    
    Args:
        train_history: 训练历史 {'loss': [...], 'metrics': [...]}
        val_history: 验证历史
        save_path: 保存路径
        figsize: 图像大小
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    epochs = range(1, len(train_history['loss']) + 1)
    
    # 损失曲线
    axes[0].plot(epochs, train_history['loss'], 'b-', label='训练损失', linewidth=2)
    axes[0].plot(epochs, val_history['loss'], 'r-', label='验证损失', linewidth=2)
    axes[0].set_title('训练和验证损失')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # MSE指标
    if train_history['metrics'] and 'mse' in train_history['metrics'][0]:
        train_mse = [m['mse'] for m in train_history['metrics']]
        val_mse = [m['mse'] for m in val_history['metrics']]
        
        axes[1].plot(epochs, train_mse, 'b-', label='训练MSE', linewidth=2)
        axes[1].plot(epochs, val_mse, 'r-', label='验证MSE', linewidth=2)
        axes[1].set_title('MSE指标')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('MSE')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    # R²指标
    if train_history['metrics'] and 'r2' in train_history['metrics'][0]:
        train_r2 = [m['r2'] for m in train_history['metrics']]
        val_r2 = [m['r2'] for m in val_history['metrics']]
        
        axes[2].plot(epochs, train_r2, 'b-', label='训练R²', linewidth=2)
        axes[2].plot(epochs, val_r2, 'r-', label='验证R²', linewidth=2)
        axes[2].set_title('R²指标')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('R²')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"训练历史图已保存到: {save_path}")
    
    plt.show()

def plot_error_distribution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[str] = None,
    figsize: tuple = (12, 8)
):
    """
    绘制误差分布图
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        save_path: 保存路径
        figsize: 图像大小
    """
    errors = np.abs(y_true - y_pred)
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # 整体误差分布直方图
    axes[0, 0].hist(errors.flatten(), bins=50, alpha=0.7, density=True, color='skyblue')
    axes[0, 0].set_title('整体误差分布')
    axes[0, 0].set_xlabel('绝对误差')
    axes[0, 0].set_ylabel('密度')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 每个关节的平均误差
    joint_errors = errors.mean(axis=(0, 1))
    axes[0, 1].bar(range(len(joint_errors)), joint_errors, color='lightcoral')
    axes[0, 1].set_title('每个关节的平均误差')
    axes[0, 1].set_xlabel('关节索引')
    axes[0, 1].set_ylabel('平均绝对误差')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 时间步误差变化
    timestep_errors = errors.mean(axis=(0, 2))
    axes[1, 0].plot(timestep_errors, color='green', linewidth=2)
    axes[1, 0].set_title('各时间步的平均误差')
    axes[1, 0].set_xlabel('时间步')
    axes[1, 0].set_ylabel('平均绝对误差')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 误差热力图
    avg_errors = errors.mean(axis=0)
    im = axes[1, 1].imshow(avg_errors.T, aspect='auto', cmap='hot', origin='lower')
    axes[1, 1].set_title('误差热力图 (时间 x 关节)')
    axes[1, 1].set_xlabel('时间步')
    axes[1, 1].set_ylabel('关节索引')
    plt.colorbar(im, ax=axes[1, 1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"误差分布图已保存到: {save_path}")
    
    plt.show()

def plot_joint_comparison(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    joint_idx: int,
    sample_indices: Optional[List[int]] = None,
    save_path: Optional[str] = None,
    figsize: tuple = (15, 8)
):
    """
    绘制特定关节在多个样本上的预测对比
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        joint_idx: 关节索引
        sample_indices: 样本索引列表
        save_path: 保存路径
        figsize: 图像大小
    """
    if sample_indices is None:
        sample_indices = list(range(min(4, y_true.shape[0])))
    
    n_samples = len(sample_indices)
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    
    for i, sample_idx in enumerate(sample_indices[:4]):
        if i >= 4:
            break
            
        ax = axes[i]
        time_steps = range(y_true.shape[1])
        
        ax.plot(time_steps, y_true[sample_idx, :, joint_idx], 
               'b-', label='真实值', linewidth=2, alpha=0.8)
        ax.plot(time_steps, y_pred[sample_idx, :, joint_idx], 
               'r--', label='预测值', linewidth=2, alpha=0.8)
        
        # 计算误差
        error = np.abs(y_true[sample_idx, :, joint_idx] - y_pred[sample_idx, :, joint_idx])
        mae = np.mean(error)
        
        ax.set_title(f'样本 {sample_idx} - 关节 {joint_idx} (MAE: {mae:.4f})')
        ax.set_xlabel('时间步')
        ax.set_ylabel('关节角度')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"关节对比图已保存到: {save_path}")
    
    plt.show()

def plot_model_comparison(
    comparison_results: Dict[str, Dict[str, Any]],
    metrics: List[str] = ['mse', 'mae', 'r2'],
    save_path: Optional[str] = None,
    figsize: tuple = (12, 4)
):
    """
    绘制多个模型的性能对比
    
    Args:
        comparison_results: 模型比较结果
        metrics: 要对比的指标
        save_path: 保存路径
        figsize: 图像大小
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    if n_metrics == 1:
        axes = [axes]
    
    model_names = list(comparison_results.keys())
    
    for i, metric in enumerate(metrics):
        values = []
        for model_name in model_names:
            if metric in comparison_results[model_name]['overall_metrics']:
                values.append(comparison_results[model_name]['overall_metrics'][metric])
            else:
                values.append(0)
        
        bars = axes[i].bar(model_names, values, alpha=0.7)
        axes[i].set_title(f'{metric.upper()} 对比')
        axes[i].set_ylabel(metric.upper())
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, value in zip(bars, values):
            axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(values),
                        f'{value:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"模型对比图已保存到: {save_path}")
    
    plt.show()

def create_animation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_idx: int = 0,
    joint_indices: Optional[List[int]] = None,
    save_path: Optional[str] = None,
    interval: int = 100
):
    """
    创建预测动画
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        sample_idx: 样本索引
        joint_indices: 关节索引列表
        save_path: 保存路径
        interval: 动画间隔（毫秒）
    """
    try:
        from matplotlib.animation import FuncAnimation
    except ImportError:
        logger.warning("matplotlib.animation不可用，跳过动画创建")
        return
    
    if joint_indices is None:
        joint_indices = list(range(min(4, y_true.shape[-1])))
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    lines_true = []
    lines_pred = []
    
    for i, joint_idx in enumerate(joint_indices[:4]):
        ax = axes[i]
        ax.set_xlim(0, y_true.shape[1])
        ax.set_ylim(np.min([y_true[sample_idx, :, joint_idx], y_pred[sample_idx, :, joint_idx]]) - 0.1,
                   np.max([y_true[sample_idx, :, joint_idx], y_pred[sample_idx, :, joint_idx]]) + 0.1)
        
        line_true, = ax.plot([], [], 'b-', label='真实值', linewidth=2)
        line_pred, = ax.plot([], [], 'r--', label='预测值', linewidth=2)
        
        lines_true.append(line_true)
        lines_pred.append(line_pred)
        
        ax.set_title(f'关节 {joint_idx}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def animate(frame):
        for i, joint_idx in enumerate(joint_indices[:4]):
            x_data = range(frame + 1)
            y_true_data = y_true[sample_idx, :frame+1, joint_idx]
            y_pred_data = y_pred[sample_idx, :frame+1, joint_idx]
            
            lines_true[i].set_data(x_data, y_true_data)
            lines_pred[i].set_data(x_data, y_pred_data)
        
        return lines_true + lines_pred
    
    anim = FuncAnimation(fig, animate, frames=y_true.shape[1], 
                        interval=interval, blit=True, repeat=True)
    
    if save_path:
        anim.save(save_path, writer='pillow', fps=10)
        logger.info(f"动画已保存到: {save_path}")
    
    plt.show()
    return anim
