"""
模型评估器
"""

import os
import logging
from typing import Dict, Any, List, Optional
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.metrics import calculate_metrics, calculate_per_joint_metrics
from utils.visualization import plot_predictions, plot_error_distribution

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """模型评估器，提供详细的评估和可视化功能"""
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        save_dir: str = "evaluation_results"
    ):
        """
        初始化评估器
        
        Args:
            model: 要评估的模型
            device: 计算设备
            save_dir: 结果保存目录
        """
        self.model = model
        self.device = device
        self.save_dir = save_dir
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        # 损失函数
        self.criterion = nn.MSELoss()
        
    def evaluate_dataset(
        self, 
        dataloader: DataLoader,
        dataset_name: str = "test"
    ) -> Dict[str, Any]:
        """
        在指定数据集上评估模型
        
        Args:
            dataloader: 数据加载器
            dataset_name: 数据集名称
            
        Returns:
            评估结果字典
        """
        logger.info(f"开始评估 {dataset_name} 数据集...")
        
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        prediction_errors = []
        
        with torch.no_grad():
            for batch_idx, (emg_data, angle_data) in enumerate(dataloader):
                # 数据移到设备
                emg_data = emg_data.to(self.device)
                angle_data = angle_data.to(self.device)
                
                # 前向传播
                predictions = self.model(emg_data)
                loss = self.criterion(predictions, angle_data)
                
                # 计算误差
                batch_errors = torch.abs(predictions - angle_data)
                
                # 统计
                total_loss += loss.item()
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(angle_data.cpu().numpy())
                prediction_errors.append(batch_errors.cpu().numpy())
                
                if batch_idx % 50 == 0:
                    logger.info(f"已处理 {batch_idx + 1}/{len(dataloader)} 批次")
        
        # 合并所有批次的结果
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        prediction_errors = np.concatenate(prediction_errors, axis=0)
        
        # 计算整体指标
        avg_loss = total_loss / len(dataloader)
        overall_metrics = calculate_metrics(all_targets, all_predictions)
        
        # 计算每个关节的指标
        per_joint_metrics = calculate_per_joint_metrics(all_targets, all_predictions)
        
        # 创建评估结果
        results = {
            'dataset_name': dataset_name,
            'num_samples': len(all_predictions),
            'avg_loss': avg_loss,
            'overall_metrics': overall_metrics,
            'per_joint_metrics': per_joint_metrics,
            'predictions': all_predictions,
            'targets': all_targets,
            'errors': prediction_errors
        }
        
        logger.info(f"{dataset_name} 数据集评估完成")
        logger.info(f"样本数: {len(all_predictions)}")
        logger.info(f"平均损失: {avg_loss:.6f}")
        for metric_name, metric_value in overall_metrics.items():
            logger.info(f"{metric_name}: {metric_value:.6f}")
        
        return results
    
    def compare_models(
        self, 
        models: Dict[str, nn.Module],
        dataloader: DataLoader,
        dataset_name: str = "test"
    ) -> Dict[str, Any]:
        """
        比较多个模型在同一数据集上的性能
        
        Args:
            models: 模型字典 {model_name: model}
            dataloader: 数据加载器
            dataset_name: 数据集名称
            
        Returns:
            比较结果
        """
        logger.info(f"开始比较 {len(models)} 个模型...")
        
        comparison_results = {}
        
        for model_name, model in models.items():
            # 临时保存当前模型
            original_model = self.model
            self.model = model.to(self.device)
            
            # 评估当前模型
            results = self.evaluate_dataset(dataloader, f"{dataset_name}_{model_name}")
            comparison_results[model_name] = results
            
            # 恢复原模型
            self.model = original_model
        
        # 创建比较表格
        self._create_comparison_table(comparison_results, dataset_name)
        
        return comparison_results
    
    def _create_comparison_table(
        self, 
        comparison_results: Dict[str, Any],
        dataset_name: str
    ):
        """创建模型比较表格"""
        import pandas as pd
        
        # 收集所有模型的指标
        data = []
        for model_name, results in comparison_results.items():
            row = {'Model': model_name}
            row.update(results['overall_metrics'])
            row['Loss'] = results['avg_loss']
            data.append(row)
        
        # 创建DataFrame
        df = pd.DataFrame(data)
        
        # 保存为CSV
        csv_path = os.path.join(self.save_dir, f"model_comparison_{dataset_name}.csv")
        df.to_csv(csv_path, index=False)
        
        logger.info(f"模型比较表格已保存到: {csv_path}")
        
        return df
    
    def analyze_predictions(
        self, 
        predictions: np.ndarray,
        targets: np.ndarray,
        dataset_name: str = "test",
        num_samples: int = 5
    ):
        """
        分析预测结果并生成可视化
        
        Args:
            predictions: 预测结果
            targets: 真实值
            dataset_name: 数据集名称
            num_samples: 可视化的样本数量
        """
        logger.info("开始预测结果分析...")
        
        # 创建可视化目录
        viz_dir = os.path.join(self.save_dir, f"visualizations_{dataset_name}")
        os.makedirs(viz_dir, exist_ok=True)
        
        # 1. 预测vs真实值对比图
        self._plot_prediction_comparison(
            predictions, targets, viz_dir, num_samples
        )
        
        # 2. 误差分布图
        self._plot_error_distribution(
            predictions, targets, viz_dir
        )
        
        # 3. 每个关节的误差分析
        self._plot_per_joint_analysis(
            predictions, targets, viz_dir
        )
        
        # 4. 时序预测可视化
        self._plot_time_series_predictions(
            predictions, targets, viz_dir, num_samples
        )
        
        logger.info(f"预测分析完成，结果保存在: {viz_dir}")
    
    def _plot_prediction_comparison(
        self, 
        predictions: np.ndarray,
        targets: np.ndarray,
        save_dir: str,
        num_samples: int
    ):
        """绘制预测vs真实值对比图"""
        # 随机选择样本
        sample_indices = np.random.choice(len(predictions), min(num_samples, len(predictions)), replace=False)
        
        for i, idx in enumerate(sample_indices):
            fig, axes = plt.subplots(4, 5, figsize=(20, 16))
            fig.suptitle(f'Sample {idx}: Prediction vs Target for All Joints', fontsize=16)
            
            for joint_idx in range(min(20, predictions.shape[-1])):
                row = joint_idx // 5
                col = joint_idx % 5
                
                ax = axes[row, col]
                ax.plot(targets[idx, :, joint_idx], label='Target', alpha=0.7)
                ax.plot(predictions[idx, :, joint_idx], label='Prediction', alpha=0.7)
                ax.set_title(f'Joint {joint_idx}')
                ax.legend()
                ax.grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'prediction_comparison_sample_{i}.png'), dpi=150)
            plt.close()
    
    def _plot_error_distribution(
        self, 
        predictions: np.ndarray,
        targets: np.ndarray,
        save_dir: str
    ):
        """绘制误差分布图"""
        errors = np.abs(predictions - targets)
        
        # 整体误差分布
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.hist(errors.flatten(), bins=50, alpha=0.7, density=True)
        plt.title('Overall Error Distribution')
        plt.xlabel('Absolute Error')
        plt.ylabel('Density')
        plt.grid(True)
        
        # 每个关节的平均误差
        plt.subplot(2, 2, 2)
        joint_errors = errors.mean(axis=(0, 1))
        plt.bar(range(len(joint_errors)), joint_errors)
        plt.title('Average Error per Joint')
        plt.xlabel('Joint Index')
        plt.ylabel('Average Absolute Error')
        plt.grid(True)
        
        # 时间步误差
        plt.subplot(2, 2, 3)
        timestep_errors = errors.mean(axis=(0, 2))
        plt.plot(timestep_errors)
        plt.title('Average Error per Timestep')
        plt.xlabel('Timestep')
        plt.ylabel('Average Absolute Error')
        plt.grid(True)
        
        # 误差热力图
        plt.subplot(2, 2, 4)
        avg_errors = errors.mean(axis=0)
        im = plt.imshow(avg_errors.T, aspect='auto', cmap='hot')
        plt.colorbar(im)
        plt.title('Error Heatmap (Time x Joint)')
        plt.xlabel('Timestep')
        plt.ylabel('Joint Index')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'error_distribution.png'), dpi=150)
        plt.close()
    
    def _plot_per_joint_analysis(
        self, 
        predictions: np.ndarray,
        targets: np.ndarray,
        save_dir: str
    ):
        """绘制每个关节的详细分析"""
        num_joints = predictions.shape[-1]
        errors = np.abs(predictions - targets)
        
        # 每个关节的误差箱线图
        fig, axes = plt.subplots(4, 5, figsize=(20, 16))
        fig.suptitle('Per-Joint Error Analysis', fontsize=16)
        
        for joint_idx in range(min(20, num_joints)):
            row = joint_idx // 5
            col = joint_idx % 5
            
            ax = axes[row, col]
            joint_errors = errors[:, :, joint_idx].flatten()
            ax.boxplot(joint_errors)
            ax.set_title(f'Joint {joint_idx}')
            ax.set_ylabel('Absolute Error')
            ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'per_joint_error_boxplot.png'), dpi=150)
        plt.close()
    
    def _plot_time_series_predictions(
        self, 
        predictions: np.ndarray,
        targets: np.ndarray,
        save_dir: str,
        num_samples: int
    ):
        """绘制时序预测可视化"""
        sample_indices = np.random.choice(len(predictions), min(num_samples, len(predictions)), replace=False)
        
        for i, idx in enumerate(sample_indices):
            # 选择几个代表性关节
            selected_joints = [0, 4, 8, 12, 16] if predictions.shape[-1] > 16 else list(range(min(5, predictions.shape[-1])))
            
            fig, axes = plt.subplots(len(selected_joints), 1, figsize=(15, 3*len(selected_joints)))
            if len(selected_joints) == 1:
                axes = [axes]
                
            fig.suptitle(f'Time Series Prediction - Sample {idx}', fontsize=16)
            
            for j, joint_idx in enumerate(selected_joints):
                ax = axes[j]
                time_steps = range(predictions.shape[1])
                
                ax.plot(time_steps, targets[idx, :, joint_idx], 'b-', label='Target', linewidth=2)
                ax.plot(time_steps, predictions[idx, :, joint_idx], 'r--', label='Prediction', linewidth=2)
                
                ax.set_title(f'Joint {joint_idx}')
                ax.set_xlabel('Time Step')
                ax.set_ylabel('Joint Angle')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'time_series_sample_{i}.png'), dpi=150)
            plt.close()
    
    def generate_report(
        self, 
        evaluation_results: Dict[str, Any],
        save_path: Optional[str] = None
    ) -> str:
        """
        生成评估报告
        
        Args:
            evaluation_results: 评估结果
            save_path: 报告保存路径
            
        Returns:
            报告内容
        """
        if save_path is None:
            save_path = os.path.join(self.save_dir, "evaluation_report.txt")
        
        report = []
        report.append("=" * 80)
        report.append("模型评估报告")
        report.append("=" * 80)
        report.append()
        
        # 基本信息
        report.append(f"数据集: {evaluation_results['dataset_name']}")
        report.append(f"样本数: {evaluation_results['num_samples']}")
        report.append(f"平均损失: {evaluation_results['avg_loss']:.6f}")
        report.append()
        
        # 整体指标
        report.append("整体性能指标:")
        report.append("-" * 40)
        for metric_name, metric_value in evaluation_results['overall_metrics'].items():
            report.append(f"{metric_name:15s}: {metric_value:.6f}")
        report.append()
        
        # 每个关节的指标
        report.append("每个关节的性能指标:")
        report.append("-" * 40)
        per_joint = evaluation_results['per_joint_metrics']
        for joint_idx in range(len(per_joint['mse'])):
            report.append(f"关节 {joint_idx:2d} - MSE: {per_joint['mse'][joint_idx]:.6f}, "
                         f"MAE: {per_joint['mae'][joint_idx]:.6f}, "
                         f"R²: {per_joint['r2'][joint_idx]:.6f}")
        
        report.append()
        report.append("=" * 80)
        
        # 保存报告
        report_content = "\n".join(report)
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"评估报告已保存到: {save_path}")
        
        return report_content
