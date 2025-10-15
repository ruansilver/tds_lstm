"""
模型检查点管理
"""

import os
import torch
import logging
from typing import Dict, Any, Optional
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class CheckpointManager:
    """
    模型检查点管理器
    负责保存和加载模型、优化器状态等
    """
    
    def __init__(
        self,
        checkpoint_dir: str = "checkpoints",
        save_best_only: bool = True,
        monitor: str = "val_loss",
        mode: str = "min",
        save_optimizer: bool = True,
        save_scheduler: bool = True
    ):
        """
        初始化检查点管理器
        
        Args:
            checkpoint_dir: 检查点保存目录
            save_best_only: 是否只保存最佳模型
            monitor: 监控的指标名称
            mode: 监控模式，'min'或'max'
            save_optimizer: 是否保存优化器状态
            save_scheduler: 是否保存调度器状态
        """
        self.checkpoint_dir = checkpoint_dir
        self.save_best_only = save_best_only
        self.monitor = monitor
        self.mode = mode
        self.save_optimizer = save_optimizer
        self.save_scheduler = save_scheduler
        
        # 创建检查点目录
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 跟踪最佳值
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.best_epoch = 0
        
        # 检查点历史
        self.checkpoint_history = []
        
        logger.info(f"检查点管理器初始化完成，保存目录: {checkpoint_dir}")
        logger.info(f"监控指标: {monitor} ({mode})")
    
    def save_checkpoint(
        self,
        epoch: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer = None,
        scheduler: torch.optim.lr_scheduler._LRScheduler = None,
        loss: float = None,
        metrics: Dict[str, float] = None,
        is_best: bool = False,
        extra_info: Dict[str, Any] = None
    ) -> str:
        """
        保存检查点
        
        Args:
            epoch: 当前epoch
            model: 模型
            optimizer: 优化器
            scheduler: 学习率调度器
            loss: 当前损失
            metrics: 评估指标
            is_best: 是否为最佳模型
            extra_info: 额外信息
            
        Returns:
            保存的文件路径
        """
        # 准备检查点数据
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'loss': loss,
            'metrics': metrics or {},
            'timestamp': datetime.now().isoformat(),
            'extra_info': extra_info or {}
        }
        
        # 保存优化器状态
        if optimizer and self.save_optimizer:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        # 保存调度器状态
        if scheduler and self.save_scheduler:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        # 确定保存文件名
        if self.save_best_only and not is_best:
            # 如果只保存最佳模型且当前不是最佳，则不保存
            return None
        
        if is_best:
            filename = "best_model.pth"
            self.best_epoch = epoch
            if metrics and self.monitor in metrics:
                self.best_value = metrics[self.monitor]
        else:
            filename = f"checkpoint_epoch_{epoch:04d}.pth"
        
        filepath = os.path.join(self.checkpoint_dir, filename)
        
        # 保存检查点
        try:
            torch.save(checkpoint, filepath)
            
            # 更新历史记录
            self.checkpoint_history.append({
                'epoch': epoch,
                'filepath': filepath,
                'loss': loss,
                'metrics': metrics,
                'is_best': is_best,
                'timestamp': checkpoint['timestamp']
            })
            
            # 保存元数据
            self._save_metadata()
            
            logger.info(f"检查点已保存: {filepath}")
            if is_best:
                logger.info(f"新的最佳模型! {self.monitor}: {self.best_value:.6f}")
            
            return filepath
            
        except Exception as e:
            logger.error(f"保存检查点失败: {str(e)}")
            return None
    
    def load_checkpoint(
        self,
        filepath: str,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer = None,
        scheduler: torch.optim.lr_scheduler._LRScheduler = None,
        device: torch.device = None
    ) -> Dict[str, Any]:
        """
        加载检查点
        
        Args:
            filepath: 检查点文件路径
            model: 模型
            optimizer: 优化器
            scheduler: 学习率调度器
            device: 设备
            
        Returns:
            检查点信息
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"检查点文件不存在: {filepath}")
        
        try:
            # 加载检查点
            if device:
                checkpoint = torch.load(filepath, map_location=device)
            else:
                checkpoint = torch.load(filepath)
            
            # 加载模型状态
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # 加载优化器状态
            if optimizer and 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # 加载调度器状态
            if scheduler and 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            logger.info(f"检查点加载完成: {filepath}")
            logger.info(f"Epoch: {checkpoint['epoch']}, Loss: {checkpoint.get('loss', 'N/A')}")
            
            return checkpoint
            
        except Exception as e:
            logger.error(f"加载检查点失败: {str(e)}")
            raise
    
    def load_best_model(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer = None,
        scheduler: torch.optim.lr_scheduler._LRScheduler = None,
        device: torch.device = None
    ) -> Dict[str, Any]:
        """
        加载最佳模型
        
        Args:
            model: 模型
            optimizer: 优化器
            scheduler: 学习率调度器
            device: 设备
            
        Returns:
            检查点信息
        """
        best_model_path = os.path.join(self.checkpoint_dir, "best_model.pth")
        return self.load_checkpoint(best_model_path, model, optimizer, scheduler, device)
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """
        获取最新的检查点文件路径
        
        Returns:
            最新检查点的文件路径，如果没有则返回None
        """
        if not self.checkpoint_history:
            return None
        
        # 按epoch排序，返回最新的
        latest = max(self.checkpoint_history, key=lambda x: x['epoch'])
        return latest['filepath']
    
    def list_checkpoints(self) -> list:
        """
        列出所有检查点
        
        Returns:
            检查点信息列表
        """
        return sorted(self.checkpoint_history, key=lambda x: x['epoch'])
    
    def clean_old_checkpoints(self, keep_last_n: int = 5):
        """
        清理旧的检查点，只保留最新的N个
        
        Args:
            keep_last_n: 保留的检查点数量
        """
        if len(self.checkpoint_history) <= keep_last_n:
            return
        
        # 按epoch排序
        sorted_checkpoints = sorted(self.checkpoint_history, key=lambda x: x['epoch'])
        
        # 要删除的检查点（保留最新的N个和最佳模型）
        to_delete = sorted_checkpoints[:-keep_last_n]
        
        for checkpoint_info in to_delete:
            if not checkpoint_info['is_best']:  # 不删除最佳模型
                try:
                    if os.path.exists(checkpoint_info['filepath']):
                        os.remove(checkpoint_info['filepath'])
                        logger.info(f"已删除旧检查点: {checkpoint_info['filepath']}")
                    
                    # 从历史记录中移除
                    self.checkpoint_history.remove(checkpoint_info)
                    
                except Exception as e:
                    logger.warning(f"删除检查点失败: {str(e)}")
        
        # 更新元数据
        self._save_metadata()
    
    def _save_metadata(self):
        """保存元数据"""
        metadata = {
            'best_value': self.best_value,
            'best_epoch': self.best_epoch,
            'monitor': self.monitor,
            'mode': self.mode,
            'checkpoint_history': self.checkpoint_history
        }
        
        metadata_path = os.path.join(self.checkpoint_dir, "metadata.json")
        try:
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"保存元数据失败: {str(e)}")
    
    def load_metadata(self) -> bool:
        """
        加载元数据
        
        Returns:
            是否成功加载
        """
        metadata_path = os.path.join(self.checkpoint_dir, "metadata.json")
        if not os.path.exists(metadata_path):
            return False
        
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            self.best_value = metadata.get('best_value', self.best_value)
            self.best_epoch = metadata.get('best_epoch', self.best_epoch)
            self.checkpoint_history = metadata.get('checkpoint_history', [])
            
            logger.info("元数据加载完成")
            return True
            
        except Exception as e:
            logger.warning(f"加载元数据失败: {str(e)}")
            return False
    
    def get_checkpoint_info(self) -> Dict[str, Any]:
        """
        获取检查点管理器信息
        
        Returns:
            检查点信息字典
        """
        return {
            'checkpoint_dir': self.checkpoint_dir,
            'best_value': self.best_value,
            'best_epoch': self.best_epoch,
            'monitor': self.monitor,
            'mode': self.mode,
            'num_checkpoints': len(self.checkpoint_history),
            'save_best_only': self.save_best_only
        }
