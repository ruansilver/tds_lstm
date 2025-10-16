"""
早停机制实现
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

class EarlyStopping:
    """
    早停机制，用于防止过拟合
    当监控指标在指定轮数内没有改善时停止训练
    """
    
    def __init__(
        self,
        patience: int = 10,  # 耐心等待的轮数
        min_delta: float = 1e-4,   # 最小改善幅度
        mode: str = 'min',   # 'min'表示指标越小越好，'max'表示指标越大越好    去检查
        restore_best_weights: bool = True,  # 是否恢复最佳权重
        verbose: bool = True  # 是否打印详细信息
    ):
        """
        初始化早停机制
        
        Args:
            patience: 耐心等待的轮数
            min_delta: 最小改善幅度
            mode: 监控模式，'min'表示越小越好，'max'表示越大越好
            restore_best_weights: 是否在停止时恢复最佳权重
            verbose: 是否打印详细信息
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode.lower()
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose
        
        # 验证模式
        if self.mode not in ['min', 'max']:
            raise ValueError(f"模式必须是 'min' 或 'max'，得到: {mode}")
        
        # 初始化状态
        self.reset()
        
        if self.verbose:
            logger.info(f"早停机制初始化完成")
            logger.info(f"  耐心: {patience} epochs")
            logger.info(f"  最小改善: {min_delta}")
            logger.info(f"  模式: {mode}")
    
    def reset(self):
        """重置早停状态"""
        self.wait = 0
        self.stopped_epoch = 0
        self.best_value = float('inf') if self.mode == 'min' else float('-inf')
        self.best_weights = None
        self.stop_training = False
    
    def __call__(self, current_value: float, model_weights: Optional[dict] = None) -> bool:
        """
        检查是否应该早停
        
        Args:
            current_value: 当前监控指标值
            model_weights: 当前模型权重（用于恢复最佳权重）
            
        Returns:
            是否应该停止训练
        """
        # 检查是否有改善
        if self._is_improvement(current_value):
            self.best_value = current_value
            self.wait = 0
            
            # 保存最佳权重
            if self.restore_best_weights and model_weights is not None:
                self.best_weights = model_weights.copy()
            
            if self.verbose:
                logger.info(f"指标改善: {current_value:.6f} (最佳: {self.best_value:.6f})")
        
        else:
            self.wait += 1
            if self.verbose:
                logger.info(f"指标未改善: {current_value:.6f} (最佳: {self.best_value:.6f}) "
                           f"等待: {self.wait}/{self.patience}")
            
            # 检查是否达到耐心限制
            if self.wait >= self.patience:
                self.stop_training = True
                self.stopped_epoch = self.wait
                
                if self.verbose:
                    logger.info(f"早停触发! 在 {self.patience} 轮后停止训练")
                    logger.info(f"最佳指标值: {self.best_value:.6f}")
        
        return self.stop_training
    
    def _is_improvement(self, current_value: float) -> bool:
        """
        检查当前值是否比最佳值有改善
        
        Args:
            current_value: 当前值
            
        Returns:
            是否有改善
        """
        if self.mode == 'min':
            return current_value < (self.best_value - self.min_delta)
        else:  # mode == 'max'
            return current_value > (self.best_value + self.min_delta)
    
    def should_stop(self) -> bool:
        """
        检查是否应该停止训练
        
        Returns:
            是否应该停止
        """
        return self.stop_training
    
    def get_best_weights(self) -> Optional[dict]:
        """
        获取最佳权重
        
        Returns:
            最佳权重字典，如果没有则返回None
        """
        return self.best_weights
    
    def get_info(self) -> dict:
        """
        获取早停信息
        
        Returns:
            早停状态信息
        """
        return {
            'patience': self.patience,
            'min_delta': self.min_delta,
            'mode': self.mode,
            'wait': self.wait,
            'best_value': self.best_value,
            'stopped_epoch': self.stopped_epoch,
            'stop_training': self.stop_training
        }
    
    def restore_best_model(self, model) -> bool:
        """
        恢复最佳模型权重
        
        Args:
            model: 要恢复的模型
            
        Returns:
            是否成功恢复
        """
        if self.best_weights is not None:
            try:
                model.load_state_dict(self.best_weights)
                if self.verbose:
                    logger.info(f"✅ 已恢复最佳模型权重（指标值: {self.best_value:.6f}）")
                return True
            except Exception as e:
                logger.error(f"❌ 恢复最佳权重失败: {str(e)}")
                return False
        else:
            logger.warning("⚠️  没有保存的最佳权重")
            return False
    
    def summary(self) -> str:
        """
        生成早停机制的总结报告
        
        Returns:
            总结文本
        """
        lines = [
            "=" * 60,
            "早停机制总结",
            "=" * 60,
            f"配置:",
            f"  - 耐心: {self.patience} epochs",
            f"  - 最小改善: {self.min_delta}",
            f"  - 模式: {self.mode}",
            "",
            f"结果:",
            f"  - 是否触发早停: {'是' if self.stop_training else '否'}",
            f"  - 等待轮数: {self.wait}/{self.patience}",
            f"  - 最佳指标值: {self.best_value:.6f}",
        ]
        
        if self.stop_training:
            lines.append(f"  - 停止于第 {self.stopped_epoch} 轮")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)


class ReduceLROnPlateau:
    """
    当指标停止改善时降低学习率
    """
    
    def __init__(
        self,
        factor: float = 0.5,
        patience: int = 5,
        min_delta: float = 1e-4,
        mode: str = 'min',
        min_lr: float = 1e-8,
        verbose: bool = True
    ):
        """
        初始化学习率衰减器
        
        Args:
            factor: 学习率衰减因子
            patience: 耐心等待的轮数
            min_delta: 最小改善幅度
            mode: 监控模式
            min_lr: 最小学习率
            verbose: 是否打印详细信息
        """
        self.factor = factor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode.lower()
        self.min_lr = min_lr
        self.verbose = verbose
        
        # 初始化状态
        self.wait = 0
        self.best_value = float('inf') if self.mode == 'min' else float('-inf')
        self.num_bad_epochs = 0
        
        if self.verbose:
            logger.info(f"学习率衰减器初始化完成")
            logger.info(f"  因子: {factor}")
            logger.info(f"  耐心: {patience} epochs")
            logger.info(f"  最小学习率: {min_lr}")
    
    def __call__(self, current_value: float, optimizer) -> bool:
        """
        检查是否应该降低学习率
        
        Args:
            current_value: 当前监控指标值
            optimizer: 优化器
            
        Returns:
            是否降低了学习率
        """
        # 检查是否有改善
        if self._is_improvement(current_value):
            self.best_value = current_value
            self.wait = 0
        else:
            self.wait += 1
            
            # 检查是否需要降低学习率
            if self.wait >= self.patience:
                old_lr = optimizer.param_groups[0]['lr']
                new_lr = max(old_lr * self.factor, self.min_lr)
                
                if new_lr < old_lr:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = new_lr
                    
                    if self.verbose:
                        logger.info(f"学习率从 {old_lr:.2e} 降低到 {new_lr:.2e}")
                    
                    self.wait = 0
                    self.num_bad_epochs += 1
                    return True
        
        return False
    
    def _is_improvement(self, current_value: float) -> bool:
        """检查当前值是否比最佳值有改善"""
        if self.mode == 'min':
            return current_value < (self.best_value - self.min_delta)
        else:
            return current_value > (self.best_value + self.min_delta)
