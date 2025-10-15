"""
模型训练器
"""

import os
import time
import logging
from typing import Dict, Any, Optional, Tuple
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from config import Config
from utils.metrics import calculate_metrics
from utils.checkpoint import CheckpointManager
from utils.early_stopping import EarlyStopping

logger = logging.getLogger(__name__)

# TensorBoard导入处理
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    SummaryWriter = None
    TENSORBOARD_AVAILABLE = False
    logger.warning("TensorBoard不可用，将跳过可视化日志")


class EMG2PoseTrainer:
    """EMG到姿态预测的训练器"""
    
    def __init__(
        self,
        model: nn.Module,
        config: Config,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader] = None
    ):
        """
        初始化训练器
        
        Args:
            model: 要训练的模型
            config: 配置对象
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            test_loader: 测试数据加载器（可选）
        """
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        # 设备配置
        self.device = torch.device(config.training.device)
        self.model.to(self.device)
        
        # 损失函数
        self.criterion = self._create_criterion()
        
        # 优化器
        self.optimizer = self._create_optimizer()
        
        # 学习率调度器
        self.scheduler = self._create_scheduler()
        self.warmup_scheduler = self._create_warmup_scheduler() if config.training.warmup_epochs > 0 else None
        
        # 早停机制
        self.early_stopping = EarlyStopping(
            patience=config.training.patience,
            min_delta=config.training.min_delta,
            mode='min'
        ) if config.training.early_stopping else None
        
        # 检查点管理器
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=config.logging.checkpoint_dir,
            save_best_only=config.logging.save_best_only,
            monitor=config.logging.monitor,
            mode=config.logging.mode
        )
        
        # TensorBoard写入器（如果可用）
        if TENSORBOARD_AVAILABLE and SummaryWriter is not None:
            try:
                self.writer = SummaryWriter(config.logging.log_dir)
                logger.info(f"✅ TensorBoard初始化成功，日志目录: {config.logging.log_dir}")
            except Exception as e:
                logger.warning(f"⚠️ TensorBoard初始化失败: {e}")
                self.writer = None
        else:
            self.writer = None
        
        # 训练状态
        self.current_epoch = 0    # 当前训练的epoch
        self.global_step = 0     # 全局训练步数
        self.best_val_loss = float('inf')    # 最佳验证损失
        self.train_history = {'loss': [], 'metrics': []}    # 训练历史记录
        self.val_history = {'loss': [], 'metrics': []}   # 验证历史记录
        
    def _create_criterion(self):
        """创建损失函数"""
        loss_type = getattr(self.config.training, 'loss_type', 'mse')
        
        if loss_type.lower() == 'mse':
            return nn.MSELoss()
        elif loss_type.lower() == 'mae':
            return nn.L1Loss()
        elif loss_type.lower() == 'mse+mae':
            # 组合损失
            mae_weight = getattr(self.config.training, 'mae_weight', 0.5)
            mse_criterion = nn.MSELoss()
            mae_criterion = nn.L1Loss()
            
            def combined_loss(pred, target):
                mse = mse_criterion(pred, target)
                mae = mae_criterion(pred, target)
                return mse + mae_weight * mae
            
            return combined_loss
        else:
            logger.warning(f"未知的损失类型: {loss_type}，使用默认MSE")
            return nn.MSELoss()
    
    def _create_optimizer(self) -> optim.Optimizer:
        """创建优化器"""
        if self.config.training.optimizer.lower() == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay
            )
        elif self.config.training.optimizer.lower() == 'adamw':
            return optim.AdamW(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay
            )
        elif self.config.training.optimizer.lower() == 'sgd':
            return optim.SGD(
                self.model.parameters(),     
                lr=self.config.training.learning_rate,
                momentum=0.9,
                weight_decay=self.config.training.weight_decay
            )
        else:
            raise ValueError(f"不支持的优化器: {self.config.training.optimizer}")
    
    def _create_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """创建学习率调度器"""
        scheduler_type = self.config.training.scheduler.lower()
        
        if scheduler_type == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.training.step_size,
                gamma=self.config.training.gamma
            )
        elif scheduler_type == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.training.num_epochs
            )
        elif scheduler_type == 'plateau':  
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                verbose=True
            )
        elif scheduler_type == 'none':
            return None
        else:
            return None
    
    def _create_warmup_scheduler(self):
        """创建warmup学习率调度器"""
        warmup_epochs = getattr(self.config.training, 'warmup_epochs', 0)
        if warmup_epochs <= 0:
            return None
        
        # 使用线性warmup
        def warmup_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            return 1.0
        
        return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=warmup_lambda)
    
    def train_epoch(self) -> Dict[str, float]:   
        """训练一个epoch"""
        self.model.train()  # 设置模型为训练模式
        total_loss = 0.0   # 累计损失
        all_predictions = []   # 预测结果
        all_targets = []  # 真实标签
        
        # 梯度累积步数
        accumulation_steps = getattr(self.config.training, 'gradient_accumulation_steps', 1)
        
        for batch_idx, (emg_data, angle_data) in enumerate(self.train_loader):
            # 数据移到设备
            emg_data = emg_data.to(self.device)
            angle_data = angle_data.to(self.device)
            
            # 前向传播
            predictions = self.model(emg_data)  # 2. 前向传播
            loss = self.criterion(predictions, angle_data) # 3. 计算损失
            
            # 梯度累积：损失需要除以累积步数
            loss = loss / accumulation_steps
            
            # 反向传播
            loss.backward()     # 4. 反向传播计算梯度
            
            # 每accumulation_steps步或最后一步才更新参数
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(self.train_loader):
                # 梯度裁剪
                grad_clip_norm = getattr(self.config.training, 'gradient_clip_norm', 1.0)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=grad_clip_norm)
                
                # 优化器步骤
                self.optimizer.step()  # 5. 优化器更新参数
                self.optimizer.zero_grad()    # 6. 清零梯度
            
            # 统计（使用原始loss，不是除以accumulation_steps后的）
            total_loss += loss.item() * accumulation_steps
            all_predictions.append(predictions.detach().cpu().numpy())
            all_targets.append(angle_data.detach().cpu().numpy())
            
            # 内存清理：及时释放不需要的tensor
            del emg_data, angle_data, predictions
            if batch_idx % 100 == 0:
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # 日志记录   使用tensorboard记录训练过程
            if batch_idx % self.config.logging.log_every == 0:
                if self.writer:
                    self.writer.add_scalar('Train/BatchLoss', loss.item(), self.global_step)
                    self.writer.add_scalar('Train/LearningRate', 
                                         self.optimizer.param_groups[0]['lr'], self.global_step)
                
                logger.info(f'Epoch: {self.current_epoch}, Batch: {batch_idx}/{len(self.train_loader)}, '
                           f'Loss: {loss.item():.6f}')
            
            self.global_step += 1
        
        # 计算epoch平均损失和指标
        avg_loss = total_loss / len(self.train_loader)      #平均损失
        all_predictions = np.concatenate(all_predictions, axis=0) #将所有批次的预测结果拼接起来
        all_targets = np.concatenate(all_targets, axis=0) #将所有批次的真实标签拼接起来
        metrics = calculate_metrics(all_targets, all_predictions) #计算指标
        
        return {'loss': avg_loss, **metrics}
    
    def validate_epoch(self) -> Dict[str, float]:
        """验证一个epoch"""
        self.model.eval() # 设置模型为评估模式
        total_loss = 0.0 # 累计损失
        all_predictions = []
        all_targets = []
        
        with torch.no_grad(): # 禁用梯度计算
            for batch_idx, (emg_data, angle_data) in enumerate(self.val_loader):
                # 数据移到设备
                emg_data = emg_data.to(self.device)
                angle_data = angle_data.to(self.device)
                
                # 前向传播
                predictions = self.model(emg_data)
                loss = self.criterion(predictions, angle_data)
                
                # 统计
                total_loss += loss.item()
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(angle_data.cpu().numpy())
                
                # 内存清理
                del emg_data, angle_data, predictions
                if batch_idx % 50 == 0:
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # 计算epoch平均损失和指标
        avg_loss = total_loss / len(self.val_loader)
        all_predictions = np.concatenate(all_predictions, axis=0)  #将所有批次的预测结果拼接起来
        all_targets = np.concatenate(all_targets, axis=0)  #将所有批次的真实标签拼接起来
        metrics = calculate_metrics(all_targets, all_predictions)  #计算指标并保存在一个字典中（指标在metrics中）
        
        return {'loss': avg_loss, **metrics}    #返回包含损失和指标的字典
    
    def train(self) -> Dict[str, Any]:
        """执行完整的训练过程"""
        logger.info("开始训练...")
        logger.info(f"设备: {self.device}")
        logger.info(f"训练样本数: {len(self.train_loader.dataset)}")
        logger.info(f"验证样本数: {len(self.val_loader.dataset)}")
        
        start_time = time.time()
        
        for epoch in range(self.config.training.num_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            # 训练阶段
            train_results = self.train_epoch()
            self.train_history['loss'].append(train_results['loss'])     #每个epoch的训练损失
            self.train_history['metrics'].append({k: v for k, v in train_results.items() if k != 'loss'})  #每个epoch的训练指标
            
            # 验证阶段
            val_results = self.validate_epoch()
            self.val_history['loss'].append(val_results['loss'])
            self.val_history['metrics'].append({k: v for k, v in val_results.items() if k != 'loss'})
            
            # 学习率调度
            warmup_epochs = getattr(self.config.training, 'warmup_epochs', 0)
            
            if epoch < warmup_epochs and self.warmup_scheduler:
                # Warmup阶段
                self.warmup_scheduler.step()
            elif self.scheduler:
                # 正常调度
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_results['loss'])
                else:
                    self.scheduler.step()
            
            # 记录到TensorBoard
            if self.writer:
                self.writer.add_scalar('Train/Loss', train_results['loss'], epoch)
                self.writer.add_scalar('Val/Loss', val_results['loss'], epoch)
                for metric_name, metric_value in train_results.items():
                    if metric_name != 'loss':
                        self.writer.add_scalar(f'Train/{metric_name}', metric_value, epoch)
                for metric_name, metric_value in val_results.items():
                    if metric_name != 'loss':
                        self.writer.add_scalar(f'Val/{metric_name}', metric_value, epoch)
            
            # 检查并保存最佳检查点
            is_best = val_results['loss'] < self.best_val_loss   
            if is_best:
                self.best_val_loss = val_results['loss']
            
            if epoch % self.config.logging.save_every == 0 or is_best:
                self.checkpoint_manager.save_checkpoint(
                    epoch=epoch,
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    loss=val_results['loss'],
                    metrics=val_results,
                    is_best=is_best
                )
            
            # 早停检查
            if self.early_stopping:
                self.early_stopping(val_results['loss'])
                if self.early_stopping.should_stop():
                    logger.info(f"早停触发，在第 {epoch} 轮停止训练")
                    break
            
            # 打印epoch结果
            epoch_time = time.time() - epoch_start_time
            logger.info(f'Epoch {epoch}/{self.config.training.num_epochs} - '
                       f'Time: {epoch_time:.2f}s - '
                       f'Train Loss: {train_results["loss"]:.6f} - '
                       f'Val Loss: {val_results["loss"]:.6f} - '
                       f'Val MSE: {val_results.get("mse", 0):.6f}')
        
        total_time = time.time() - start_time
        logger.info(f"训练完成，总用时: {total_time:.2f}秒")
        
        # 关闭TensorBoard写入器
        if self.writer:
            self.writer.close()
        
        return {
            'train_history': self.train_history,
            'val_history': self.val_history,
            'best_val_loss': self.best_val_loss,
            'total_time': total_time
        }
    
    def evaluate(self) -> Dict[str, float]:
        """在测试集上评估模型"""
        if self.test_loader is None:
            logger.warning("没有提供测试数据加载器")
            return {}
        
        logger.info("开始测试集评估...")
        
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for emg_data, angle_data in self.test_loader:
                # 数据移到设备
                emg_data = emg_data.to(self.device)
                angle_data = angle_data.to(self.device)
                
                # 前向传播
                predictions = self.model(emg_data)
                loss = self.criterion(predictions, angle_data)
                
                # 统计
                total_loss += loss.item()
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(angle_data.cpu().numpy())
        
        # 计算平均损失和指标
        avg_loss = total_loss / len(self.test_loader)
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        metrics = calculate_metrics(all_targets, all_predictions)
        
        test_results = {'loss': avg_loss, **metrics}
        
        logger.info("测试集评估结果:")
        for metric_name, metric_value in test_results.items():
            logger.info(f"  {metric_name}: {metric_value:.6f}")
        
        return test_results
