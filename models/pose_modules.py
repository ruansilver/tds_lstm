"""
姿态模块 - 对齐版本
实现BasePoseModule和StatePoseModule，处理context和时间对齐
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# 常量
EMG_SAMPLE_RATE = 2000  # EMG采样率


class BasePoseModule(nn.Module):
    """
    基础姿态模块
    
    包含一个network（encoder），处理left/right context和时间对齐
    预测跨越inputs[left_context : -right_context]，并重采样以匹配输入采样率
    
    Args:
        network: 编码器网络（必须有left_context和right_context属性）
        out_channels: 输出通道数（关节角度数）
    """
    
    def __init__(
        self,
        network: nn.Module,
        out_channels: int = 20
    ):
        super().__init__()
        self.network = network
        self.out_channels = out_channels
        
        # 从网络获取context信息
        self.left_context = getattr(network, 'left_context', 0)
        self.right_context = getattr(network, 'right_context', 0)
        
        logger.info(f"BasePoseModule初始化:")
        logger.info(f"  left_context: {self.left_context}")
        logger.info(f"  right_context: {self.right_context}")
        logger.info(f"  out_channels: {self.out_channels}")
    
    def forward(
        self, 
        batch: Dict[str, torch.Tensor], 
        provide_initial_pos: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            batch: 字典包含 {'emg': BCT, 'joint_angles': BCT, 'no_ik_failure': BT}
            provide_initial_pos: 是否提供初始位置给模型
            
        Returns:
            (pred, target, no_ik_failure) 元组
                - pred: (B, C, T') 预测的关节角度
                - target: (B, C, T') 对齐后的目标角度
                - no_ik_failure: (B, T') 对齐后的mask
        """
        emg = batch['emg']  # (B, C, T)
        joint_angles = batch['joint_angles']  # (B, C, T)
        no_ik_failure = batch['no_ik_failure']  # (B, T)
        
        # 获取初始位置（如果需要）
        initial_pos = joint_angles[..., self.left_context]  # (B, C)
        if not provide_initial_pos:
            initial_pos = torch.zeros_like(initial_pos)
        
        # 生成预测
        pred = self._predict_pose(emg, initial_pos)  # (B, C, T')
        
        # 裁剪joint_angles以匹配预测的时间范围
        start = self.left_context
        stop = None if self.right_context == 0 else -self.right_context
        joint_angles = joint_angles[..., slice(start, stop)]  # (B, C, T')
        no_ik_failure = no_ik_failure[..., slice(start, stop)]  # (B, T')
        
        # 对齐预测和目标的采样率
        n_time = joint_angles.shape[-1]
        pred = self.align_predictions(pred, n_time)
        no_ik_failure = self.align_mask(no_ik_failure, n_time)
        
        return pred, joint_angles, no_ik_failure
    
    def _predict_pose(
        self, 
        emg: torch.Tensor, 
        initial_pos: torch.Tensor
    ) -> torch.Tensor:
        """
        预测姿态（子类需要实现）
        
        Args:
            emg: (B, C, T) EMG输入
            initial_pos: (B, C) 初始位置
            
        Returns:
            (B, C, T') 预测结果
        """
        raise NotImplementedError
    
    def align_predictions(
        self, 
        pred: torch.Tensor, 
        n_time: int
    ) -> torch.Tensor:
        """
        时间重采样，将预测对齐到目标长度
        
        Args:
            pred: (B, C, T) 预测结果
            n_time: 目标时间长度
            
        Returns:
            (B, C, n_time) 对齐后的预测
        """
        if pred.shape[-1] == n_time:
            return pred
        return F.interpolate(pred, size=n_time, mode='linear', align_corners=True)
    
    def align_mask(
        self, 
        mask: torch.Tensor, 
        n_time: int
    ) -> torch.Tensor:
        """
        时间重采样，将mask对齐到目标长度
        
        Args:
            mask: (B, T) mask
            n_time: 目标时间长度
            
        Returns:
            (B, n_time) 对齐后的mask
        """
        if mask.shape[-1] == n_time:
            return mask
        
        # 添加channel维度用于interpolate
        mask = mask[:, None].to(torch.float32)  # (B, 1, T)
        aligned = F.interpolate(mask, size=n_time, mode='nearest')  # (B, 1, n_time)
        return aligned.squeeze(1).to(torch.bool)  # (B, n_time)


class PoseModule(BasePoseModule):
    """
    简单姿态模块
    
    通过预测位置或速度来跟踪姿态，可选地使用初始状态
    
    Args:
        network: 编码器网络
        out_channels: 输出通道数
        predict_vel: 是否预测速度（False则预测位置）
    """
    
    def __init__(
        self, 
        network: nn.Module, 
        out_channels: int = 20,
        predict_vel: bool = False
    ):
        super().__init__(network, out_channels)
        self.predict_vel = predict_vel
        
        logger.info(f"PoseModule初始化:")
        logger.info(f"  predict_vel: {predict_vel}")
    
    def _predict_pose(
        self, 
        emg: torch.Tensor, 
        initial_pos: torch.Tensor
    ) -> torch.Tensor:
        """
        预测姿态
        
        Args:
            emg: (B, C, T) EMG输入
            initial_pos: (B, C) 初始位置
            
        Returns:
            (B, C, T) 预测结果
        """
        pred = self.network(emg)  # (B, C, T)
        
        if self.predict_vel:
            # 速度模式：累积速度从初始位置
            pred = initial_pos[..., None] + torch.cumsum(pred, dim=-1)
        
        return pred


class StatePoseModule(BasePoseModule):
    """
    状态姿态模块
    
    通过预测位置或速度来跟踪姿态，每个时间点都以前一状态为条件
    
    Args:
        network: 编码器网络
        decoder: 解码器（SequentialLSTM或MLP）
        out_channels: 输出通道数
        state_condition: 是否使用状态条件（将前一输出作为输入）
        predict_vel: 是否预测速度
        rollout_freq: rollout频率（Hz）
    """
    
    def __init__(
        self,
        network: nn.Module,
        decoder: nn.Module,
        out_channels: int = 20,
        state_condition: bool = True,
        predict_vel: bool = False,
        rollout_freq: int = 50
    ):
        super().__init__(network, out_channels)
        self.decoder = decoder
        self.state_condition = state_condition
        self.predict_vel = predict_vel
        self.rollout_freq = rollout_freq
        
        logger.info(f"StatePoseModule初始化:")
        logger.info(f"  state_condition: {state_condition}")
        logger.info(f"  predict_vel: {predict_vel}")
        logger.info(f"  rollout_freq: {rollout_freq}")
    
    def _predict_pose(
        self, 
        emg: torch.Tensor, 
        initial_pos: torch.Tensor
    ) -> torch.Tensor:
        """
        预测姿态（逐时间步解码）
        
        Args:
            emg: (B, C, T) EMG输入
            initial_pos: (B, C) 初始位置
            
        Returns:
            (B, C, T') 预测结果
        """
        # 1. 编码器提取特征
        features = self.network(emg)  # (B, C', T')
        
        # 2. 计算rollout的时间长度
        # 考虑context后的有效时间
        seconds = (emg.shape[-1] - self.left_context - self.right_context) / EMG_SAMPLE_RATE
        n_time = round(seconds * self.rollout_freq)
        n_time = max(1, n_time)  # 至少1个时间步
        
        # 3. 重采样特征到rollout频率
        features = F.interpolate(
            features, 
            size=n_time, 
            mode='linear', 
            align_corners=True
        )  # (B, C', n_time)
        
        # 4. 重置解码器状态
        if hasattr(self.decoder, 'reset_state'):
            self.decoder.reset_state()
        
        # 5. 逐时间步解码
        preds = [initial_pos]  # 初始位置
        
        for t in range(n_time):
            # 当前特征
            inputs = features[:, :, t]  # (B, C')
            
            # 如果使用状态条件，拼接上一步输出
            if self.state_condition:
                inputs = torch.cat([inputs, preds[-1]], dim=-1)  # (B, C'+out_C)
            
            # 解码器预测
            pred = self.decoder(inputs, preds[-1] if self.state_condition else None)  # (B, C)
            
            # 速度/位置积分
            if self.predict_vel:
                pred = pred + preds[-1]  # 累积速度
            
            preds.append(pred)
        
        # 6. 堆叠结果（去掉initial_pos）
        preds = torch.stack(preds[1:], dim=-1)  # (B, C, n_time)
        
        return preds

