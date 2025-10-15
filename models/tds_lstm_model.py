"""
TDS-LSTM组合模型
将TDS卷积编码器和LSTM解码器组合，用于EMG到关节角度的预测

架构流程：
1. EMG信号 -> TDS编码器 -> 时序特征
2. 重采样到rollout频率（降低计算量）
3. 逐时间步LSTM解码 -> 关节角度预测
4. 重采样回原始时间长度
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List
import logging

from models.tds_network import TdsNetwork
from models.lstm_decoder import SequentialLSTM, MLPDecoder, create_decoder

logger = logging.getLogger(__name__)


class TDSLSTMModel(nn.Module):
    """
    TDS编码器 + LSTM解码器组合模型
    
    这个模型专门设计用于处理高采样率的EMG信号（如2000Hz），
    预测低频率的关节角度运动（如50Hz）
    
    Args:
        input_size: EMG通道数
        output_size: 关节角度数
        tds_stages_config: TDS各阶段的配置
        decoder_type: 解码器类型 ("sequential_lstm" 或 "mlp")
        decoder_hidden_size: 解码器隐藏层大小
        decoder_num_layers: 解码器层数
        decoder_state_condition: 是否使用状态条件
        rollout_freq: 解码频率（Hz）
        original_sampling_rate: EMG原始采样率（Hz）
        predict_velocity: 是否预测速度（False则预测位置）
        dropout: Dropout率
    """
    
    def __init__(
        self,
        input_size: int = 16,
        output_size: int = 20,
        tds_stages_config: Optional[List[Dict]] = None,
        decoder_type: str = "sequential_lstm",
        decoder_hidden_size: int = 128,
        decoder_num_layers: int = 2,
        decoder_state_condition: bool = True,
        rollout_freq: int = 50,
        original_sampling_rate: int = 2000,
        predict_velocity: bool = False,
        dropout: float = 0.1,
        **kwargs  # 额外参数
    ):
        super().__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.rollout_freq = rollout_freq
        self.original_sampling_rate = original_sampling_rate
        self.predict_velocity = predict_velocity
        self.decoder_type = decoder_type
        
        # 默认TDS配置（如果未提供）
        if tds_stages_config is None:
            tds_stages_config = [
                {
                    'in_conv_kernel': 5,
                    'in_conv_stride': 2,
                    'num_blocks': 2,
                    'channels': 8,
                    'feature_width': 4,
                    'kernel_width': 21,
                },
                {
                    'in_conv_kernel': 5,
                    'in_conv_stride': 2,
                    'num_blocks': 2,
                    'channels': 8,
                    'feature_width': 8,
                    'kernel_width': 11,
                }
            ]
        
        # 创建TDS编码器
        self.encoder = TdsNetwork(
            in_channels=input_size,
            stages_config=tds_stages_config,
            dropout=dropout
        )
        
        # 获取编码器输出通道数
        encoder_output_channels = self.encoder.get_output_channels()
        
        # 创建解码器
        self.decoder = create_decoder(
            decoder_type=decoder_type,
            in_channels=encoder_output_channels,
            out_channels=output_size,
            hidden_size=decoder_hidden_size,
            num_layers=decoder_num_layers,
            state_condition=decoder_state_condition,
            dropout=dropout
        )
        
        # 保存context信息（用于时间对齐）
        self.left_context = self.encoder.left_context
        self.right_context = self.encoder.right_context
        
        logger.info("=" * 60)
        logger.info("TDS-LSTM模型初始化完成")
        logger.info("=" * 60)
        logger.info(f"输入: {input_size} EMG通道")
        logger.info(f"输出: {output_size} 关节角度")
        logger.info(f"TDS编码器: {len(tds_stages_config)} 个stage")
        logger.info(f"  - 输出通道数: {encoder_output_channels}")
        logger.info(f"  - Left context: {self.left_context}")
        logger.info(f"解码器类型: {decoder_type}")
        if decoder_type == "sequential_lstm":
            logger.info(f"  - 隐藏层大小: {decoder_hidden_size}")
            logger.info(f"  - 层数: {decoder_num_layers}")
            logger.info(f"  - 状态条件: {decoder_state_condition}")
        logger.info(f"采样设置:")
        logger.info(f"  - 原始采样率: {original_sampling_rate} Hz")
        logger.info(f"  - Rollout频率: {rollout_freq} Hz")
        logger.info(f"  - 降采样比例: {original_sampling_rate / rollout_freq:.1f}x")
        logger.info(f"预测模式: {'速度' if predict_velocity else '位置'}")
        logger.info("=" * 60)
        
    def forward(
        self,
        x: torch.Tensor,
        initial_pose: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: EMG输入 (batch, time, channels)
            initial_pose: 初始姿态 (batch, output_size)，可选
            
        Returns:
            关节角度预测 (batch, time, output_size)
        """
        batch_size, time_len, _ = x.shape
        device = x.device
        
        # 1. 转换输入格式：(batch, time, channels) -> (batch, channels, time)
        x = x.transpose(1, 2)
        
        # 2. TDS编码器提取特征
        features = self.encoder(x)  # (batch, encoder_channels, time')
        
        # 3. 计算经过编码器后的有效时间长度
        # 考虑left_context和right_context
        encoder_time_len = features.shape[-1]
        
        # 4. 重采样特征到rollout频率
        if self.rollout_freq != self.original_sampling_rate:
            # 计算目标时间长度
            # 使用编码器输出的实际时间长度来计算
            target_time_len = max(1, int(
                encoder_time_len * self.rollout_freq / self.original_sampling_rate
            ))
            
            features = F.interpolate(
                features,
                size=target_time_len,
                mode='linear',
                align_corners=True
            )
        else:
            target_time_len = encoder_time_len
        
        # 5. 初始化预测
        self.decoder.reset_state()
        
        if initial_pose is None:
            current_pose = torch.zeros(batch_size, self.output_size, device=device)
        else:
            current_pose = initial_pose
        
        predictions = []
        
        # 6. 逐时间步解码
        for t in range(target_time_len):
            # 获取当前时间步特征
            current_features = features[:, :, t]  # (batch, encoder_channels)
            
            # LSTM/MLP解码
            pred = self.decoder(current_features, current_pose)
            
            # 更新当前姿态
            if self.predict_velocity:
                # 速度模式：累积速度
                current_pose = current_pose + pred
            else:
                # 位置模式：直接使用预测
                current_pose = pred
            
            predictions.append(current_pose)
        
        # 7. 堆叠输出
        output = torch.stack(predictions, dim=1)  # (batch, target_time, output_size)
        
        # 8. 重采样回原始时间长度
        if target_time_len != time_len:
            # 转换维度进行插值：(batch, time, output) -> (batch, output, time)
            output = output.transpose(1, 2)
            output = F.interpolate(
                output,
                size=time_len,
                mode='linear',
                align_corners=True
            )
            output = output.transpose(1, 2)  # (batch, time, output)
        
        return output
    
    def predict_step(
        self,
        x: torch.Tensor,
        initial_pose: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        推理模式的单步预测（不计算梯度）
        
        Args:
            x: EMG输入 (batch, time, channels)
            initial_pose: 初始姿态 (batch, output_size)
            
        Returns:
            关节角度预测 (batch, time, output_size)
        """
        self.eval()
        with torch.no_grad():
            return self.forward(x, initial_pose)
    
    def get_model_size(self) -> int:
        """获取模型总参数数量"""
        return sum(p.numel() for p in self.parameters())
    
    def get_trainable_params(self) -> int:
        """获取可训练参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_info(self) -> Dict[str, any]:
        """获取模型详细信息"""
        total_params = self.get_model_size()
        trainable_params = self.get_trainable_params()
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'model_size_mb': total_params * 4 / 1024 / 1024,  # FP32
            'input_size': self.input_size,
            'output_size': self.output_size,
            'encoder_type': 'TdsNetwork',
            'decoder_type': self.decoder_type,
            'rollout_freq': self.rollout_freq,
            'original_sampling_rate': self.original_sampling_rate,
            'predict_velocity': self.predict_velocity,
            'left_context': self.left_context,
            'right_context': self.right_context
        }
