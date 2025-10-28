"""
LSTM解码器模块
用于从TDS编码器的特征逐步解码出关节角度预测

支持状态条件的自回归预测
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class SequentialLSTM(nn.Module):
    """
    单步LSTM解码器
    
    特点：
    - 每次forward()只处理一个时间步
    - 维护内部隐藏状态，需要手动重置
    - 支持状态条件（将上一步输出作为输入）
    
    这种设计适合自回归预测场景，其中当前预测依赖于历史状态
    
    Args:
        in_channels: 输入特征维度
        out_channels: 输出维度（关节角度数）
        hidden_size: LSTM隐藏层大小
        num_layers: LSTM层数
        state_condition: 是否使用状态条件（将上一步输出拼接到输入）
        dropout: Dropout率（仅在多层LSTM时有效）
        output_scale: 输出缩放因子
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        state_condition: bool = True,
        dropout: float = 0.1,
        output_scale: float = 1.0
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.state_condition = state_condition
        self.output_scale = output_scale
        
        # 如果使用状态条件，LSTM输入需要包含上一步输出
        # 注意：这里的in_channels已经包含了特征+状态条件，所以不需要再加out_channels
        lstm_input_size = in_channels
        
        # LSTM核心
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # 输出层（将LSTM隐藏状态映射到关节角度）
        self.output_layer = nn.Sequential(
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_size, out_channels)
        )
        
        # 内部隐藏状态
        self.hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        
        logger.info(f"SequentialLSTM初始化:")
        logger.info(f"  输入维度: {lstm_input_size} (特征:{in_channels}, 状态条件:{out_channels if state_condition else 0})")
        logger.info(f"  隐藏层大小: {hidden_size}")
        logger.info(f"  层数: {num_layers}")
        logger.info(f"  输出维度: {out_channels}")
        
    def reset_state(self):
        """重置LSTM隐藏状态"""
        self.hidden_state = None
        
    def forward(self, x: torch.Tensor, prev_output: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        单时间步前向传播
        
        Args:
            x: 当前时间步的特征 (batch, in_channels)
            prev_output: 上一时间步的输出 (batch, out_channels)，仅在state_condition=True时使用
            
        Returns:
            当前时间步的预测 (batch, out_channels)
        """
        batch_size = x.size(0)
        device = x.device
        
        # 初始化隐藏状态（如果需要）
        if self.hidden_state is None:
            h_0 = torch.zeros(
                self.num_layers, batch_size, self.hidden_size,
                device=device, dtype=x.dtype
            )
            c_0 = torch.zeros(
                self.num_layers, batch_size, self.hidden_size,
                device=device, dtype=x.dtype
            )
            self.hidden_state = (h_0, c_0)
        
        # 准备LSTM输入
        if self.state_condition and prev_output is not None:
            # 拼接当前特征和上一步输出
            lstm_input = torch.cat([x, prev_output], dim=-1)
        else:
            lstm_input = x
        
        # 添加时间维度: (batch, 1, features)
        lstm_input = lstm_input.unsqueeze(1)
        
        # LSTM前向传播
        lstm_out, self.hidden_state = self.lstm(lstm_input, self.hidden_state)
        
        # 移除时间维度: (batch, hidden_size)
        lstm_out = lstm_out.squeeze(1)
        
        # 输出层
        output = self.output_layer(lstm_out)
        
        # 缩放输出
        if self.output_scale != 1.0:
            output = output * self.output_scale
        
        return output
    
    def batch_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        批量前向传播（用于非自回归场景）
        
        Args:
            x: 输入特征 (batch, time, in_channels)
            
        Returns:
            预测结果 (batch, time, out_channels)
        """
        # 注意：这个方法不使用状态条件
        lstm_out, _ = self.lstm(x)
        output = self.output_layer(lstm_out)
        
        if self.output_scale != 1.0:
            output = output * self.output_scale
            
        return output


class MLPDecoder(nn.Module):
    """
    简单的MLP解码器（作为对比baseline）
    
    不维护时间状态，每个时间步独立预测
    
    Args:
        in_channels: 输入特征维度
        out_channels: 输出维度
        hidden_sizes: 隐藏层大小列表
        dropout: Dropout率
        use_layer_norm: 是否使用LayerNorm
        output_scale: 输出缩放因子
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_sizes: list = [256, 128],
        dropout: float = 0.1,
        use_layer_norm: bool = True,
        output_scale: float = 1.0
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.output_scale = output_scale
        
        # 构建MLP层
        layers = []
        prev_size = in_channels
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_size))
            
            layers.append(nn.LeakyReLU(0.1))
            layers.append(nn.Dropout(dropout))
            
            prev_size = hidden_size
        
        # 输出层
        layers.append(nn.Linear(prev_size, out_channels))
        
        self.mlp = nn.Sequential(*layers)
        
        logger.info(f"MLPDecoder初始化:")
        logger.info(f"  输入维度: {in_channels}")
        logger.info(f"  隐藏层: {hidden_sizes}")
        logger.info(f"  输出维度: {out_channels}")
        
    def reset_state(self):
        """MLP无状态，此方法为了接口兼容"""
        pass
        
    def forward(self, x: torch.Tensor, prev_output: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入特征 (batch, in_channels)
            prev_output: 忽略（保持接口一致）
            
        Returns:
            预测结果 (batch, out_channels)
        """
        output = self.mlp(x)
        
        if self.output_scale != 1.0:
            output = output * self.output_scale
            
        return output
    
    def batch_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        批量前向传播
        
        Args:
            x: 输入特征 (batch, time, in_channels)
            
        Returns:
            预测结果 (batch, time, out_channels)
        """
        output = self.mlp(x)
        
        if self.output_scale != 1.0:
            output = output * self.output_scale
            
        return output


def create_decoder(
    decoder_type: str,
    in_channels: int,
    out_channels: int,
    **kwargs
) -> nn.Module:
    """
    工厂函数：创建解码器
    
    Args:
        decoder_type: 解码器类型 ("sequential_lstm" 或 "mlp")
        in_channels: 输入通道数
        out_channels: 输出通道数
        **kwargs: 其他参数
        
    Returns:
        解码器模块
    """
    if decoder_type.lower() == "sequential_lstm":
        return SequentialLSTM(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_size=kwargs.get('hidden_size', 128),
            num_layers=kwargs.get('num_layers', 2),
            state_condition=kwargs.get('state_condition', True),
            dropout=kwargs.get('dropout', 0.1),
            output_scale=kwargs.get('output_scale', 1.0)
        )
    elif decoder_type.lower() == "mlp":
        return MLPDecoder(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_sizes=kwargs.get('hidden_sizes', [256, 128]),
            dropout=kwargs.get('dropout', 0.1),
            use_layer_norm=kwargs.get('use_layer_norm', True),
            output_scale=kwargs.get('output_scale', 1.0)
        )
    else:
        raise ValueError(f"未知的解码器类型: {decoder_type}")

