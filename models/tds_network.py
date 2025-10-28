"""
标准时间-深度可分离(TDS)卷积网络
Time-Depth Separable Convolutions for efficient temporal feature extraction

参考论文: "Sequence-to-Sequence Speech Recognition with Time-Depth Separable Convolutions"
架构设计基于公开的TDS卷积原理，重新实现以适配EMG信号处理
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class Conv1dBlock(nn.Module):
    """
    标准1D卷积块
    包含: Conv1d -> LayerNorm -> ReLU -> Dropout
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.kernel_size = kernel_size
        self.stride = stride
        
        # 使用padding=0，后续在forward中处理padding
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            bias=False
        )
        
        self.layer_norm = nn.LayerNorm(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入: (batch, channels, time)
        输出: (batch, out_channels, time')
        """
        x = self.conv(x)  # (B, C, T')
        
        # LayerNorm需要在channel维度上操作
        x = x.transpose(1, 2)  # (B, T', C)
        x = self.layer_norm(x)
        x = x.transpose(1, 2)  # (B, C, T')
        
        x = self.relu(x)
        x = self.dropout(x)
        
        return x


class TDSConv2dBlock(nn.Module):
    """
    时间-深度可分离2D卷积块
    
    核心思想：将特征分组，在时间维度上进行分组卷积
    这样可以减少参数量同时保持时间建模能力
    
    Args:
        channels: 通道数（等于 num_groups * group_width）
        group_width: 每组的宽度
        kernel_width: 时间卷积核宽度（必须是奇数）
    """
    
    def __init__(
        self,
        channels: int,
        group_width: int,
        kernel_width: int
    ):
        super().__init__()
        
        assert kernel_width % 2 == 1, "kernel_width必须是奇数"
        assert channels % group_width == 0, "channels必须能被group_width整除"
        
        self.channels = channels
        self.num_groups = channels // group_width
        self.group_width = group_width
        
        # 确保kernel_width不超过group_width，避免维度问题
        self.kernel_width = min(kernel_width, group_width)
        if self.kernel_width != kernel_width:
            logger.warning(
                f"TDSConv2dBlock: kernel_width ({kernel_width}) > group_width ({group_width}), "
                f"调整为 {self.kernel_width}"
            )
        
        # 确保kernel_width是奇数
        if self.kernel_width % 2 == 0:
            self.kernel_width -= 1
            logger.warning(f"TDSConv2dBlock: kernel_width调整为奇数 {self.kernel_width}")
        
        # 2D卷积：在时间维度上操作
        self.conv2d = nn.Conv2d(
            in_channels=self.num_groups,
            out_channels=self.num_groups,
            kernel_size=(1, self.kernel_width),
            stride=(1, 1),
            padding=(0, 0),  # 手动处理padding
            bias=True
        )
        
        self.relu = nn.ReLU(inplace=True)
        self.layer_norm = nn.LayerNorm(channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入: (batch, channels, time)
        输出: (batch, channels, time')
        """
        B, C, T = x.shape
        
        # 重塑为分组格式: (B, num_groups, group_width, T)
        x_reshaped = x.view(B, self.num_groups, self.group_width, T)
        
        # 2D卷积
        conv_out = self.conv2d(x_reshaped)  # (B, num_groups, group_width, T')
        conv_out = self.relu(conv_out)
        
        # 恢复形状
        T_out = conv_out.shape[-1]
        conv_out = conv_out.view(B, C, T_out)
        
        # 残差连接（对齐输出时间长度）
        residual = x[..., -T_out:]
        x_out = conv_out + residual
        
        # LayerNorm
        x_out = x_out.transpose(1, 2)  # (B, T', C)
        x_out = self.layer_norm(x_out)
        x_out = x_out.transpose(1, 2)  # (B, C, T')
        
        return x_out


class TDSFullyConnectedBlock(nn.Module):
    """
    全连接残差块
    
    在每个时间步上应用全连接层，带残差连接
    """
    
    def __init__(self, num_features: int):
        super().__init__()
        
        self.fc1 = nn.Linear(num_features, num_features)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(num_features, num_features)
        self.layer_norm = nn.LayerNorm(num_features)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入: (batch, channels, time)
        输出: (batch, channels, time)
        """
        # 转换维度以应用线性层
        x_t = x.transpose(1, 2)  # (B, T, C)
        
        # 全连接块
        fc_out = self.fc1(x_t)
        fc_out = self.relu(fc_out)
        fc_out = self.fc2(fc_out)
        
        # 残差连接
        x_t = x_t + fc_out
        
        # LayerNorm
        x_t = self.layer_norm(x_t)
        
        # 转换回原始维度
        return x_t.transpose(1, 2)  # (B, C, T)


class TDSConvEncoder(nn.Module):
    """
    TDS卷积编码器
    
    由多个TDS块组成，每个块包含：
    - TDSConv2dBlock（时间-深度可分离卷积）
    - TDSFullyConnectedBlock（全连接残差块）
    
    Args:
        num_features: 特征维度
        num_blocks: TDS块的数量
        num_groups: 分组数（num_features必须能被整除）
        kernel_width: 卷积核宽度
    """
    
    def __init__(
        self,
        num_features: int,
        num_blocks: int = 2,
        num_groups: int = 8,
        kernel_width: int = 21
    ):
        super().__init__()
        
        assert num_features % num_groups == 0, \
            f"num_features ({num_features}) 必须能被 num_groups ({num_groups}) 整除"
        
        self.num_blocks = num_blocks
        self.kernel_width = kernel_width
        
        group_width = num_features // num_groups
        
        # 构建TDS块序列
        blocks = []
        for _ in range(num_blocks):
            blocks.extend([
                TDSConv2dBlock(num_features, group_width, kernel_width),
                TDSFullyConnectedBlock(num_features)
            ])
        
        self.tds_blocks = nn.Sequential(*blocks)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入: (batch, num_features, time)
        输出: (batch, num_features, time')
        """
        return self.tds_blocks(x)


class TdsStage(nn.Module):
    """
    TDS阶段模块（对齐参考实现）
    
    包含：
    1. 初始Conv1d块（可选，用于降采样和通道变换）
    2. TDS编码器（多个TDS块）
    3. 线性投影（可选，用于输出通道变换）
    
    Args:
        in_channels: 输入通道数
        in_conv_kernel_width: 初始卷积核大小（对应参考实现）
        in_conv_stride: 初始卷积步长
        num_blocks: TDS块数量
        channels: TDS内部通道数
        feature_width: 特征宽度（与channels相乘得到总特征数）
        kernel_width: TDS卷积核宽度
        out_channels: 输出通道数（可选）
        dropout: Dropout率
    """
    
    def __init__(
        self,
        in_channels: int = 16,
        in_conv_kernel_width: int = 5,
        in_conv_stride: int = 1,
        num_blocks: int = 1,
        channels: int = 8,
        feature_width: int = 2,
        kernel_width: int = 1,
        out_channels: Optional[int] = None,
        dropout: float = 0.1,
        # 兼容旧参数名
        in_conv_kernel: Optional[int] = None
    ):
        super().__init__()
        
        # 兼容参数名
        if in_conv_kernel is not None:
            in_conv_kernel_width = in_conv_kernel
            
        self.in_conv_kernel_width = in_conv_kernel_width
        self.in_conv_stride = in_conv_stride
        self.out_channels = out_channels
        self.channels = channels
        self.feature_width = feature_width
        
        num_features = channels * feature_width
        
        # 初始卷积块
        if in_conv_kernel_width > 0:
            self.conv1d_block = Conv1dBlock(
                in_channels,
                num_features,
                kernel_size=in_conv_kernel_width,
                stride=in_conv_stride,
                dropout=dropout
            )
        else:
            # 如果不需要初始卷积，确保输入通道数匹配
            if in_channels != num_features:
                raise ValueError(
                    f"当in_conv_kernel_width=0时，in_channels ({in_channels}) "
                    f"必须等于 channels * feature_width ({num_features})"
                )
            self.conv1d_block = None
        
        # TDS编码器
        self.tds_encoder = TDSConvEncoder(
            num_features=num_features,
            num_blocks=num_blocks,
            num_groups=channels,
            kernel_width=kernel_width
        )
        
        # 可选的输出投影
        if out_channels is not None:
            self.output_projection = nn.Linear(num_features, out_channels)
        else:
            self.output_projection = None
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入: (batch, in_channels, time)
        输出: (batch, out_channels or num_features, time')
        """
        # 初始卷积
        if self.conv1d_block is not None:
            x = self.conv1d_block(x)
        
        # TDS编码
        x = self.tds_encoder(x)
        
        # 输出投影
        if self.output_projection is not None:
            x = x.transpose(1, 2)  # (B, T, C)
            x = self.output_projection(x)
            x = x.transpose(1, 2)  # (B, C', T)
        
        return x


class TdsNetwork(nn.Module):
    """
    完整的TDS网络（对齐参考实现）
    
    由多个Conv1d块和TDS阶段组成，每个阶段可以进行降采样和特征变换
    
    Args:
        conv_blocks: Conv1d块序列
        tds_stages: TDS阶段序列
    """
    
    def __init__(self, conv_blocks, tds_stages):
        super().__init__()
        self.layers = nn.Sequential(*conv_blocks, *tds_stages)
        self.left_context = self._get_left_context(conv_blocks, tds_stages)
        self.right_context = 0
        
        # 计算输出通道数
        if tds_stages and hasattr(tds_stages[-1], 'out_channels') and tds_stages[-1].out_channels:
            self.output_channels = tds_stages[-1].out_channels
        else:
            # 从最后一个stage计算
            last_stage = tds_stages[-1] if tds_stages else None
            if last_stage and hasattr(last_stage, 'channels') and hasattr(last_stage, 'feature_width'):
                self.output_channels = last_stage.channels * last_stage.feature_width
            else:
                self.output_channels = 64  # 默认值
        
        logger.info(f"TdsNetwork初始化完成:")
        logger.info(f"  Conv1d块数量: {len(conv_blocks)}")
        logger.info(f"  TDS阶段数量: {len(tds_stages)}")
        logger.info(f"  输出通道数: {self.output_channels}")
        logger.info(f"  Left context: {self.left_context}")

    def forward(self, x):
        return self.layers(x)

    def _get_left_context(self, conv_blocks, tds_stages) -> int:
        """计算left context（对齐参考实现）"""
        left, stride = 0, 1

        for conv_block in conv_blocks:
            left += (conv_block.kernel_size - 1) * stride
            stride *= conv_block.stride

        for tds_stage in tds_stages:
            # 从TDS stage中获取相关信息
            if hasattr(tds_stage, 'conv1d_block') and tds_stage.conv1d_block:
                conv_block = tds_stage.conv1d_block
                left += (conv_block.kernel_size - 1) * stride
                stride *= conv_block.stride

            if hasattr(tds_stage, 'tds_encoder'):
                tds_encoder = tds_stage.tds_encoder
                if hasattr(tds_encoder, 'kernel_width') and hasattr(tds_encoder, 'num_blocks'):
                    for _ in range(tds_encoder.num_blocks):
                        left += (tds_encoder.kernel_width - 1) * stride

        return left
    
    def get_output_channels(self) -> int:
        """获取输出通道数"""
        return self.output_channels

