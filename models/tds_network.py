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
    TDS阶段模块
    
    包含：
    1. 初始Conv1d块（可选，用于降采样和通道变换）
    2. TDS编码器（多个TDS块）
    3. 线性投影（可选，用于输出通道变换）
    
    Args:
        in_channels: 输入通道数
        in_conv_kernel: 初始卷积核大小
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
        in_channels: int,
        in_conv_kernel: int,
        in_conv_stride: int,
        num_blocks: int,
        channels: int,
        feature_width: int,
        kernel_width: int,
        out_channels: Optional[int] = None,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.in_conv_kernel = in_conv_kernel
        self.in_conv_stride = in_conv_stride
        self.out_channels = out_channels
        
        num_features = channels * feature_width
        
        # 初始卷积块
        if in_conv_kernel > 0:
            self.conv1d_block = Conv1dBlock(
                in_channels,
                num_features,
                kernel_size=in_conv_kernel,
                stride=in_conv_stride,
                dropout=dropout
            )
        else:
            # 如果不需要初始卷积，确保输入通道数匹配
            if in_channels != num_features:
                raise ValueError(
                    f"当in_conv_kernel=0时，in_channels ({in_channels}) "
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
    完整的TDS网络
    
    由多个TDS阶段组成，每个阶段可以进行降采样和特征变换
    
    Args:
        in_channels: 输入通道数（EMG通道数）
        stages_config: 各阶段的配置列表
    """
    
    def __init__(
        self,
        in_channels: int,
        stages_config: List[Dict],
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.stages_config = stages_config
        
        # 构建各个stage
        stages = []
        current_channels = in_channels
        
        for i, stage_cfg in enumerate(stages_config):
            stage = TdsStage(
                in_channels=current_channels,
                in_conv_kernel=stage_cfg.get('in_conv_kernel', 5),
                in_conv_stride=stage_cfg.get('in_conv_stride', 2),
                num_blocks=stage_cfg.get('num_blocks', 2),
                channels=stage_cfg.get('channels', 8),
                feature_width=stage_cfg.get('feature_width', 4),
                kernel_width=stage_cfg.get('kernel_width', 21),
                out_channels=stage_cfg.get('out_channels', None),
                dropout=dropout
            )
            stages.append(stage)
            
            # 更新下一个stage的输入通道数
            if stage.output_projection is not None:
                current_channels = stage.out_channels
            else:
                current_channels = stage_cfg['channels'] * stage_cfg['feature_width']
        
        self.stages = nn.Sequential(*stages)
        self.output_channels = current_channels
        
        # 计算left_context（用于时间对齐）
        self.left_context = self._calculate_left_context()
        self.right_context = 0
        
        logger.info(f"TdsNetwork初始化完成:")
        logger.info(f"  输入通道数: {in_channels}")
        logger.info(f"  输出通道数: {self.output_channels}")
        logger.info(f"  Stage数量: {len(stages_config)}")
        logger.info(f"  Left context: {self.left_context}")
        
    def _calculate_left_context(self) -> int:
        """
        计算网络的left context（感受野）
        
        这对于时间对齐很重要
        """
        left_context = 0
        stride_product = 1
        
        for stage_cfg in self.stages_config:
            # Conv1d块的贡献
            kernel_size = stage_cfg.get('in_conv_kernel', 5)
            stride = stage_cfg.get('in_conv_stride', 2)
            
            if kernel_size > 0:
                left_context += (kernel_size - 1) * stride_product
                stride_product *= stride
            
            # TDS块的贡献
            num_blocks = stage_cfg.get('num_blocks', 2)
            kernel_width = stage_cfg.get('kernel_width', 21)
            
            # 每个TDS块包含1个Conv2d和1个FC，只有Conv2d贡献context
            left_context += num_blocks * (kernel_width - 1) * stride_product
        
        return left_context
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        输入: (batch, in_channels, time)
        输出: (batch, output_channels, time')
        """
        return self.stages(x)
    
    def get_output_channels(self) -> int:
        """获取输出通道数"""
        return self.output_channels

