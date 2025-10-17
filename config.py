"""
统一配置加载器
基于YAML的配置系统，支持配置验证和自动设备检测
"""

import os
import yaml
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Optional

logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    """数据配置"""
    dataset_path: str = "D:/Dataset/emg2pose_dataset_mini"
    hdf5_group: str = "emg2pose"
    table_name: str = "timeseries"
    stride: int = 1
    emg_channels: int = 16
    joint_angles_channels: int = 20
    window_size: int = 500
    split_ratio: List[float] = field(default_factory=lambda: [0.7, 0.15, 0.15])
    normalize: bool = False
    shuffle: bool = True
    
    # 内存优化配置
    max_samples_for_normalize: int = 10000  # 标准化时的最大采样数
    chunk_cache_size: int = 64 * 1024 * 1024  # HDF5 chunk cache大小（字节）64MB


@dataclass
class ModelConfig:
    """模型配置"""
    input_size: int = 16
    output_size: int = 20
    dropout: float = 0.1
    
    # 传统LSTM/GRU参数（备用）
    hidden_size: int = 128
    num_layers: int = 2
    bidirectional: bool = True
    
    # TDS-LSTM专用参数
    # TDS编码器配置
    tds_stages: Optional[List[Dict]] = None  # TDS各阶段配置
    
    # LSTM解码器配置
    decoder_type: str = "sequential_lstm"  # "sequential_lstm" 或 "mlp"
    decoder_hidden_size: int = 128
    decoder_num_layers: int = 2
    decoder_state_condition: bool = True
    
    # 预测设置
    rollout_freq: int = 50
    original_sampling_rate: int = 2000  # EMG原始采样率(Hz)
    predict_velocity: bool = False
    
    # 旧参数（向后兼容）
    encoder_features: int = 64
    decoder_hidden: int = 64
    decoder_layers: int = 2


@dataclass
class TrainingConfig:
    """训练配置"""
    num_epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 0.0005
    weight_decay: float = 0.01
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    step_size: int = 20
    gamma: float = 0.5
    
    # 损失函数配置
    loss_type: str = "mse+mae"  # "mse", "mae", "mse+mae"
    mae_weight: float = 0.5  # MAE损失权重（当loss_type="mse+mae"时）
    
    # 学习率调度器配置
    warmup_epochs: int = 5  # warmup轮数
    
    # 梯度相关
    gradient_clip_norm: float = 1.0
    gradient_accumulation_steps: int = 1
    
    # 早停
    early_stopping: bool = True
    patience: int = 20
    min_delta: float = 1e-4
    monitor_metric: str = "val_loss"  # 监控指标：val_loss, val_mse, val_mae等
    restore_best_weights: bool = True  # 是否恢复最佳权重
    
    # 设备配置
    device: str = "auto"
    num_workers: int = 4
    pin_memory: bool = True
    
    # 内存优化配置
    persistent_workers: bool = True  # DataLoader worker持久化（PyTorch 1.7+）
    prefetch_factor: int = 2  # 每个worker预取的batch数
    compute_metrics_online: bool = True  # 在线计算metrics，避免大数组拼接


@dataclass
class LoggingConfig:
    """日志配置"""
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    save_every: int = 5
    log_every: int = 10
    save_best_only: bool = True
    monitor: str = "val_loss"
    mode: str = "min"


@dataclass
class Config:
    """统一配置类"""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    def __post_init__(self):
        """初始化后处理"""
        self._setup_directories()
        self._setup_device()
        self._validate_config()
    
    def _setup_directories(self):
        """创建必要目录"""
        Path(self.logging.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.logging.log_dir).mkdir(parents=True, exist_ok=True)
        logger.info(f"确保目录存在: {self.logging.checkpoint_dir}, {self.logging.log_dir}")
    
    def _setup_device(self):
        """自动设置设备"""
        if self.training.device == "auto":
            try:
                import torch
                self.training.device = 'cuda' if torch.cuda.is_available() else 'cpu'
                logger.info(f"自动选择设备: {self.training.device}")
            except ImportError:
                self.training.device = 'cpu'
                logger.warning("PyTorch未安装，使用CPU")
    
    def _validate_config(self):
        """验证配置参数"""
        # 验证数据分割比例
        if abs(sum(self.data.split_ratio) - 1.0) > 1e-6:
            raise ValueError(f"数据分割比例之和必须为1.0，当前为: {sum(self.data.split_ratio)}")
        
        # 验证通道数一致性
        if self.model.input_size != self.data.emg_channels:
            logger.warning(f"模型输入维度 ({self.model.input_size}) 与EMG通道数 ({self.data.emg_channels}) 不一致")
        
        if self.model.output_size != self.data.joint_angles_channels:
            logger.warning(f"模型输出维度 ({self.model.output_size}) 与关节角度通道数 ({self.data.joint_angles_channels}) 不一致")
        
        # 验证优化器和调度器
        valid_optimizers = ["adam", "adamw", "sgd"]
        if self.training.optimizer.lower() not in valid_optimizers:
            raise ValueError(f"不支持的优化器: {self.training.optimizer}，支持: {valid_optimizers}")
        
        valid_schedulers = ["step", "cosine", "plateau", "none"]
        if self.training.scheduler.lower() not in valid_schedulers:
            raise ValueError(f"不支持的调度器: {self.training.scheduler}，支持: {valid_schedulers}")
        
        logger.info("配置验证通过")
    
    def save_config(self, filepath: str):
        """保存配置到YAML文件"""
        config_dict = {
            'data': {
                'dataset_path': self.data.dataset_path,
                'hdf5_group': self.data.hdf5_group,
                'table_name': self.data.table_name,
                'stride': self.data.stride,
                'emg_channels': self.data.emg_channels,
                'joint_angles_channels': self.data.joint_angles_channels,
                'window_size': self.data.window_size,
                'split_ratio': self.data.split_ratio,
                'normalize': self.data.normalize,
                'shuffle': self.data.shuffle,
                'max_samples_for_normalize': self.data.max_samples_for_normalize,
                'chunk_cache_size': self.data.chunk_cache_size,
            },
            'model': {
                'input_size': self.model.input_size,
                'output_size': self.model.output_size,
                'dropout': self.model.dropout,
                'hidden_size': self.model.hidden_size,
                'num_layers': self.model.num_layers,
                'bidirectional': self.model.bidirectional,
                'tds_stages': self.model.tds_stages,
                'decoder_type': self.model.decoder_type,
                'decoder_hidden_size': self.model.decoder_hidden_size,
                'decoder_num_layers': self.model.decoder_num_layers,
                'decoder_state_condition': self.model.decoder_state_condition,
                'rollout_freq': self.model.rollout_freq,
                'original_sampling_rate': self.model.original_sampling_rate,
                'predict_velocity': self.model.predict_velocity,
                'encoder_features': self.model.encoder_features,
                'decoder_hidden': self.model.decoder_hidden,
                'decoder_layers': self.model.decoder_layers,
            },
            'training': {
                'num_epochs': self.training.num_epochs,
                'batch_size': self.training.batch_size,
                'learning_rate': self.training.learning_rate,
                'weight_decay': self.training.weight_decay,
                'optimizer': self.training.optimizer,
                'scheduler': self.training.scheduler,
                'step_size': self.training.step_size,
                'gamma': self.training.gamma,
                'loss_type': self.training.loss_type,
                'mae_weight': self.training.mae_weight,
                'warmup_epochs': self.training.warmup_epochs,
                'gradient_clip_norm': self.training.gradient_clip_norm,
                'gradient_accumulation_steps': self.training.gradient_accumulation_steps,
                'early_stopping': self.training.early_stopping,
                'patience': self.training.patience,
                'min_delta': self.training.min_delta,
                'monitor_metric': getattr(self.training, 'monitor_metric', 'val_loss'),
                'restore_best_weights': getattr(self.training, 'restore_best_weights', True),
                'device': self.training.device,
                'num_workers': self.training.num_workers,
                'pin_memory': self.training.pin_memory,
                'persistent_workers': self.training.persistent_workers,
                'prefetch_factor': self.training.prefetch_factor,
                'compute_metrics_online': self.training.compute_metrics_online,
            },
            'logging': {
                'checkpoint_dir': self.logging.checkpoint_dir,
                'log_dir': self.logging.log_dir,
                'save_every': self.logging.save_every,
                'log_every': self.logging.log_every,
                'save_best_only': self.logging.save_best_only,
                'monitor': self.logging.monitor,
                'mode': self.logging.mode,
            }
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
        
        logger.info(f"配置已保存到: {filepath}")


def load_config_from_yaml(yaml_path: str) -> Config:
    """
    从YAML文件加载配置
    
    Args:
        yaml_path: YAML配置文件路径
        
    Returns:
        Config对象
    """
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"配置文件不存在: {yaml_path}")
    
    logger.info(f"从YAML文件加载配置: {yaml_path}")
    
    with open(yaml_path, 'r', encoding='utf-8') as f:
        yaml_config = yaml.safe_load(f)
    
    # 创建配置对象
    config = Config()
    
    # 加载数据配置
    if 'data' in yaml_config:
        data_config = yaml_config['data']
        for key, value in data_config.items():
            if hasattr(config.data, key):
                setattr(config.data, key, value)
    
    # 加载模型配置
    if 'model' in yaml_config:
        model_config = yaml_config['model']
        for key, value in model_config.items():
            if hasattr(config.model, key):
                setattr(config.model, key, value)
    
    # 加载训练配置
    if 'training' in yaml_config:
        training_config = yaml_config['training']
        for key, value in training_config.items():
            if hasattr(config.training, key):
                setattr(config.training, key, value)
    
    # 加载日志配置
    if 'logging' in yaml_config:
        logging_config = yaml_config['logging']
        for key, value in logging_config.items():
            if hasattr(config.logging, key):
                setattr(config.logging, key, value)
    
    # 重新执行后处理以验证配置
    config.__post_init__()
    
    return config


def get_available_configs() -> Dict[str, str]:
    """
    获取可用的配置文件
    
    Returns:
        配置文件字典 {名称: 路径}
    """
    config_dir = Path("configs")
    if not config_dir.exists():
        return {}
    
    configs = {}
    for yaml_file in config_dir.glob("*.yaml"):
        config_name = yaml_file.stem
        configs[config_name] = str(yaml_file)
    
    return configs


def create_default_config() -> Config:
    """创建默认配置"""
    logger.info("创建默认配置")
    return Config()


# 配置文件映射（为了向后兼容）
CONFIG_FILES = {
    'quick_demo': 'configs/quick_demo.yaml',
    'default': 'configs/default.yaml',
    'high_performance': 'configs/high_performance.yaml',
    'emg2pose_mimic': 'configs/emg2pose_mimic.yaml'
}


def get_config_by_name(config_name: str) -> Config:
    """
    根据配置名称获取配置
    
    Args:
        config_name: 配置名称
        
    Returns:
        Config对象
    """
    if config_name in CONFIG_FILES:
        config_path = CONFIG_FILES[config_name]
        if os.path.exists(config_path):
            return load_config_from_yaml(config_path)
        else:
            logger.warning(f"配置文件不存在: {config_path}，使用默认配置")
            return create_default_config()
    else:
        logger.warning(f"未知配置名称: {config_name}，使用默认配置")
        return create_default_config()



