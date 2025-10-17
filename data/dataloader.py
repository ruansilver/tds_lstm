"""
数据加载器创建和管理（内存优化版本）
"""

import logging
from typing import Tuple, List
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import train_test_split

from .dataset import TimeSeriesDataset, load_hdf5_files
from config import Config

logger = logging.getLogger(__name__)


def create_dataloaders(config: Config) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    创建训练、验证和测试数据加载器（内存优化版本）
    
    优化点：
    - 使用persistent_workers避免每个epoch重建workers
    - 配置prefetch_factor控制预取数据量
    - 根据配置动态调整pin_memory
    
    Args:
        config: 配置对象
        
    Returns:
        train_loader, val_loader, test_loader
    """
    # 加载所有HDF5文件
    hdf5_files = load_hdf5_files(config.data.dataset_path)
    
    if not hdf5_files:
        raise ValueError(f"在路径 {config.data.dataset_path} 中没有找到HDF5文件")
    
    # 按文件分割数据集，确保同一文件的数据不会同时出现在训练和测试集中
    train_files, temp_files = train_test_split(
        hdf5_files, 
        test_size=1-config.data.split_ratio[0], 
        random_state=42
    )
    
    val_size = config.data.split_ratio[1] / (config.data.split_ratio[1] + config.data.split_ratio[2])
    val_files, test_files = train_test_split(
        temp_files,
        test_size=1-val_size,
        random_state=42
    )
    
    logger.info(f"数据集分割:")
    logger.info(f"  训练文件数: {len(train_files)}")
    logger.info(f"  验证文件数: {len(val_files)}")
    logger.info(f"  测试文件数: {len(test_files)}")
    
    # 获取内存优化配置
    max_samples_for_normalize = getattr(config.data, 'max_samples_for_normalize', 10000)
    chunk_cache_size = getattr(config.data, 'chunk_cache_size', 64 * 1024 * 1024)
    
    # 获取新增的对齐配置
    padding = getattr(config.data, 'padding', (0, 0))
    skip_ik_failures = getattr(config.data, 'skip_ik_failures', False)
    
    # 创建数据集（使用对齐的TimeSeriesDataset）
    train_dataset = TimeSeriesDataset(
        hdf5_files=train_files,
        window_size=config.data.window_size,
        hdf5_group=config.data.hdf5_group,
        table_name=config.data.table_name,
        normalize=config.data.normalize,
        stride=config.data.stride,
        padding=padding,
        jitter=True,  # 训练时启用jitter
        skip_ik_failures=skip_ik_failures,
        max_samples_for_normalize=max_samples_for_normalize,
        chunk_cache_size=chunk_cache_size
    )
    
    val_dataset = TimeSeriesDataset(
        hdf5_files=val_files,
        window_size=config.data.window_size,
        hdf5_group=config.data.hdf5_group,
        table_name=config.data.table_name,
        normalize=config.data.normalize,
        stride=config.data.stride,
        padding=padding,
        jitter=False,  # 验证时不使用jitter
        skip_ik_failures=skip_ik_failures,
        max_samples_for_normalize=max_samples_for_normalize,
        chunk_cache_size=chunk_cache_size
    )
    
    test_dataset = TimeSeriesDataset(
        hdf5_files=test_files,
        window_size=config.data.window_size,
        hdf5_group=config.data.hdf5_group,
        table_name=config.data.table_name,
        normalize=config.data.normalize,
        stride=config.data.stride,
        padding=(0, 0),  # 测试时不使用padding
        jitter=False,  # 测试时不使用jitter
        skip_ik_failures=skip_ik_failures,
        max_samples_for_normalize=max_samples_for_normalize,
        chunk_cache_size=chunk_cache_size
    )
    
    logger.info(f"数据集样本数:")
    logger.info(f"  训练样本: {len(train_dataset)}")
    logger.info(f"  验证样本: {len(val_dataset)}")
    logger.info(f"  测试样本: {len(test_dataset)}")
    
    # 获取DataLoader优化配置
    num_workers = config.training.num_workers
    pin_memory = config.training.pin_memory
    persistent_workers = getattr(config.training, 'persistent_workers', True) if num_workers > 0 else False
    prefetch_factor = getattr(config.training, 'prefetch_factor', 2) if num_workers > 0 else None
    
    logger.info(f"DataLoader配置:")
    logger.info(f"  num_workers: {num_workers}")
    logger.info(f"  pin_memory: {pin_memory}")
    logger.info(f"  persistent_workers: {persistent_workers}")
    logger.info(f"  prefetch_factor: {prefetch_factor}")
    
    # 创建数据加载器（内存优化配置）
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=config.data.shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor
    )
    
    return train_loader, val_loader, test_loader
