"""
数据加载器创建和管理
"""

import logging
from typing import Tuple, List
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import train_test_split

from .dataset import EMG2PoseDataset, load_hdf5_files
from config import Config

logger = logging.getLogger(__name__)

def create_dataloaders(config: Config) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    创建训练、验证和测试数据加载器
    
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
    #这里没有用random_split，因为它只能分成两部分      而且random_split是按样本数分割的，不是按文件分割的
    # 我们需要分成三部分：训练集、验证集、测试集 
    train_files, temp_files = train_test_split(
        hdf5_files, 
        test_size=1-config.data.split_ratio[0], 
        random_state=42     #设置随机种子，保证每次分割结果一致
    )
    
    val_size = config.data.split_ratio[1] / (config.data.split_ratio[1] + config.data.split_ratio[2])
    val_files, test_files = train_test_split(
        temp_files,    # 剩余的30%数据
        test_size=1-val_size,   # 0.5
        random_state=42  # 保持和上次分割一致的随机种子
    )
    
    logger.info(f"数据集分割:")
    logger.info(f"  训练文件数: {len(train_files)}")
    logger.info(f"  验证文件数: {len(val_files)}")
    logger.info(f"  测试文件数: {len(test_files)}")
    
    # 创建数据集
    train_dataset = EMG2PoseDataset(
        hdf5_files=train_files,
        window_size=config.data.window_size,
        hdf5_group=config.data.hdf5_group,
        table_name=config.data.table_name,
        normalize=config.data.normalize,
        stride=config.data.stride
    )
    
    val_dataset = EMG2PoseDataset(
        hdf5_files=val_files,
        window_size=config.data.window_size,
        hdf5_group=config.data.hdf5_group,
        table_name=config.data.table_name,
        normalize=config.data.normalize,
        stride=config.data.stride
    )
    
    test_dataset = EMG2PoseDataset(
        hdf5_files=test_files,
        window_size=config.data.window_size,
        hdf5_group=config.data.hdf5_group,
        table_name=config.data.table_name,
        normalize=config.data.normalize,
        stride=config.data.stride
    )
    
    logger.info(f"数据集样本数:")
    logger.info(f"  训练样本: {len(train_dataset)}")
    logger.info(f"  验证样本: {len(val_dataset)}")
    logger.info(f"  测试样本: {len(test_dataset)}")
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=config.data.shuffle,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory,
        drop_last=False
    )
    
    return train_loader, val_loader, test_loader
