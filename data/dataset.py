"""
EMG到姿态预测的数据集类
处理HDF5文件中的时序数据
"""

import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Dict, Any
import pandas as pd
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)

class EMG2PoseDataset(Dataset):
    """
    EMG到姿态预测的数据集类
    
    Args:
        hdf5_files: HDF5文件路径列表
        window_size: 滑动窗口大小
        hdf5_group: HDF5文件中的组名
        table_name: 表名
        normalize: 是否标准化数据
        stride: 滑动窗口步长，默认为1
    """
    
    def __init__(
        self,   #接受配置文件的参数，在dataloader.py中传入
        hdf5_files: List[str],   # HDF5文件路径列表
        window_size: int = 11750,
        hdf5_group: str = "emg2pose",
        table_name: str = "timeseries",
        normalize: bool = False,
        stride: int = 1
    ):
        self.hdf5_files = hdf5_files
        self.window_size = window_size
        self.hdf5_group = hdf5_group
        self.table_name = table_name
        self.normalize = normalize
        self.stride = stride
        
        # 存储数据和索引
        self.data_indices = []  # (file_idx, start_idx) 的列表
        self.emg_scalers = {}   # 每个文件的EMG标准化器
        self.angle_scalers = {} # 每个文件的角度标准化器
        
        # 加载和预处理数据
        self._load_data()
        
    def _load_data(self):   #这种是类的私有方法，外部不应该调用
        """加载所有HDF5文件并创建索引"""
        
        logger.info(f"开始加载 {len(self.hdf5_files)} 个HDF5文件...")
        
        for file_idx, hdf5_file in enumerate(self.hdf5_files):   # 遍历每个hdf5文件
            if not os.path.exists(hdf5_file):
                logger.warning(f"文件不存在: {hdf5_file}")
                continue
                
            try:
                with h5py.File(hdf5_file, 'r') as f:
                    # 检查数据结构
                    if self.hdf5_group not in f:
                        logger.warning(f"文件 {hdf5_file} 中没有找到组 '{self.hdf5_group}'")
                        continue
                        
                    group = f[self.hdf5_group]
                    if self.table_name not in group:
                        logger.warning(f"文件 {hdf5_file} 中没有找到表 '{self.table_name}'")
                        continue
                    
                    # 读取数据
                    table = group[self.table_name]  # 这里的table是一个h5py.Dataset对象
                    data_length = len(table)
                    
                    # 检查数据长度是否足够
                    if data_length < self.window_size:
                        logger.warning(f"文件 {hdf5_file} 数据长度 {data_length} 小于窗口大小 {self.window_size}")
                        continue
                    
                    # 创建滑动窗口索引
                    num_windows = (data_length - self.window_size) // self.stride + 1
                    for i in range(num_windows):
                        start_idx = i * self.stride
                        self.data_indices.append((file_idx, start_idx))
                    
                    # 如果需要标准化，读取全部数据进行拟合
                    if self.normalize:
                        self._fit_scalers(file_idx, table)
                        
                    logger.info(f"文件 {hdf5_file} 加载完成，数据长度: {data_length}, 创建窗口数: {num_windows}")
                    
            except Exception as e:
                logger.error(f"加载文件 {hdf5_file} 时出错: {str(e)}")
                continue
        
        logger.info(f"数据加载完成，总共 {len(self.data_indices)} 个样本")
        
    def _fit_scalers(self, file_idx: int, table: Any):
        """为指定文件拟合标准化器"""
        try:
            # 检查数据结构
            if hasattr(table.dtype, 'names') and table.dtype.names:
                # 结构化数组，根据实际数据格式处理
                field_names = table.dtype.names
                
                # 提取EMG数据
                if 'emg' in field_names:
                    emg_data = table['emg'][:]  # 形状应该是 (n_samples, 16)
                    if len(emg_data.shape) == 2 and emg_data.shape[1] == 16:
                        emg_scaler = StandardScaler()
                        emg_scaler.fit(emg_data)
                        self.emg_scalers[file_idx] = emg_scaler
                
                # 提取关节角度数据
                if 'joint_angles' in field_names:
                    angle_data = table['joint_angles'][:]  # 形状应该是 (n_samples, 20)
                    if len(angle_data.shape) == 2 and angle_data.shape[1] == 20:  #确定有20个关节角度
                        angle_scaler = StandardScaler()   # 创建标准化器
                        angle_scaler.fit(angle_data)  
                        self.angle_scalers[file_idx] = angle_scaler
                        
            else:
                # 非结构化数组，假设是简单的数值矩阵
                data = table[:]
                if len(data.shape) == 2:
                    # 假设前16列是EMG，后20列是关节角度
                    if data.shape[1] >= 16:
                        emg_data = data[:, :16]
                        emg_scaler = StandardScaler()
                        emg_scaler.fit(emg_data)
                        self.emg_scalers[file_idx] = emg_scaler
                    
                    if data.shape[1] >= 36:  # 16 EMG + 20 joint angles
                        angle_data = data[:, 16:36]
                        angle_scaler = StandardScaler()
                        angle_scaler.fit(angle_data)
                        self.angle_scalers[file_idx] = angle_scaler
                        
        except Exception as e:
            logger.warning(f"为文件索引 {file_idx} 拟合标准化器时出错: {str(e)}")
    
    def _read_window_data(self, file_idx: int, start_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """读取指定窗口的数据"""
        hdf5_file = self.hdf5_files[file_idx]
        
        with h5py.File(hdf5_file, 'r') as f:
            table = f[self.hdf5_group][self.table_name]
            
            # 读取窗口数据
            end_idx = start_idx + self.window_size
            window_raw = table[start_idx:end_idx]
            
            # 根据数据结构解析
            if hasattr(table.dtype, 'names') and table.dtype.names:
                # 结构化数组，直接访问字段
                field_names = table.dtype.names
                
                # 提取EMG数据
                if 'emg' in field_names:
                    emg_data = window_raw['emg'].astype(np.float32)  # 形状: (window_size, 16)
                else:
                    emg_data = np.zeros((self.window_size, 16), dtype=np.float32)
                
                # 提取关节角度数据
                if 'joint_angles' in field_names:
                    angle_data = window_raw['joint_angles'].astype(np.float32)  # 形状: (window_size, 20)
                else:
                    angle_data = np.zeros((self.window_size, 20), dtype=np.float32)
                    
            else:
                # 非结构化数组
                window_data = window_raw.astype(np.float32)
                if len(window_data.shape) == 2:
                    # 假设前16列是EMG，后20列是关节角度
                    if window_data.shape[1] >= 16:
                        emg_data = window_data[:, :16]
                    else:
                        emg_data = np.zeros((self.window_size, 16), dtype=np.float32)
                    
                    if window_data.shape[1] >= 36:
                        angle_data = window_data[:, 16:36]
                    else:
                        angle_data = np.zeros((self.window_size, 20), dtype=np.float32)
                else:
                    # 意外的数据形状
                    emg_data = np.zeros((self.window_size, 16), dtype=np.float32)
                    angle_data = np.zeros((self.window_size, 20), dtype=np.float32)
            
            # 标准化
            if self.normalize:
                if file_idx in self.emg_scalers:
                    emg_data = self.emg_scalers[file_idx].transform(emg_data)
                if file_idx in self.angle_scalers:
                    angle_data = self.angle_scalers[file_idx].transform(angle_data)
            
            return emg_data, angle_data
    
    def __len__(self) -> int:  #这种是类的公共方法，外部可以调用
        """返回数据集大小"""
        return len(self.data_indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取指定索引的数据样本
        
        Args:
            idx: 样本索引
            
        Returns:
            emg_data: EMG数据 (window_size, emg_channels)
            angle_data: 角度数据 (window_size, angle_channels)
        """
        file_idx, start_idx = self.data_indices[idx]
        emg_data, angle_data = self._read_window_data(file_idx, start_idx)
        
        # 转换为PyTorch张量
        emg_tensor = torch.from_numpy(emg_data)
        angle_tensor = torch.from_numpy(angle_data)
        
        return emg_tensor, angle_tensor
    
    def get_scalers(self) -> Tuple[Dict[int, Any], Dict[int, Any]]:
        """获取标准化器，用于反向变换"""
        return self.emg_scalers, self.angle_scalers


def load_hdf5_files(dataset_path: str) -> List[str]:   #这种是模块级函数，外部可以调用
    """
    从指定目录加载所有HDF5文件
    
    Args:
        dataset_path: 数据集路径
        
    Returns:
        HDF5文件路径列表
    """
    hdf5_files = []
    
    if not os.path.exists(dataset_path):
        logger.error(f"数据集路径不存在: {dataset_path}")
        return hdf5_files
    
    # 遍历目录下的所有文件
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.hdf5') or file.endswith('.h5'):
                file_path = os.path.join(root, file)
                hdf5_files.append(file_path)
    
    logger.info(f"在路径 {dataset_path} 中找到 {len(hdf5_files)} 个HDF5文件")
    return sorted(hdf5_files)
