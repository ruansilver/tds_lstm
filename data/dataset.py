"""
数据集类 - 对齐版本
基于WindowedEmgDataset设计，支持padding、jitter、IK失败检测等功能
"""

import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def get_ik_failures_mask(joint_angles: np.ndarray) -> np.ndarray:
    """
    计算IK失败mask（True表示没有失败）
    
    检测全零的joint_angles（表示IK失败）
    
    Args:
        joint_angles: (..., num_joints) 关节角度数组
        
    Returns:
        mask: (...,) 布尔数组，True表示该位置没有IK失败
    """
    zeros = np.zeros_like(joint_angles)
    is_zero = np.isclose(joint_angles, zeros)
    return ~np.all(is_zero, axis=-1)


def get_contiguous_ones(binary_vector: np.ndarray) -> List[Tuple[int, int]]:
    """
    获取连续True值的(start_idx, end_idx)列表
    
    Args:
        binary_vector: 布尔向量
        
    Returns:
        连续块的列表 [(start, end), ...]
    """
    if (binary_vector == 0).all():
        return []
    
    ones = np.where(binary_vector)[0]
    boundaries = np.where(np.diff(ones) != 1)[0]
    return [
        (ones[i], ones[j])
        for i, j in zip(
            np.insert(boundaries + 1, 0, 0), 
            np.append(boundaries, len(ones) - 1)
        )
    ]


class SessionData:
    """
    单个HDF5会话的只读接口
    
    特点：
    - 保持HDF5文件打开以提高访问效率
    - 只加载metadata到内存
    - 数据直接从磁盘读取（懒加载）
    - 支持上下文管理器
    
    Args:
        hdf5_path: HDF5文件路径
        hdf5_group: HDF5组名
        table_name: 数据表名
        chunk_cache_size: HDF5 chunk缓存大小（字节）
    """
    
    def __init__(
        self,
        hdf5_path: str,
        hdf5_group: str = "emg2pose",
        table_name: str = "timeseries",
        chunk_cache_size: int = 64 * 1024 * 1024  # 64MB
    ):
        self.hdf5_path = hdf5_path
        self.hdf5_group = hdf5_group
        self.table_name = table_name
        
        # 使用chunk cache优化连续读取性能
        self._file = h5py.File(
            hdf5_path, 
            'r', 
            rdcc_nbytes=chunk_cache_size,
            rdcc_w0=0.75  # chunk cache权重参数
        )
        
        # 检查数据结构
        if hdf5_group not in self._file:
            raise ValueError(f"HDF5文件中未找到组 '{hdf5_group}'")
        
        group = self._file[hdf5_group]
        if table_name not in group:
            raise ValueError(f"HDF5组中未找到表 '{table_name}'")
        
        # 引用数据表（不加载到内存）
        self.timeseries = group[table_name]
        self.data_length = len(self.timeseries)
        
        # 加载metadata（通常很小）
        self.metadata = dict(group.attrs.items()) if hasattr(group, 'attrs') else {}
        
        # 检测数据字段
        self.is_structured = hasattr(self.timeseries.dtype, 'names') and self.timeseries.dtype.names
        if self.is_structured:
            self.field_names = list(self.timeseries.dtype.names)
        else:
            self.field_names = []
        
        logger.debug(f"SessionData初始化: {os.path.basename(hdf5_path)}, 长度: {self.data_length}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
    
    def __len__(self):
        return self.data_length
    
    def close(self):
        """关闭HDF5文件"""
        if hasattr(self, '_file') and self._file:
            self._file.close()
            self._file = None
    
    @property
    def no_ik_failure(self) -> np.ndarray:
        """
        获取IK失败mask（缓存）
        
        Returns:
            布尔数组，True表示该位置没有IK失败
        """
        if not hasattr(self, '_no_ik_failure'):
            # 读取所有joint_angles数据并计算mask
            if self.is_structured and 'joint_angles' in self.field_names:
                joint_angles = self.timeseries['joint_angles'][:]
                self._no_ik_failure = get_ik_failures_mask(joint_angles)
            else:
                # 如果没有joint_angles字段，假设全部有效
                self._no_ik_failure = np.ones(self.data_length, dtype=bool)
        return self._no_ik_failure
    
    # def get_window(self, start_idx: int, end_idx: int) -> Tuple[np.ndarray, np.ndarray]:
    #     """
    #     读取指定窗口的数据
        
    #     Args:
    #         start_idx: 起始索引
    #         end_idx: 结束索引
            
    #     Returns:
    #         emg_data: EMG数据 (window_size, 16)
    #         angle_data: 关节角度数据 (window_size, 20)
    #     """
    #     # 从磁盘读取窗口数据（懒加载）
    #     window_raw = self.timeseries[start_idx:end_idx]
        
    #     # 根据数据结构解析
    #     if self.is_structured:
    #         # 结构化数组
    #         emg_data = window_raw['emg'].astype(np.float32) if 'emg' in self.field_names else None
    #         angle_data = window_raw['joint_angles'].astype(np.float32) if 'joint_angles' in self.field_names else None
    #     else:
    #         # 非结构化数组：假设前16列是EMG，后20列是关节角度
    #         window_data = window_raw.astype(np.float32)
    #         if len(window_data.shape) == 2 and window_data.shape[1] >= 36:
    #             emg_data = window_data[:, :16]
    #             angle_data = window_data[:, 16:36]
    #         else:
    #             emg_data = None
    #             angle_data = None
        
    #     return emg_data, angle_data
    def get_window(self, start_idx: int, end_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
    读取指定窗口的数据
    
    Args:
        start_idx: 起始索引
        end_idx: 结束索引
        
    Returns:
        emg_data: EMG数据 (window_size, 16)
        angle_data: 关节角度数据 (window_size, 20)
        """
        # 注意：self.timeseries 是 h5py.Dataset，类型是结构化数据
        try:
            emg_data = self.timeseries['emg'][start_idx:end_idx].astype(np.float32)
            angle_data = self.timeseries['joint_angles'][start_idx:end_idx].astype(np.float32)
        except Exception as e:
            print(f"[get_window] 数据提取失败: {e}")
            print("timeseries dtype:", self.timeseries.dtype)
            return None, None
        return emg_data, angle_data
    def compute_statistics_sampled(
        self, 
        max_samples: int = 10000
    ) -> Tuple[Optional[Dict], Optional[Dict]]:
        """
        计算标准化统计量（采样版本，内存友好）
        
        Args:
            max_samples: 最大采样数量
            
        Returns:
            emg_stats: EMG统计量 {'mean': array, 'std': array}
            angle_stats: 角度统计量 {'mean': array, 'std': array}
        """
        sample_size = min(max_samples, self.data_length)
        
        # 均匀采样索引
        if sample_size < self.data_length:
            indices = np.linspace(0, self.data_length - 1, sample_size, dtype=int)
        else:
            indices = np.arange(self.data_length)
        
        # 读取采样数据
        sampled = self.timeseries[indices]
        
        emg_stats = None
        angle_stats = None
        
        if self.is_structured:
            if 'emg' in self.field_names:
                emg_data = sampled['emg'].astype(np.float32)
                emg_stats = {
                    'mean': np.mean(emg_data, axis=0),
                    'std': np.std(emg_data, axis=0) + 1e-8  # 避免除零
                }
            
            if 'joint_angles' in self.field_names:
                angle_data = sampled['joint_angles'].astype(np.float32)
                angle_stats = {
                    'mean': np.mean(angle_data, axis=0),
                    'std': np.std(angle_data, axis=0) + 1e-8
                }
        else:
            sampled_data = sampled.astype(np.float32)
            if len(sampled_data.shape) == 2:
                if sampled_data.shape[1] >= 16:
                    emg_data = sampled_data[:, :16]
                    emg_stats = {
                        'mean': np.mean(emg_data, axis=0),
                        'std': np.std(emg_data, axis=0) + 1e-8
                    }
                
                if sampled_data.shape[1] >= 36:
                    angle_data = sampled_data[:, 16:36]
                    angle_stats = {
                        'mean': np.mean(angle_data, axis=0),
                        'std': np.std(angle_data, axis=0) + 1e-8
                    }
        
        return emg_stats, angle_stats


class TimeSeriesDataset(Dataset):
    """
    时序数据集类（对齐版本）
    
    特点：
    - 懒加载：数据保留在磁盘，按需读取
    - padding支持：left和right padding用于上下文
    - jitter：训练时随机抖动窗口位置
    - skip_ik_failures：自动跳过IK失败的窗口
    - 字典格式返回：与训练器对齐
    - BCT格式：(batch, channels, time)
    
    Args:
        hdf5_files: HDF5文件路径列表
        window_size: 滑动窗口大小
        hdf5_group: HDF5文件中的组名
        table_name: 表名
        normalize: 是否标准化数据
        stride: 滑动窗口步长
        padding: (left_padding, right_padding)元组
        jitter: 是否启用训练时抖动
        skip_ik_failures: 是否跳过IK失败的窗口
        max_samples_for_normalize: 标准化时的最大采样数
        chunk_cache_size: HDF5 chunk缓存大小（字节）
    """
    
    def __init__(
        self,
        hdf5_files: List[str],
        window_size: int = 11750,
        hdf5_group: str = "emg2pose",
        table_name: str = "timeseries",
        normalize: bool = False,
        stride: int = 1,
        padding: Tuple[int, int] = (0, 0),
        jitter: bool = False,
        skip_ik_failures: bool = False,
        max_samples_for_normalize: int = 10000,
        chunk_cache_size: int = 64 * 1024 * 1024
    ):
        self.hdf5_files = hdf5_files
        self.window_size = window_size
        self.hdf5_group = hdf5_group
        self.table_name = table_name
        self.normalize = normalize
        self.stride = stride
        self.left_padding, self.right_padding = padding
        self.jitter = jitter
        self.skip_ik_failures = skip_ik_failures
        self.max_samples_for_normalize = max_samples_for_normalize
        self.chunk_cache_size = chunk_cache_size
        
        # 窗口索引：存储 (file_idx, start_idx, end_idx) 元组
        self.window_indices = []
        
        # 标准化统计量（轻量级）
        self.emg_stats = {}   # {file_idx: {'mean': array, 'std': array}}
        self.angle_stats = {}
        
        # 会话数据缓存（用于worker进程）
        self._sessions = {}
        
        # 预处理：创建索引和计算统计量
        self._preprocess()
        
        logger.info(f"数据集初始化完成：{len(self.window_indices)} 个样本窗口")
        logger.info(f"  - padding: left={self.left_padding}, right={self.right_padding}")
        logger.info(f"  - jitter: {self.jitter}")
        logger.info(f"  - skip_ik_failures: {self.skip_ik_failures}")
    
    def _preprocess(self):
        """预处理：构建窗口索引并计算标准化统计量"""
        logger.info(f"开始预处理 {len(self.hdf5_files)} 个HDF5文件...")
        
        for file_idx, hdf5_file in enumerate(self.hdf5_files):
            if not os.path.exists(hdf5_file):
                logger.warning(f"文件不存在，跳过: {hdf5_file}")
                continue
            
            try:
                # 使用SessionData快速获取元数据
                with SessionData(
                    hdf5_file, 
                    self.hdf5_group, 
                    self.table_name,
                    self.chunk_cache_size
                ) as session:
                    data_length = len(session)
                    
                    # 检查数据长度
                    if data_length < self.window_size:
                        logger.warning(
                            f"文件 {os.path.basename(hdf5_file)} 数据长度 {data_length} "
                            f"小于窗口大小 {self.window_size}，跳过"
                        )
                        continue
                    
                    # 根据skip_ik_failures决定如何创建窗口索引
                    if self.skip_ik_failures:
                        # 获取IK失败mask并找到连续有效块
                        no_ik_failure = session.no_ik_failure
                        blocks = get_contiguous_ones(no_ik_failure)
                        
                        # 在每个有效块内创建窗口
                        num_windows = 0
                        for block_start, block_end in blocks:
                            block_length = block_end - block_start + 1
                            if block_length < self.window_size:
                                continue
                            
                            # 在块内创建窗口
                            for i in range((block_length - self.window_size) // self.stride + 1):
                                start_idx = block_start + i * self.stride
                                end_idx = block_end
                                self.window_indices.append((file_idx, start_idx, end_idx))
                                num_windows += 1
                    else:
                        # 不过滤IK失败，创建所有窗口
                        num_windows = (data_length - self.window_size) // self.stride + 1
                        for i in range(num_windows):
                            start_idx = i * self.stride
                            end_idx = data_length - 1
                            self.window_indices.append((file_idx, start_idx, end_idx))
                    
                    # 如果需要标准化，计算统计量（采样方式）
                    if self.normalize:
                        emg_stats, angle_stats = session.compute_statistics_sampled(
                            self.max_samples_for_normalize
                        )
                        if emg_stats:
                            self.emg_stats[file_idx] = emg_stats
                        if angle_stats:
                            self.angle_stats[file_idx] = angle_stats
                    
                    logger.info(
                        f"文件 {os.path.basename(hdf5_file)}: "
                        f"长度 {data_length}, 窗口数 {num_windows}"
                    )
            
            except Exception as e:
                logger.error(f"处理文件 {hdf5_file} 时出错: {str(e)}")
                continue
        
        logger.info(f"预处理完成，总样本数: {len(self.window_indices)}")
    
    def _get_session(self, file_idx: int) -> SessionData:
        """
        获取会话数据（带缓存）
        
        在多worker环境下，每个worker维护自己的会话缓存
        """
        if file_idx not in self._sessions:
            hdf5_file = self.hdf5_files[file_idx]
            self._sessions[file_idx] = SessionData(
                hdf5_file,
                self.hdf5_group,
                self.table_name,
                self.chunk_cache_size
            )
        return self._sessions[file_idx]
    
    def __len__(self) -> int:
        return len(self.window_indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取指定索引的数据样本
        
        Args:
            idx: 样本索引
            
        Returns:
            字典包含:
                'emg': (emg_channels, time) EMG数据
                'joint_angles': (angle_channels, time) 关节角度
                'no_ik_failure': (time,) IK失败mask
                'window_start_idx': 起始索引
                'window_end_idx': 结束索引
        """
        file_idx, start_idx, end_idx = self.window_indices[idx]
        
        # 支持jitter：随机抖动窗口位置
        offset = start_idx
        leftover = end_idx - (offset + self.window_size)
        if leftover > 0 and self.jitter:
            # 随机抖动，但不超过stride
            offset += np.random.randint(0, min(self.stride, leftover) + 1)
        
        # 获取会话数据（带缓存）
        session = self._get_session(file_idx)
        
        # 扩展窗口以包含padding
        window_start = max(offset - self.left_padding, 0)
        window_end = min(offset + self.window_size + self.right_padding, len(session))
        
        # 从磁盘读取窗口数据
        emg_data, angle_data = session.get_window(window_start, window_end)
        # 处理缺失数据
        expected_length = window_end - window_start
        if emg_data is None:
            emg_data = np.zeros((expected_length, 16), dtype=np.float32)
        if angle_data is None:
            angle_data = np.zeros((expected_length, 20), dtype=np.float32)
        
        # 获取IK失败mask
        no_ik_failure = session.no_ik_failure[window_start:window_end]
        
        # 标准化
        if self.normalize:
            if file_idx in self.emg_stats:
                mean = self.emg_stats[file_idx]['mean']
                std = self.emg_stats[file_idx]['std']
                emg_data = (emg_data - mean) / std
            
            if file_idx in self.angle_stats:
                mean = self.angle_stats[file_idx]['mean']
                std = self.angle_stats[file_idx]['std']
                angle_data = (angle_data - mean) / std
        
        # 转换为PyTorch张量并转置为(C, T)格式
        emg_tensor = torch.from_numpy(emg_data).T  # (T, C) -> (C, T)
        angle_tensor = torch.from_numpy(angle_data).T  # (T, C) -> (C, T)
        no_ik_failure_tensor = torch.from_numpy(no_ik_failure.astype(np.float32))
        
        return {
            'emg': emg_tensor,
            'joint_angles': angle_tensor,
            'no_ik_failure': no_ik_failure_tensor,
            'window_start_idx': window_start,
            'window_end_idx': window_end
        }
    
    def __del__(self):
        """清理：关闭所有打开的HDF5文件"""
        for session in self._sessions.values():
            session.close()
        self._sessions.clear()
    
    def get_stats(self) -> Tuple[Dict[int, Dict], Dict[int, Dict]]:
        """获取标准化统计量"""
        return self.emg_stats, self.angle_stats


def load_hdf5_files(dataset_path: str) -> List[str]:
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
