"""
日志配置工具
"""

import logging
import os
import sys
from datetime import datetime
from typing import Optional

def setup_logging(
    log_dir: str = "logs",
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    console_output: bool = True,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    设置日志系统
    
    Args:
        log_dir: 日志文件目录
        log_level: 日志级别
        log_file: 日志文件名，如果为None则自动生成
        console_output: 是否输出到控制台
        format_string: 自定义格式字符串
        
    Returns:
        配置好的logger
    """
    # 创建日志目录
    os.makedirs(log_dir, exist_ok=True)
    
    # 设置日志级别
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    # 设置格式
    if format_string is None:
        format_string = '[%(asctime)s] %(name)s - %(levelname)s - %(message)s'
    
    formatter = logging.Formatter(
        format_string,
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 获取根logger
    logger = logging.getLogger()
    logger.setLevel(level)
    
    # 清除现有的处理器
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 控制台处理器
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # 文件处理器
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"emg2pose_{timestamp}.log"
    
    file_path = os.path.join(log_dir, log_file)
    file_handler = logging.FileHandler(file_path, encoding='utf-8')
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # 记录初始化信息
    logger.info("=" * 60)
    logger.info("EMG2Pose 深度学习项目日志系统初始化")
    logger.info(f"日志级别: {log_level}")
    logger.info(f"日志文件: {file_path}")
    logger.info(f"控制台输出: {'启用' if console_output else '禁用'}")
    logger.info("=" * 60)
    
    return logger

def get_logger(name: str) -> logging.Logger:
    """
    获取指定名称的logger
    
    Args:
        name: logger名称
        
    Returns:
        logger实例
    """
    return logging.getLogger(name)

class LoggerManager:
    """日志管理器"""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        self.loggers = {}
    
    def get_logger(
        self,
        name: str,
        log_level: str = "INFO",
        separate_file: bool = False
    ) -> logging.Logger:
        """
        获取或创建logger
        
        Args:
            name: logger名称
            log_level: 日志级别
            separate_file: 是否使用单独的日志文件
            
        Returns:
            logger实例
        """
        if name in self.loggers:
            return self.loggers[name]
        
        logger = logging.getLogger(name)
        level = getattr(logging, log_level.upper(), logging.INFO)
        logger.setLevel(level)
        
        # 如果需要单独的文件，创建文件处理器
        if separate_file:
            os.makedirs(self.log_dir, exist_ok=True)
            file_path = os.path.join(self.log_dir, f"{name}.log")
            
            file_handler = logging.FileHandler(file_path, encoding='utf-8')
            file_handler.setLevel(level)
            
            formatter = logging.Formatter(
                '[%(asctime)s] %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        self.loggers[name] = logger
        return logger

def log_function_call(func):
    """
    装饰器：记录函数调用
    
    Args:
        func: 要装饰的函数
        
    Returns:
        装饰后的函数
    """
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        logger.debug(f"调用函数: {func.__name__}")
        logger.debug(f"参数: args={args}, kwargs={kwargs}")
        
        try:
            result = func(*args, **kwargs)
            logger.debug(f"函数 {func.__name__} 执行成功")
            return result
        except Exception as e:
            logger.error(f"函数 {func.__name__} 执行失败: {str(e)}")
            raise
    
    return wrapper

def log_time(func):
    """
    装饰器：记录函数执行时间
    
    Args:
        func: 要装饰的函数
        
    Returns:
        装饰后的函数
    """
    import time
    
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            end_time = time.time()
            execution_time = end_time - start_time
            logger.info(f"函数 {func.__name__} 执行时间: {execution_time:.2f}秒")
            return result
        except Exception as e:
            end_time = time.time()
            execution_time = end_time - start_time
            logger.error(f"函数 {func.__name__} 执行失败 (用时 {execution_time:.2f}秒): {str(e)}")
            raise
    
    return wrapper

class TensorBoardLogger:
    """TensorBoard日志记录器"""
    
    def __init__(self, log_dir: str = "logs/tensorboard"):
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir)
            self.enabled = True
        except ImportError:
            logging.warning("TensorBoard未安装，跳过TensorBoard日志")
            self.enabled = False
    
    def log_scalar(self, tag: str, value: float, step: int):
        """记录标量值"""
        if self.enabled:
            self.writer.add_scalar(tag, value, step)
    
    def log_histogram(self, tag: str, values, step: int):
        """记录直方图"""
        if self.enabled:
            self.writer.add_histogram(tag, values, step)
    
    def log_image(self, tag: str, image, step: int):
        """记录图像"""
        if self.enabled:
            self.writer.add_image(tag, image, step)
    
    def close(self):
        """关闭writer"""
        if self.enabled:
            self.writer.close()
