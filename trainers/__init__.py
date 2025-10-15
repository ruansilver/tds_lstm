"""
训练和评估模块
"""

from .trainer import EMG2PoseTrainer
from .evaluator import ModelEvaluator

__all__ = ['EMG2PoseTrainer', 'ModelEvaluator']
