"""
EMG到姿态预测 - 统一训练主程序
基于YAML配置文件驱动的一键训练系统

使用方法:
1. 修改下面的SELECTED_CONFIG变量选择配置
2. 运行: python main.py

配置选项:
- 'default': 标准配置（50轮训练）
- 'quick_demo': 快速演示配置（10轮训练） 
- 'high_performance': 高性能配置（200轮训练）
"""

import os
import sys
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
import yaml
import torch
import numpy as np

# 项目模块导入
from config import get_config_by_name, get_available_configs
from data import create_dataloaders
from models import create_model
from trainers import EMG2PoseTrainer
from utils import setup_logging
from utils.metrics import calculate_metrics, calculate_per_joint_metrics
from utils.visualization import plot_predictions, plot_training_history

# ===== 配置选择区域 =====
# 修改这里来选择不同的配置文件
SELECTED_CONFIG = 'emg2pose_mimic'  # 可选: 'quick_demo', 'default', 'high_performance'

# 高级选项
RUN_MODE = 'full'           # 'train' | 'evaluate' | 'full'
USE_TENSORBOARD = True      # 是否启用TensorBoard
SAVE_VISUALIZATIONS = True  # 是否保存可视化结果

logger = logging.getLogger(__name__)


class TrainingPipeline:
    """EMG到姿态预测训练管道"""
    
    def __init__(self, config_name: str, run_mode: str = 'full'):
        """
        初始化训练管道
        
        Args:
            config_name: 配置名称
            run_mode: 运行模式 ('train', 'evaluate', 'full')
        """
        self.config_name = config_name
        self.run_mode = run_mode
        
        # 加载配置
        self.config = get_config_by_name(config_name)
        
        # 创建时间戳输出目录
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(self.config.logging.log_dir) / f"{config_name}_{self.timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置日志
        self._setup_logging()
        
        # 保存配置快照
        self._save_config_snapshot()
        
        # 初始化结果存储
        self.results = {}
        
        logger.info("🚀 EMG到姿态预测训练管道初始化完成")
        logger.info(f"📁 输出目录: {self.output_dir}")
        logger.info(f"🔧 运行模式: {run_mode}")
        logger.info(f"📊 设备: {self.config.training.device}")
        logger.info(f"⚙️  配置: {config_name}")

    def _setup_logging(self):
        """设置日志系统"""
        log_file = self.output_dir / f"training_{self.timestamp}.log"
        
        # 配置日志格式
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        logger.info(f"📝 日志文件: {log_file}")

    def _save_config_snapshot(self):
        """保存当前实验的配置快照"""
        config_file = self.output_dir / f"config_{self.timestamp}.yaml"
        self.config.save_config(str(config_file))
        logger.info(f"💾 配置快照已保存: {config_file}")

    def _print_system_info(self):
        """打印系统信息"""
        logger.info("🖥️  系统信息:")
        logger.info(f"   Python版本: {sys.version}")
        logger.info(f"   PyTorch版本: {torch.__version__}")
        logger.info(f"   CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"   GPU设备: {torch.cuda.get_device_name()}")
            logger.info(f"   GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    def load_data(self) -> Tuple[Optional[Any], Optional[Any], Optional[Any]]:
        """
        加载数据
        
        Returns:
            (train_loader, val_loader, test_loader)
        """
        logger.info("📥 开始数据加载...")
        
        try:
            train_loader, val_loader, test_loader = create_dataloaders(self.config)
            
            logger.info("✅ 数据加载成功:")
            logger.info(f"   📊 训练集样本数: {len(train_loader.dataset):,}")
            logger.info(f"   📊 验证集样本数: {len(val_loader.dataset):,}")
            logger.info(f"   📊 测试集样本数: {len(test_loader.dataset):,}")
            
            # 检查数据形状
            for batch in train_loader:
                logger.info(f"   📏 EMG数据形状: {batch['emg'].shape}")
                logger.info(f"   📏 关节角度形状: {batch['joint_angles'].shape}")
                logger.info(f"   📏 IK失败mask形状: {batch['no_ik_failure'].shape}")
                break
                
            return train_loader, val_loader, test_loader
            
        except Exception as e:
            logger.error(f"❌ 数据加载失败: {str(e)}")
            return None, None, None

    def create_model(self) -> torch.nn.Module:
        """
        创建模型
        
        Returns:
            模型实例
        """
        logger.info("🔧 创建模型...")
        
        try:
            # 根据配置确定模型类型
            if hasattr(self.config.model, 'type') and self.config.model.type == 'VEMG2PoseWithInitialState':
                model = create_model('vemg2pose', self.config)
                logger.info("使用VEMG2PoseWithInitialState模型")
            else:
                model = create_model('tds_lstm', self.config)
                logger.info("使用TDS-LSTM模型")
            
            # 记录模型信息
            if hasattr(model, 'get_model_size'):
                total_params = model.get_model_size()
                trainable_params = model.get_trainable_params() if hasattr(model, 'get_trainable_params') else total_params
                logger.info(f"📊 模型参数统计:")
                logger.info(f"   📏 总参数数量: {total_params:,}")
                logger.info(f"   🎯 可训练参数: {trainable_params:,}")
                logger.info(f"   💾 模型大小: {total_params * 4 / 1024 / 1024:.2f} MB (FP32)")
            
            logger.info("✅ 模型创建成功")
            return model
            
        except Exception as e:
            logger.error(f"❌ 模型创建失败: {str(e)}")
            raise

    def train_model(self, model: torch.nn.Module, train_loader, val_loader, test_loader) -> Tuple[Any, Dict[str, Any]]:
        """
        训练模型
        
        Args:
            model: 要训练的模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器 
            test_loader: 测试数据加载器
            
        Returns:
            (trainer, results)
        """
        logger.info("🎯 开始模型训练...")
        logger.info(f"📊 训练配置:")
        logger.info(f"   🔄 训练轮数: {self.config.training.num_epochs}")
        logger.info(f"   📦 批次大小: {self.config.training.batch_size}")
        logger.info(f"   📈 学习率: {self.config.training.learning_rate}")
        logger.info(f"   🎛️ 优化器: {self.config.training.optimizer}")
        
        try:
            # 创建训练器
            trainer = EMG2PoseTrainer(
                model=model,
                config=self.config,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader
            )
            
            # 开始训练
            start_time = time.time()
            results = trainer.train()
            training_time = time.time() - start_time
            
            results['training_time'] = training_time
            self.results['training'] = results
            
            logger.info("🎉 模型训练完成!")
            logger.info(f"⏱️ 总训练时间: {training_time:.2f} 秒")
            logger.info(f"📈 最佳验证损失: {results['best_val_loss']:.6f}")
            
            return trainer, results
            
        except Exception as e:
            logger.error(f"❌ 模型训练失败: {str(e)}")
            raise

    def evaluate_model(self, trainer: Any) -> Dict[str, Any]:
        """
        评估模型
        
        Args:
            trainer: 训练器实例
            
        Returns:
            评估结果
        """
        logger.info("🔍 开始模型评估...")
        
        try:
            evaluation_results = trainer.evaluate()
            self.results['evaluation'] = evaluation_results
            
            if evaluation_results:
                logger.info("✅ 模型评估完成:")
                logger.info(f"   📉 测试损失: {evaluation_results['loss']:.6f}")
                if 'mse' in evaluation_results:
                    logger.info(f"   📉 MSE: {evaluation_results['mse']:.6f}")
                if 'r2' in evaluation_results:
                    logger.info(f"   📈 R²: {evaluation_results['r2']:.6f}")
                if 'mae' in evaluation_results:
                    logger.info(f"   📏 MAE: {evaluation_results['mae']:.6f}")
            
            return evaluation_results
            
        except Exception as e:
            logger.error(f"❌ 模型评估失败: {str(e)}")
            return {}

    def create_visualizations(self):
        """创建可视化结果"""
        if not SAVE_VISUALIZATIONS:
            return
            
        logger.info("🎨 生成可视化结果...")
        
        try:
            vis_dir = self.output_dir / "visualizations"
            vis_dir.mkdir(exist_ok=True)
            
            # 如果有训练历史，绘制训练曲线
            if 'training' in self.results and 'train_history' in self.results['training']:
                train_hist = self.results['training']['train_history']['loss']
                val_hist = self.results['training']['val_history']['loss']
                
                plot_training_history(
                    train_hist, val_hist,
                    save_path=str(vis_dir / "training_history.png")
                )
                logger.info(f"✅ 训练历史曲线已保存: {vis_dir / 'training_history.png'}")
                
        except ImportError:
            logger.warning("⚠️ 可视化库未安装，跳过可视化生成")
            logger.info("💡 提示: pip install matplotlib seaborn")
        except Exception as e:
            logger.warning(f"⚠️ 可视化生成失败: {str(e)}")

    def save_model(self, model: torch.nn.Module):
        """保存最终模型"""
        logger.info("💾 保存最终模型...")
        
        try:
            model_dir = self.output_dir / "models"
            model_dir.mkdir(exist_ok=True)
            
            # 保存完整模型
            model_path = model_dir / f"best_model_{self.timestamp}.pth"
            torch.save({
                'model_state_dict': model.state_dict(),
                'config_name': self.config_name,
                'timestamp': self.timestamp,
                'results': self.results
            }, model_path)
            
            logger.info(f"✅ 模型已保存: {model_path}")
            
        except Exception as e:
            logger.error(f"❌ 模型保存失败: {str(e)}")

    def generate_report(self):
        """生成实验报告"""
        logger.info("📄 生成实验报告...")
        
        report_file = self.output_dir / f"experiment_report_{self.timestamp}.md"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(f"# EMG到姿态预测实验报告\n\n")
            f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**配置名称**: {self.config_name}\n")
            f.write(f"**运行模式**: {self.run_mode}\n\n")
            
            # 配置信息
            f.write("## 实验配置\n\n")
            f.write(f"- **数据集路径**: {self.config.data.dataset_path}\n")
            f.write(f"- **训练轮数**: {self.config.training.num_epochs}\n")
            f.write(f"- **批次大小**: {self.config.training.batch_size}\n")
            f.write(f"- **学习率**: {self.config.training.learning_rate}\n")
            f.write(f"- **设备**: {self.config.training.device}\n\n")
            
            # 模型配置
            f.write("## 模型配置\n\n")
            f.write(f"- **输入维度**: {self.config.model.input_size}\n")
            f.write(f"- **输出维度**: {self.config.model.output_size}\n")
            f.write(f"- **编码器特征**: {self.config.model.encoder_features}\n")
            f.write(f"- **解码器隐藏层**: {self.config.model.decoder_hidden}\n")
            f.write(f"- **解码器层数**: {self.config.model.decoder_layers}\n\n")
            
            # 实验结果
            if 'training' in self.results:
                training_results = self.results['training']
                f.write("## 训练结果\n\n")
                f.write(f"- **最佳验证损失**: {training_results['best_val_loss']:.6f}\n")
                f.write(f"- **训练时间**: {training_results.get('training_time', 0):.2f} 秒\n\n")
            
            if 'evaluation' in self.results and self.results['evaluation']:
                eval_results = self.results['evaluation']
                f.write("## 测试结果\n\n")
                f.write(f"- **测试损失**: {eval_results['loss']:.6f}\n")
                if 'mse' in eval_results:
                    f.write(f"- **MSE**: {eval_results['mse']:.6f}\n")
                if 'mae' in eval_results:
                    f.write(f"- **MAE**: {eval_results['mae']:.6f}\n")
                if 'r2' in eval_results:
                    f.write(f"- **R²**: {eval_results['r2']:.6f}\n")
                f.write("\n")
            
            # 文件列表
            f.write("## 生成的文件\n\n")
            for file_path in sorted(self.output_dir.rglob("*")):
                if file_path.is_file() and file_path != report_file:
                    relative_path = file_path.relative_to(self.output_dir)
                    f.write(f"- `{relative_path}`\n")
        
        logger.info(f"✅ 实验报告已保存: {report_file}")

    def run(self) -> bool:
        """
        执行完整的训练流程
        
        Returns:
            是否成功完成
        """
        logger.info("🚀 开始EMG到姿态预测训练...")
        
        try:
            # 打印系统信息
            self._print_system_info()
            
            # 设置随机种子
            torch.manual_seed(42)
            np.random.seed(42)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(42)
            
            # 数据加载
            train_loader, val_loader, test_loader = self.load_data()
            if train_loader is None:
                logger.error("❌ 数据加载失败，终止执行")
                return False
            
            # 创建模型
            model = self.create_model()
            
            # 训练或评估
            if self.run_mode in ['train', 'full']:
                trainer, training_results = self.train_model(model, train_loader, val_loader, test_loader)
                self.save_model(model)
                
                # 如果是完整模式，继续评估
                if self.run_mode == 'full' and test_loader is not None:
                    self.evaluate_model(trainer)
                    
            elif self.run_mode == 'evaluate':
                logger.error("❌ 仅评估模式需要预训练模型，暂不支持")
                return False
            
            # 创建可视化
            self.create_visualizations()
            
            # 生成报告
            self.generate_report()
            
            # TensorBoard启动提示
            if USE_TENSORBOARD:
                tb_cmd = f"tensorboard --logdir={self.config.logging.log_dir}"
                logger.info(f"💡 启动TensorBoard查看训练过程: {tb_cmd}")
            
            logger.info("🎉 训练流程完成!")
            return True
            
        except Exception as e:
            logger.error(f"❌ 执行过程中发生错误: {str(e)}")
            return False


def main():
    """主函数"""
    print("=" * 60)
    print("EMG到姿态预测 - 深度学习训练系统")
    print("=" * 60)
    
    # 显示可用配置
    available_configs = get_available_configs()
    print(f"\n📋 可用配置文件:")
    for name, path in available_configs.items():
        indicator = "👉" if name == SELECTED_CONFIG else "  "
        print(f"{indicator} {name}: {path}")
    
    print(f"\n🎯 当前选择的配置: {SELECTED_CONFIG}")
    print(f"🔧 运行模式: {RUN_MODE}")
    print("-" * 60)
    
    try:
        # 创建并运行训练管道
        pipeline = TrainingPipeline(
            config_name=SELECTED_CONFIG,
            run_mode=RUN_MODE
        )
        
        success = pipeline.run()
        
        if success:
            print("\n🎉 训练任务成功完成!")
        else:
            print("\n❌ 训练任务失败")
            return 1
            
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断执行")
        return 1
    except Exception as e:
        print(f"\n❌ 执行错误: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
