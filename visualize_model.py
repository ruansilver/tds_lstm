"""
模型可视化脚本
使用 torchview 对模型进行可视化，生成PNG/SVG/PDF图片文件
一键运行，自动加载配置、创建模型并生成可视化输出
"""

import os
import sys
import logging
from pathlib import Path
import torch
import numpy as np

# 配置常量 - 在此处修改目标配置文件
CONFIG_FILE = "configs/default.yaml"
OUTPUT_DIR = "visualizations"

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_output_directory(output_dir: str) -> Path:
    """
    创建输出目录
    
    Args:
        output_dir: 输出目录路径
        
    Returns:
        Path对象
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"输出目录: {output_path.absolute()}")
    return output_path


def load_configuration(config_file: str):
    """
    加载配置文件
    
    Args:
        config_file: 配置文件路径
        
    Returns:
        Config对象
    """
    from config import load_config_from_yaml
    
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"配置文件不存在: {config_file}")
    
    logger.info(f"加载配置文件: {config_file}")
    config = load_config_from_yaml(config_file)
    return config


def load_sample_data(config):
    """
    从数据集中加载一个样本数据
    
    Args:
        config: 配置对象
        
    Returns:
        sample_input: 样本输入张量 (1, time, channels)
    """
    from data.dataset import load_hdf5_files, TimeSeriesDataset
    
    logger.info("加载数据集...")
    
    # 获取HDF5文件列表
    hdf5_files = load_hdf5_files(config.data.dataset_path)
    
    if not hdf5_files:
        raise ValueError(f"在路径 {config.data.dataset_path} 中没有找到HDF5文件")
    
    logger.info(f"找到 {len(hdf5_files)} 个数据文件")
    
    # 创建数据集实例（只使用第一个文件以节省内存）
    dataset = TimeSeriesDataset(
        hdf5_files=hdf5_files[:1],  # 只加载第一个文件
        window_size=config.data.window_size,
        hdf5_group=config.data.hdf5_group,
        table_name=config.data.table_name,
        normalize=config.data.normalize,
        stride=config.data.stride
    )
    
    if len(dataset) == 0:
        raise ValueError("数据集为空")
    
    # 获取第一个样本
    logger.info("加载第一个样本作为示例输入...")
    emg_sample, angle_sample = dataset[0]
    
    # 添加batch维度: (time, channels) -> (1, time, channels)
    sample_input = emg_sample.unsqueeze(0)
    
    logger.info(f"样本输入形状: {sample_input.shape}")
    logger.info(f"样本输出形状: {angle_sample.shape}")
    
    return sample_input


def create_model_instance(config):
    """
    创建模型实例
    
    Args:
        config: 配置对象
        
    Returns:
        model: 模型实例
    """
    from models.model_factory import create_model
    
    logger.info("创建模型...")
    
    # 创建模型
    model = create_model(model_name='tds_lstm', config=config)
    
    # 设置为评估模式
    model.eval()
    
    # 打印模型信息
    if hasattr(model, 'get_model_info'):
        info = model.get_model_info()
        logger.info("=" * 60)
        logger.info("模型信息:")
        logger.info(f"  总参数量: {info['total_params']:,}")
        logger.info(f"  可训练参数: {info['trainable_params']:,}")
        logger.info(f"  模型大小: {info['model_size_mb']:.2f} MB")
        logger.info(f"  输入维度: {info['input_size']}")
        logger.info(f"  输出维度: {info['output_size']}")
        logger.info(f"  编码器类型: {info['encoder_type']}")
        logger.info(f"  解码器类型: {info['decoder_type']}")
        logger.info("=" * 60)
    
    return model


def visualize_model_with_torchview(model, sample_input, output_path: Path):
    """
    使用 torchview 生成模型可视化
    
    Args:
        model: PyTorch模型
        sample_input: 示例输入张量
        output_path: 输出目录路径
    """
    try:
        from torchview import draw_graph
        logger.info("使用 torchview 生成可视化...")
    except ImportError:
        logger.error("未安装 torchview，请运行: pip install torchview")
        raise ImportError("需要安装 torchview 才能运行此脚本")
    
    # 确保输入在CPU上
    sample_input_cpu = sample_input.cpu()
    model_cpu = model.cpu()
    
    # 确保模型处于评估模式
    model_cpu.eval()
    
    # 生成模型可视化
    logger.info("生成模型结构图...")
    
    try:
        # 使用 draw_graph 生成模型图
        model_graph = draw_graph(
            model_cpu,
            input_data=sample_input_cpu,
            expand_nested=True,  # 展开嵌套模块
            depth=3,  # 显示深度
            device='cpu',
            roll=False,  # 不展开循环
            hide_module_functions=False,  # 显示模块函数
            hide_inner_tensors=True,  # 隐藏内部张量以简化视图
            show_shapes=True,  # 显示张量形状
            save_graph=False,  # 暂时不自动保存，我们手动保存
        )
        
        # 保存为不同格式
        base_filename = output_path / "model_visualization"
        
        # 保存为 PNG
        try:
            png_file = f"{base_filename}.png"
            model_graph.visual_graph.render(
                filename=str(base_filename),
                format='png',
                cleanup=True
            )
            logger.info(f"PNG图片已保存: {png_file}")
        except Exception as e:
            logger.warning(f"PNG保存失败: {e}")
        
        # 保存为 SVG
        try:
            svg_file = f"{base_filename}.svg"
            model_graph.visual_graph.render(
                filename=str(base_filename),
                format='svg',
                cleanup=True
            )
            logger.info(f"SVG图片已保存: {svg_file}")
        except Exception as e:
            logger.warning(f"SVG保存失败: {e}")
        
        # 保存为 PDF
        try:
            pdf_file = f"{base_filename}.pdf"
            model_graph.visual_graph.render(
                filename=str(base_filename),
                format='pdf',
                cleanup=True
            )
            logger.info(f"PDF文件已保存: {pdf_file}")
        except Exception as e:
            logger.warning(f"PDF保存失败: {e}")
        
        logger.info(f"可视化文件已保存到: {output_path}")
        logger.info("提示: 使用图片查看器打开 PNG/SVG 文件查看模型结构")
        
    except Exception as e:
        logger.error(f"torchview 可视化失败: {e}")
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"torchview 无法可视化此模型: {e}")


def main():
    """主函数"""
    logger.info("=" * 80)
    logger.info("模型可视化脚本")
    logger.info("=" * 80)
    
    try:
        # 1. 创建输出目录
        output_path = setup_output_directory(OUTPUT_DIR)
        
        # 2. 加载配置
        config = load_configuration(CONFIG_FILE)
        
        # 3. 加载样本数据
        sample_input = load_sample_data(config)
        
        # 4. 创建模型
        model = create_model_instance(config)
        
        # 5. 生成可视化
        visualize_model_with_torchview(model, sample_input, output_path)
        
        # 6. 完成
        logger.info("=" * 80)
        logger.info("✅ 可视化完成！")
        logger.info(f"所有输出文件已保存到: {output_path.absolute()}")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

