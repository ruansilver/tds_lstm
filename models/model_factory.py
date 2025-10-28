"""
模型工厂 - 支持动态模型注册和灵活配置
提供统一的模型创建接口，支持扩展新的模型类型
"""

import torch.nn as nn
from typing import Dict, Any, Type, List, Optional
import logging
import inspect

from .tds_lstm_model import TDSLSTMModel
from .pose_modules import VEMG2PoseWithInitialState, BasePoseModule
from .tds_network import TdsNetwork
from .lstm_decoder import SequentialLSTM
from config import Config

logger = logging.getLogger(__name__)


class ModelRegistry:
    """模型注册表 - 管理所有可用的模型类型"""
    
    def __init__(self):
        self._models: Dict[str, Type[nn.Module]] = {}
        self._model_configs: Dict[str, Dict[str, Any]] = {}
        self._register_default_models()
    
    def _register_default_models(self):
        """注册默认模型"""
        self.register_model('tds_lstm', TDSLSTMModel, {
            'input_size': 16,
            'output_size': 20,
            'encoder_features': 64,
            'decoder_hidden': 64,
            'decoder_layers': 2,
            'rollout_freq': 50,
            'original_sampling_rate': 2000,
            'dropout': 0.2,
            'predict_velocity': False,
        })
        
        # 注册VEMG2PoseWithInitialState模型
        self.register_model('vemg2pose', VEMG2PoseWithInitialState, {
            'out_channels': 20,
            'num_position_steps': 500,
            'state_condition': True,
            'rollout_freq': 50,
        })
    
    def register_model(self, name: str, model_class: Type[nn.Module], default_config: Optional[Dict[str, Any]] = None):
        """
        注册新的模型类
        
        Args:
            name: 模型名称
            model_class: 模型类
            default_config: 默认配置参数
        """
        if not issubclass(model_class, nn.Module):
            raise ValueError(f"模型类必须继承自nn.Module: {model_class}")
        
        self._models[name] = model_class
        self._model_configs[name] = default_config or {}
        logger.info(f"✅ 注册模型: {name} -> {model_class.__name__}")
    
    def get_model_class(self, name: str) -> Type[nn.Module]:
        """获取模型类"""
        if name not in self._models:
            raise ValueError(f"未知的模型类型: {name}. 可用模型: {self.list_models()}")
        return self._models[name]
    
    def get_default_config(self, name: str) -> Dict[str, Any]:
        """获取模型的默认配置"""
        return self._model_configs.get(name, {}).copy()
    
    def list_models(self) -> List[str]:
        """列出所有可用的模型"""
        return list(self._models.keys())
    
    def get_model_info(self, name: str) -> Dict[str, Any]:
        """获取模型详细信息"""
        if name not in self._models:
            raise ValueError(f"未知的模型类型: {name}")
        
        model_class = self._models[name]
        signature = inspect.signature(model_class.__init__)
        
        return {
            'name': name,
            'class': model_class.__name__,
            'module': model_class.__module__,
            'parameters': list(signature.parameters.keys())[1:],  # 排除self
            'default_config': self._model_configs.get(name, {}),
            'docstring': model_class.__doc__
        }


# 全局模型注册表
_model_registry = ModelRegistry()


class ModelFactory:
    """模型工厂类 - 提供统一的模型创建接口"""
    
    @classmethod
    def register_model(cls, name: str, model_class: Type[nn.Module], default_config: Optional[Dict[str, Any]] = None):
        """注册新的模型类"""
        _model_registry.register_model(name, model_class, default_config)
    
    @classmethod
    def create_model(
        cls,
        model_name: str = 'tds_lstm',   
        config: Optional[Config] = None,
        **kwargs
    ) -> nn.Module:
        """
        创建指定类型的模型
        
        Args:
            model_name: 模型名称
            config: 配置对象
            **kwargs: 额外的模型参数
            
        Returns:
            创建的模型实例
        """
        # 获取模型类
        model_class = _model_registry.get_model_class(model_name)
        
        # 准备模型参数
        model_params = _model_registry.get_default_config(model_name)
        
        # 从配置对象更新参数
        if config is not None:
            config_params = cls._extract_config_params(model_name, config)
            model_params.update(config_params)
        
        # 覆盖额外参数
        model_params.update(kwargs)
        
        # 验证参数
        cls._validate_model_params(model_class, model_params)
        
        # 创建模型
        try:
            # 特殊处理VEMG2PoseWithInitialState模型
            if model_name == 'vemg2pose' or model_class == VEMG2PoseWithInitialState:
                model = cls._create_vemg2pose_model(model_params, config)
            else:
                model = model_class(**model_params)
                
            logger.info(f"✅ 创建模型: {model_name} ({model_class.__name__})")
            logger.debug(f"📊 模型参数: {model_params}")
            
            # 记录模型大小信息
            if hasattr(model, 'get_model_size'):
                total_params = model.get_model_size()
                logger.info(f"📏 模型参数数量: {total_params:,}")
                
            return model
            
        except Exception as e:
            logger.error(f"❌ 模型创建失败: {str(e)}")
            logger.error(f"🔍 模型类: {model_class}")
            logger.error(f"🔍 参数: {model_params}")
            raise
    
    @classmethod
    def _create_vemg2pose_model(cls, model_params: Dict[str, Any], config: Config) -> VEMG2PoseWithInitialState:
        """创建VEMG2PoseWithInitialState模型"""
        try:
            # 从配置中提取网络和解码器参数
            if config and hasattr(config, 'model'):
                model_config = config.model
                
                # 创建TDS网络
                if hasattr(model_config, 'network'):
                    network_config = model_config.network
                    
                    # 创建conv_blocks
                    conv_blocks = []
                    if hasattr(network_config, 'conv_blocks'):
                        for conv_cfg in network_config.conv_blocks:
                            from .tds_network import Conv1dBlock
                            conv_block = Conv1dBlock(
                                in_channels=int(conv_cfg['in_channels']),
                                out_channels=int(conv_cfg['out_channels']),
                                kernel_size=int(conv_cfg['kernel_size']),
                                stride=int(conv_cfg['stride'])
                            )
                            conv_blocks.append(conv_block)
                    
                    # 创建TDS stages
                    tds_stages = []
                    if hasattr(network_config, 'tds_stages'):
                        for stage_cfg in network_config.tds_stages:
                            from .tds_network import TdsStage
                            out_ch = stage_cfg.get('out_channels')
                            tds_stage = TdsStage(
                                in_channels=int(stage_cfg.get('in_channels', 256)),
                                in_conv_kernel_width=int(stage_cfg.get('in_conv_kernel_width', 17)),
                                in_conv_stride=int(stage_cfg.get('in_conv_stride', 4)),
                                num_blocks=int(stage_cfg.get('num_blocks', 2)),
                                channels=int(stage_cfg.get('channels', 16)),
                                feature_width=int(stage_cfg.get('feature_width', 16)),
                                kernel_width=int(stage_cfg.get('kernel_width', 9)),
                                out_channels=int(out_ch) if out_ch is not None else None
                            )
                            tds_stages.append(tds_stage)
                    
                    # 创建TdsNetwork（使用正确的构造方式）
                    from .tds_network import TdsNetwork as TdsNetworkImpl
                    network = TdsNetworkImpl(conv_blocks, tds_stages)
                else:
                    raise ValueError("配置中缺少network参数")
                
                # 创建解码器
                if hasattr(model_config, 'decoder'):
                    decoder_config = model_config.decoder
                    decoder = SequentialLSTM(
                        in_channels=int(decoder_config.get('in_channels', 84)),
                        out_channels=int(decoder_config.get('out_channels', 40)),
                        hidden_size=int(decoder_config.get('hidden_size', 512)),
                        num_layers=int(decoder_config.get('num_layers', 2)),
                        output_scale=float(decoder_config.get('scale', 0.01))
                    )
                else:
                    raise ValueError("配置中缺少decoder参数")
                
                # 创建VEMG2PoseWithInitialState模型
                model = VEMG2PoseWithInitialState(
                    network=network,
                    decoder=decoder,
                    out_channels=int(model_params.get('out_channels', 20)),
                    num_position_steps=int(getattr(model_config, 'num_position_steps', 500)),
                    state_condition=bool(getattr(model_config, 'state_condition', True)),
                    rollout_freq=int(model_params.get('rollout_freq', 50))
                )
                
                return model
            else:
                raise ValueError("创建VEMG2PoseWithInitialState模型需要配置对象")
                
        except Exception as e:
            logger.error(f"❌ VEMG2PoseWithInitialState模型创建失败: {str(e)}")
            raise
    
    @classmethod
    def _extract_config_params(cls, model_name: str, config: Config) -> Dict[str, Any]:
        """从配置对象中提取模型参数"""
        params = {}
        
        if model_name == 'tds_lstm':
            # TDS-LSTM模型参数映射
            params.update({
                'input_size': config.model.input_size,
                'output_size': config.model.output_size,
                'rollout_freq': getattr(config.model, 'rollout_freq', 50),
                'original_sampling_rate': getattr(config.model, 'original_sampling_rate', 2000),
                'dropout': config.model.dropout,
                'predict_velocity': getattr(config.model, 'predict_velocity', False),
            })
            
            # TDS编码器配置
            if hasattr(config.model, 'tds_stages') and config.model.tds_stages:
                params['tds_stages_config'] = config.model.tds_stages
            
            # LSTM解码器配置
            params['decoder_type'] = getattr(config.model, 'decoder_type', 'sequential_lstm')
            params['decoder_hidden_size'] = getattr(config.model, 'decoder_hidden_size', 128)
            params['decoder_num_layers'] = getattr(config.model, 'decoder_num_layers', 2)
            params['decoder_state_condition'] = getattr(config.model, 'decoder_state_condition', True)
            
        else:
            # 通用模型参数
            params.update({
                'input_size': getattr(config.model, 'input_size', 16),
                'output_size': getattr(config.model, 'output_size', 20),
                'hidden_size': getattr(config.model, 'hidden_size', 128),
                'num_layers': getattr(config.model, 'num_layers', 2),
                'dropout': getattr(config.model, 'dropout', 0.2),
                'bidirectional': getattr(config.model, 'bidirectional', True),
            })
        
        return params
    
    @classmethod
    def _validate_model_params(cls, model_class: Type[nn.Module], params: Dict[str, Any]):
        """验证模型参数"""
        try:
            signature = inspect.signature(model_class.__init__)
            valid_params = set(signature.parameters.keys()) - {'self'}
            
            # 检查是否有无效参数
            invalid_params = set(params.keys()) - valid_params
            if invalid_params:
                logger.warning(f"⚠️ 忽略无效参数: {invalid_params}")
                # 移除无效参数
                for param in invalid_params:
                    params.pop(param, None)
            
            # 检查必需参数
            required_params = [
                name for name, param in signature.parameters.items() 
                if param.default == inspect.Parameter.empty and name != 'self'
            ]
            missing_params = set(required_params) - set(params.keys())
            if missing_params:
                raise ValueError(f"缺少必需参数: {missing_params}")
                
        except Exception as e:
            logger.warning(f"⚠️ 参数验证失败: {e}")
    
    @classmethod
    def list_models(cls) -> List[str]:
        """列出所有可用的模型"""
        return _model_registry.list_models()
    
    @classmethod
    def get_model_info(cls, model_name: str) -> Dict[str, Any]:
        """获取模型详细信息"""
        return _model_registry.get_model_info(model_name)
    
    @classmethod
    def print_available_models(cls):
        """打印所有可用的模型信息"""
        models = cls.list_models()
        logger.info(f"📋 可用模型数量: {len(models)}")
        
        for model_name in models:
            try:
                info = cls.get_model_info(model_name)
                logger.info(f"  🔧 {model_name}: {info['class']}")
                if info['docstring']:
                    logger.info(f"     📝 {info['docstring'].strip().split('.')[0]}")
            except Exception as e:
                logger.warning(f"  ❌ {model_name}: 信息获取失败 ({e})")


# 便捷函数
def create_model(model_name: str = 'tds_lstm', config: Optional[Config] = None, **kwargs) -> nn.Module:
    """
    便捷的模型创建函数
    
    Args:
        model_name: 模型名称，默认为'tds_lstm'
        config: 配置对象
        **kwargs: 额外的模型参数
        
    Returns:
        创建的模型实例
    """
    return ModelFactory.create_model(model_name, config, **kwargs)


def register_model(name: str, model_class: Type[nn.Module], default_config: Optional[Dict[str, Any]] = None):
    """
    注册新模型的便捷函数
    
    Args:
        name: 模型名称
        model_class: 模型类
        default_config: 默认配置
    """
    ModelFactory.register_model(name, model_class, default_config)


def list_models() -> List[str]:
    """列出所有可用模型的便捷函数"""
    return ModelFactory.list_models()


def get_model_info(model_name: str) -> Dict[str, Any]:
    """获取模型信息的便捷函数"""
    return ModelFactory.get_model_info(model_name)
