# EMG到姿态预测 - 深度学习训练系统

> 基于TDS-LSTM的EMG信号到人体姿态预测的深度学习项目

## 📝 项目简介

本项目实现了从肌电信号(EMG)到人体姿态预测的深度学习模型训练系统。采用Time-Delay Stabilized LSTM (TDS-LSTM)架构，支持端到端的训练和评估流程。

### ✨ 主要特性

- 🎯 **一键训练**: 通过`python main.py`即可完成完整的训练流程
- ⚙️ **配置驱动**: 所有超参数通过YAML配置文件管理，无需命令行参数
- 📊 **TensorBoard集成**: 实时监控训练过程和模型性能
- 🔧 **模块化设计**: 清晰的项目结构，便于扩展和维护
- 📈 **自动报告**: 自动生成实验报告和可视化结果
- 🚀 **多配置支持**: 内置快速演示、标准训练、高性能配置

## 📁 项目结构

```
mytrain/
├── configs/              # 配置文件目录
│   ├── default.yaml     # 默认配置（50轮训练）
│   ├── quick_demo.yaml  # 快速演示（10轮训练）
│   └── high_performance.yaml # 高性能配置（200轮训练）
├── data/                # 数据处理模块
│   ├── __init__.py
│   ├── dataset.py       # 数据集定义
│   └── dataloader.py    # 数据加载器
├── models/              # 模型定义模块
│   ├── __init__.py
│   ├── model_factory.py # 模型工厂（支持动态注册）
│   └── tds_lstm_model.py # TDS-LSTM模型实现
├── trainers/            # 训练器模块
│   ├── __init__.py
│   ├── trainer.py       # 主训练器
│   └── evaluator.py     # 评估器
├── utils/               # 工具函数模块
│   ├── __init__.py
│   ├── checkpoint.py    # 检查点管理
│   ├── early_stopping.py # 早停机制
│   ├── logging_utils.py # 日志工具
│   ├── metrics.py       # 评估指标
│   └── visualization.py # 可视化工具
├── logs/                # 训练日志输出
├── checkpoints/         # 模型检查点
├── main.py              # 统一主入口文件 ⭐
├── config.py            # 配置加载器 ⭐
├── requirements.txt     # 依赖清单
└── README.md           # 项目说明
```

## 🚀 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone <your-repo-url>
cd mytrain

# 安装依赖
pip install -r requirements.txt

# 验证PyTorch安装
python -c "import torch; print(f'PyTorch版本: {torch.__version__}')"
```

### 2. 数据准备

确保您的数据集路径正确设置在配置文件中：

```yaml
# configs/default.yaml
data:
  dataset_path: "D:/Dataset/emg2pose_dataset_mini"  # 修改为您的数据集路径
  emg_channels: 16          # EMG通道数
  joint_angles_channels: 20 # 关节角度通道数
  window_size: 500         # 时间窗口大小
```

### 3. 一键训练

编辑`main.py`选择配置，然后运行：

```python
# main.py 中修改配置选择
SELECTED_CONFIG = 'default'  # 可选: 'quick_demo', 'default', 'high_performance'

# 运行训练
python main.py
```

### 4. 查看结果

训练完成后，检查输出目录：

```
logs/default_20241014_153000/
├── training_20241014_153000.log        # 训练日志
├── config_20241014_153000.yaml         # 配置快照
├── models/
│   └── best_model_20241014_153000.pth  # 最佳模型
├── visualizations/
│   └── training_history.png            # 训练曲线
└── experiment_report_20241014_153000.md # 实验报告
```

### 5. 启动TensorBoard

```bash
tensorboard --logdir=logs
# 在浏览器中打开 http://localhost:6006
```

## ⚙️ 配置说明

### 可用配置文件

| 配置名称 | 文件 | 描述 | 训练轮数 | 适用场景 |
|---------|------|-----|---------|----------|
| `quick_demo` | `configs/quick_demo.yaml` | 快速演示配置 | 10轮 | 测试代码、快速验证 |
| `default` | `configs/default.yaml` | 标准配置 | 50轮 | 一般实验、论文复现 |
| `high_performance` | `configs/high_performance.yaml` | 高性能配置 | 200轮 | 追求最佳性能 |

### 配置文件结构

```yaml
# 数据配置
data:
  dataset_path: "D:/Dataset/emg2pose_dataset_mini"
  emg_channels: 16
  joint_angles_channels: 20
  window_size: 500
  split_ratio: [0.7, 0.15, 0.15]  # [训练, 验证, 测试]

# 模型配置
model:
  input_size: 16           # 输入维度（EMG通道数）
  output_size: 20          # 输出维度（关节数）
  encoder_features: 64     # TDS编码器特征维度
  decoder_hidden: 64       # LSTM解码器隐藏层大小
  decoder_layers: 2        # LSTM解码器层数
  dropout: 0.2            # Dropout率

# 训练配置
training:
  num_epochs: 50          # 训练轮数
  batch_size: 32          # 批次大小
  learning_rate: 0.001    # 学习率
  optimizer: "adam"       # 优化器: adam, adamw, sgd
  scheduler: "step"       # 调度器: step, cosine, plateau
  early_stopping: true    # 早停机制
  device: "auto"          # 设备: auto, cuda, cpu

# 日志配置
logging:
  checkpoint_dir: "checkpoints"
  log_dir: "logs"
  save_every: 5           # 每N轮保存检查点
  save_best_only: true    # 只保存最佳模型
```

## 🔧 高级使用

### 自定义配置

1. 复制现有配置文件：
```bash
cp configs/default.yaml configs/my_config.yaml
```

2. 修改参数：
```yaml
# configs/my_config.yaml
training:
  num_epochs: 100
  batch_size: 64
  learning_rate: 0.0005
```

3. 更新`main.py`：
```python
SELECTED_CONFIG = 'my_config'
```

### 添加新模型

```python
# 在models/目录下创建新模型文件
# models/my_model.py
class MyModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=128):
        # 模型定义
        pass

# 在models/__init__.py中注册
from models.model_factory import register_model
from .my_model import MyModel

register_model('my_model', MyModel, {
    'input_size': 16,
    'output_size': 20,
    'hidden_size': 128
})
```

### 运行模式

修改`main.py`中的`RUN_MODE`变量：

```python
RUN_MODE = 'full'     # 完整流程：训练 + 验证 + 测试
# RUN_MODE = 'train'  # 仅训练和验证
```

## 📊 监控和可视化

### TensorBoard监控

启动TensorBoard查看实时训练进度：

```bash
tensorboard --logdir=logs
```

可以监控：
- 训练和验证损失曲线
- 学习率变化
- 各项评估指标
- 模型结构图

### 实验报告

每次训练后会自动生成Markdown格式的实验报告，包含：
- 实验配置信息
- 训练结果统计
- 模型性能指标
- 生成文件清单

## 🛠️ 开发指南

### 代码风格

- 遵循科研代码规范：简洁清晰，重点关注可读性
- 使用类型提示和详细的docstring
- 模块化设计，便于扩展和维护

### 扩展建议

1. **添加新损失函数**：在`trainers/trainer.py`中修改损失计算
2. **增加评估指标**：在`utils/metrics.py`中添加新指标
3. **自定义数据预处理**：修改`data/dataset.py`
4. **集成其他优化器**：在`trainers/trainer.py`中扩展优化器选项

## 📚 技术细节

### TDS-LSTM架构

Time-Delay Stabilized LSTM结合了：
- **时间延迟卷积编码器**：提取EMG信号特征
- **LSTM解码器**：序列到序列映射
- **残差连接**：提升训练稳定性

### 数据流程

1. **数据加载**：HDF5格式，支持大规模数据集
2. **预处理**：滑动窗口分割，可选标准化
3. **批处理**：动态批处理，支持GPU加速
4. **评估**：多指标评估（MSE, MAE, R²等）

## 🐛 常见问题

### Q: CUDA内存不足怎么办？
A: 减小`batch_size`或使用`cpu`设备：
```yaml
training:
  batch_size: 16    # 减小批次大小
  device: "cpu"     # 或使用CPU训练
```

### Q: 如何使用自己的数据集？
A: 修改配置文件中的数据路径和格式：
```yaml
data:
  dataset_path: "your/dataset/path"
  emg_channels: 8      # 根据实际通道数修改
  joint_angles_channels: 15
```

### Q: 训练中断如何恢复？
A: 检查点会自动保存，可以从最后的检查点继续：
```python
# 在trainer.py中加载检查点逻辑已实现
# 会自动从最佳模型继续训练
```

### Q: 如何调试模型？
A: 使用快速演示配置进行调试：
```python
SELECTED_CONFIG = 'quick_demo'  # 快速训练10轮
```

## 📄 许可证

本项目采用MIT许可证，详见LICENSE文件。

## 🤝 贡献

欢迎提交Issue和Pull Request！

---

**享受您的EMG到姿态预测研究之旅！** 🚀✨




