# TDS-LSTM 模型使用说明

## 🚀 快速开始

### 直接运行训练

```bash
python main.py
```

默认使用 `configs/default.yaml` 配置。

### 切换配置

编辑 `main.py` 第 37 行：

```python
SELECTED_CONFIG = 'quick_demo'  # 可选: 'default', 'quick_demo', 'high_performance'
```

---

## 📋 配置说明

### 1. quick_demo - 快速验证（推荐新手）
- **训练轮数**: 10轮
- **模型大小**: 最小（1个TDS stage）
- **预计时间**: ~10分钟
- **适用场景**: 快速验证模型能否运行

### 2. default - 标准训练（推荐）
- **训练轮数**: 100轮
- **模型大小**: 中等（2个TDS stage）
- **预计时间**: ~2小时
- **适用场景**: 正式训练使用

### 3. low_memory - 低内存（推荐服务器）🔴
- **训练轮数**: 100轮
- **模型大小**: 中等（2个TDS stage）
- **内存占用**: ~1GB（使用梯度累积）
- **适用场景**: 服务器共享环境、内存受限情况
- **特点**: 使用batch_size=16 + 梯度累积×4，效果等同batch_size=64

### 4. high_performance - 高性能
- **训练轮数**: 200轮
- **模型大小**: 大（3个TDS stage）
- **预计时间**: ~5小时
- **适用场景**: 追求最佳性能
- **注意**: 需要更多GPU内存（batch_size=128）

---

## 🔧 模型架构

### TDS-LSTM 组合模型

```
EMG信号 (2000Hz, 16通道)
    ↓
TDS编码器 (多阶段时序特征提取)
    ↓
降采样到 50Hz (减少计算量)
    ↓
LSTM解码器 (自回归预测)
    ↓
关节角度 (20个关节)
```

### 关键特性

1. **TDS编码器**: 高效的时序特征提取
   - 时间-深度可分离卷积
   - 多阶段逐步降采样
   - LayerNorm + 残差连接

2. **降采样策略**: 2000Hz → 50Hz
   - 减少40倍计算量
   - 手部运动频率远低于EMG采样率

3. **LSTM解码器**: 自回归预测
   - 状态条件（state_condition）
   - 利用历史输出改善预测连续性

---

## 📊 配置对比

| 项目 | quick_demo | default | low_memory🔴 | high_performance |
|------|------------|---------|--------------|------------------|
| TDS Stages | 1 | 2 | 2 | 3 |
| 特征维度 | 16 | 32+64 | 32+64 | 64×3 |
| LSTM隐藏层 | 64 | 128 | 128 | 256 |
| LSTM层数 | 1 | 2 | 2 | 3 |
| Batch Size | 32 | 64 | 16 | 128 |
| 梯度累积 | 1 | 1 | 4 | 1 |
| 等效Batch | 32 | 64 | 64 | 128 |
| 内存占用 | ~2GB | ~4GB | **~1GB** | ~8GB |
| 训练轮数 | 10 | 100 | 100 | 200 |

---

## ⚙️ 训练优化

### 优化器配置
- **AdamW**: 更好的权重衰减正则化
- **学习率**: 0.0005（default）
- **权重衰减**: 0.01

### 学习率调度
- **Cosine退火**: 平滑的学习率衰减
- **Warmup**: 前5轮逐步提高学习率

### 损失函数
- **组合损失**: MSE + 0.5 × MAE
  - MSE关注大误差
  - MAE关注整体准确度

### 正则化
- **Dropout**: 0.1
- **梯度裁剪**: norm ≤ 1.0
- **早停**: patience=20

---

## 📁 文件结构

```
models/
├── tds_network.py          # TDS卷积网络实现
├── lstm_decoder.py         # LSTM解码器
├── tds_lstm_model.py       # TDS-LSTM组合模型
├── model_factory.py        # 模型工厂
└── __init__.py

configs/
├── default.yaml            # 标准配置（推荐）
├── quick_demo.yaml         # 快速验证
└── high_performance.yaml   # 高性能配置

trainers/
└── trainer.py              # 训练器（支持组合损失、warmup等）

main.py                     # 主训练入口
config.py                   # 配置加载器
```

---

## 🎯 关键参数说明

### TDS Stage 配置

每个stage包含以下参数：

```yaml
- in_conv_kernel: 5      # 初始卷积核大小
  in_conv_stride: 2      # 降采样步长
  num_blocks: 2          # TDS块数量
  channels: 8            # 分组数
  feature_width: 4       # 每组宽度
  kernel_width: 21       # 时间卷积核大小
```

- `总特征数 = channels × feature_width`
- `降采样倍数 = 所有stage的stride乘积`

### LSTM解码器配置

```yaml
decoder_type: "sequential_lstm"    # 解码器类型
decoder_hidden_size: 128           # 隐藏层大小
decoder_num_layers: 2              # 层数
decoder_state_condition: true      # 使用状态条件
```

---

## ⚠️ 注意事项

### 数据集路径

确保配置文件中的数据集路径正确：

```yaml
data:
  dataset_path: "D:/Dataset/emg2pose_dataset_mini"
```

### GPU内存

- `quick_demo`: ~2GB
- `default`: ~4GB
- `high_performance`: ~8GB

如果内存不足，可以降低 `batch_size`。

### CPU训练

如果没有GPU，在配置文件中设置：

```yaml
training:
  device: "cpu"
```

---

## 📈 预期性能

基于标准TDS网络架构，预期指标：

- **MAE**: < 0.15 弧度（约 8.6度）
- **MSE**: < 0.05
- **R²**: > 0.85

实际性能取决于数据质量和训练时间。

---

## 🔍 训练输出

训练过程会生成以下文件：

```
logs/
└── default_20251015_123456/
    ├── training_20251015_123456.log    # 训练日志
    ├── config_20251015_123456.yaml     # 配置快照
    └── experiment_report_*.md          # 实验报告

checkpoints/
└── best_model_*.pth                    # 最佳模型检查点
```

---

## 💡 使用技巧

### 1. 首次运行

建议先用 `quick_demo` 快速验证：

```python
SELECTED_CONFIG = 'quick_demo'
```

### 2. 正式训练

确认无误后，使用 `default` 配置：

```python
SELECTED_CONFIG = 'default'
```

### 3. 查看训练日志

实时查看日志：

```bash
tail -f logs/default_*/training_*.log
```

### 4. 使用TensorBoard

```bash
tensorboard --logdir=logs
```

在浏览器打开 http://localhost:6006

---

## ❓ 常见问题

### Q1: 服务器上出现 "Killed" 错误怎么办？🔴

**A**: 这是内存不足（OOM）导致的，使用 `low_memory` 配置：

```python
# main.py 第37行
SELECTED_CONFIG = 'low_memory'
```

**原理**：
- 小batch size (16) 减少内存占用
- 梯度累积 (×4) 保持训练效果
- 等效于 batch_size=64

**详细解决方案**: 查看 `MEMORY_OPTIMIZATION.md`

### Q2: 本地GPU内存不足怎么办？

**A**: 逐步降低配置：
1. 使用 `low_memory` 配置
2. 进一步降低：`batch_size: 8`, `gradient_accumulation_steps: 8`
3. 减少工作进程：`num_workers: 0`

### Q3: 训练很慢？

**A**: 使用GPU，或先用 `quick_demo` 验证，再用 `default` 训练。

### Q4: 如何调整模型大小？

**A**: 修改配置文件中的 `tds_stages` 和 `decoder_hidden_size`。

### Q5: 如何恢复训练？

**A**: 目前暂不支持断点续训，建议增加 `patience` 值避免过早停止。

---

## 📞 技术支持

如遇问题，请检查：

1. 数据集路径是否正确
2. PyTorch是否正确安装
3. 日志文件中的错误信息
4. GPU驱动是否正常（如使用GPU）

---

**现在可以直接运行 `python main.py` 开始训练！** 🎉

