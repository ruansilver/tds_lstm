# 对齐实施总结

## 概述

本次实施按照方案2（中等改动对齐）完成了以下主要工作，将我们的代码库与论文架构对齐。

---

## 已完成的修改

### 1. 数据加载器重写 ✅

**文件**: `data/dataset.py`, `data/dataloader.py`

**主要变更**:
- ✅ 添加IK失败检测和过滤功能
  - `get_ik_failures_mask()`: 检测全零的joint_angles
  - `get_contiguous_ones()`: 找到连续有效数据块
  - `SessionData.no_ik_failure`: 缓存IK失败mask

- ✅ 支持padding功能
  - `padding=(left_padding, right_padding)`: 配置上下文padding
  - 窗口读取时自动扩展padding区域

- ✅ 支持jitter数据增强
  - 训练时随机抖动窗口位置
  - 验证/测试时禁用jitter

- ✅ 返回字典格式
  ```python
  {
      'emg': (C, T),           # 转置为BCT格式
      'joint_angles': (C, T),   
      'no_ik_failure': (T,),    # IK失败mask
      'window_start_idx': int,
      'window_end_idx': int
  }
  ```

- ✅ skip_ik_failures支持
  - 预处理时自动跳过IK失败的窗口
  - 在有效数据块内创建窗口索引

**数据格式**: 统一使用字典格式，代码简洁无冗余

---

### 2. PoseModule包装层 ✅

**文件**: `models/pose_modules.py`

**实现的类**:

#### BasePoseModule
- 处理left_context和right_context
- 时间对齐：`align_predictions()` 和 `align_mask()`
- 提取initial_pos
- 字典输入：`{'emg': BCT, 'joint_angles': BCT, 'no_ik_failure': BT}`
- 输出对齐的`(pred, target, mask)`元组

#### PoseModule
- 简单姿态模块
- 支持位置/速度预测模式
- 直接通过网络生成预测

#### StatePoseModule
- 状态姿态模块（与论文完全对齐）
- 逐时间步LSTM解码
- 特征重采样到rollout_freq
- state_condition支持
- predict_velocity支持

**使用方式**:
```python
# 包装现有网络
from models import StatePoseModule
pose_module = StatePoseModule(
    network=encoder,
    decoder=decoder,
    state_condition=True,
    predict_vel=False,
    rollout_freq=50
)

# 训练时
pred, target, mask = pose_module(batch, provide_initial_pos=False)
```

---

### 3. 训练器增强 ✅

**文件**: `trainers/trainer.py`

**主要变更**:
- ✅ 支持字典输入格式
  - 自动检测输入类型（dict vs tuple）
  - 兼容旧格式

- ✅ Mask过滤损失计算
  ```python
  if no_ik_failure is not None:
      mask = no_ik_failure.unsqueeze(1).expand_as(predictions)
      loss = criterion(predictions[mask], angle_data[mask])
  ```

- ✅ Mask过滤metrics计算
  - 只在有效样本上计算MSE/MAE
  - 避免IK失败样本影响metrics

- ✅ use_pose_module开关
  - `use_pose_module=True`: 使用PoseModule包装
  - `use_pose_module=False`: 直接使用模型（向后兼容）

- ✅ provide_initial_pos开关
  - 控制是否向模型提供初始位置

**在三个方法中统一实现**:
- `train_epoch()`
- `validate_epoch()`
- `evaluate()`

---

### 4. Metrics系统增强 ✅

**文件**: `utils/metrics.py`

**新增函数**:

#### 角度导数指标
- `calculate_angular_derivatives()`: 计算角速度、角加速度、jerk
- 单位转换：从(radians/sample)到(radians/second)
- 平均绝对值统计

#### 对齐的MAE计算
- `calculate_angle_mae()`: 支持mask的MAE
- `calculate_per_finger_mae()`: 每个手指的MAE
  - Thumb, Index, Middle, Ring, Pinky
- `calculate_pd_groups_mae()`: 近端/中端/远端关节组MAE
  - Proximal, Mid, Distal

#### 综合指标
- `calculate_comprehensive_metrics()`: 一站式计算所有指标
  - 基本MAE
  - 导数指标
  - 每个手指MAE
  - PD组MAE

**使用方式**:
```python
metrics = calculate_comprehensive_metrics(
    pred=predictions,
    target=targets,
    mask=no_ik_failure,
    sample_rate=2000.0,
    include_derivatives=True,
    include_per_finger=True,
    include_pd_groups=True
)
```

---

### 5. 对齐配置文件 ✅

**文件**: `configs/emg2pose_mimic.yaml`

**关键配置对齐**:

#### 数据配置
- `window_size: 10000` (5秒 @ 2000Hz)
- `padding: [0, 0]`
- `skip_ik_failures: false`
- `normalize: false`

#### 模型配置
- TDS两stage设计
  - Stage1: stride=2, 32 features
  - Stage2: stride=2, 64 features
- LSTM: hidden=128, layers=2
- `rollout_freq: 50`
- `state_condition: true`
- `predict_velocity: false`

#### 训练配置
- `num_epochs: 100`
- `batch_size: 32`
- `learning_rate: 0.001`
- `optimizer: adam`
- `scheduler: none`
- `loss_type: mae`
- `weight_decay: 0.0`

#### 新增开关
- `use_pose_module: false` (可切换)
- `provide_initial_pos: false` (可切换)

---

## 代码设计原则

### 简洁优先
1. **数据格式**: 统一使用字典格式，无向后兼容代码
2. **模型接口**: 灵活选择直接使用TDSLSTMModel或PoseModule包装
3. **配置参数**: 新参数都有默认值，易于使用

### 功能开关
可以通过配置灵活控制：
1. **use_pose_module**: 是否使用PoseModule包装（默认False）
2. **provide_initial_pos**: 是否提供初始位置（默认False）
3. **skip_ik_failures**: 是否跳过IK失败样本（默认False）
4. **padding**: 配置context padding（默认[0, 0]）

---

## 使用示例

### 完全对齐模式

```python
# 1. 使用对齐配置
config = get_config_by_name('emg2pose_mimic')

# 2. 数据加载（自动使用新格式）
train_loader, val_loader, test_loader = create_dataloaders(config)

# 3. 创建模型（可选：包装为PoseModule）
from models import create_model, StatePoseModule
encoder = create_model('tds_lstm', config)

# 如果使用PoseModule:
# model = StatePoseModule(
#     network=encoder.encoder,
#     decoder=encoder.decoder,
#     state_condition=True,
#     predict_vel=False,
#     rollout_freq=50
# )
# config.training.use_pose_module = True

# 4. 训练（自动支持mask过滤）
trainer = Trainer(model, config, train_loader, val_loader, test_loader)
results = trainer.train()
```

### 标准模式（推荐）

```python
# 使用对齐配置
config = get_config_by_name('emg2pose_mimic')

# 数据加载（返回字典格式）
train_loader, val_loader, test_loader = create_dataloaders(config)

# 使用TDSLSTMModel
model = create_model('tds_lstm', config)

# 训练（自动支持mask过滤）
trainer = Trainer(model, config, train_loader, val_loader, test_loader)
results = trainer.train()
```

---

## 关键差异总结

### 已实现的对齐 ✅
1. ✅ 数据加载：padding, jitter, IK mask, 字典格式, BCT转置
2. ✅ 模型包装：BasePoseModule, StatePoseModule, context处理
3. ✅ 训练流程：mask过滤损失和metrics
4. ✅ Metrics系统：角度导数, per-finger MAE, PD组MAE
5. ✅ 配置对齐：超参数, 损失函数, 优化器设置

### 未实现的高级功能（可选）
1. ⚠️ PyTorch Lightning框架（保留自定义训练循环）
2. ⚠️ Hydra配置系统（保留简单YAML）
3. ⚠️ Forward Kinematics相关metrics（需要hand model）
4. ⚠️ Transform pipeline（在dataset内部处理）

---

## 下一步建议

### 测试验证
1. 测试数据加载：检查dict格式输出
2. 测试mask过滤：确保损失计算正确
3. 测试PoseModule：验证时间对齐
4. 对比实验：使用emg2pose_mimic.yaml运行完整训练

### 可选增强
1. 实现PoseModule的模型工厂函数
2. 添加更多metrics（如landmark距离）
3. 支持不同的解码器类型切换
4. 添加可视化工具展示对齐效果

---

## 文件修改清单

### 新增文件
- `models/pose_modules.py` (327行)
- `configs/emg2pose_mimic.yaml` (106行)
- `IMPLEMENTATION_SUMMARY.md` (本文件)

### 修改文件
- `data/dataset.py` (~180行修改)
- `data/dataloader.py` (~30行修改)
- `trainers/trainer.py` (~200行修改)
- `utils/metrics.py` (~230行新增)
- `models/__init__.py` (导出更新)

### 总代码量
- 新增: ~900行
- 修改: ~400行
- 总计: ~1300行

---

## 注意事项

1. **数据格式**: 数据加载器统一返回字典格式，包含BCT维度的tensor
2. **Mask处理**: 所有损失和metrics计算都自动应用mask过滤IK失败样本
3. **Context对齐**: PoseModule自动处理，直接使用TDSLSTMModel时无需考虑
4. **内存优化**: 保留了原有的内存优化特性（在线metrics、梯度累积等）
5. **代码简洁**: 移除了所有向后兼容代码，保持代码整洁

---

## 结论

本次实施成功完成了方案2的所有核心功能，实现了与论文架构的较完整对齐。代码采用简洁设计原则，移除了所有向后兼容代码，保持整洁易读。可以根据实验需要通过配置文件灵活启用各项新功能。

**核心优势**:
- ✅ 完整对齐数据格式和训练流程
- ✅ 自动mask过滤IK失败样本
- ✅ 丰富的metrics系统
- ✅ 代码简洁，无冗余
- ✅ 灵活的配置开关

