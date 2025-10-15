# 快速开始指南

## 🚀 30秒开始训练

### 步骤1: 选择配置

打开 `main.py`，修改第37行：

```python
# 根据你的情况选择：
SELECTED_CONFIG = 'low_memory'     # 👈 服务器/内存受限（推荐）
# SELECTED_CONFIG = 'default'      # 本地GPU（8GB显存）
# SELECTED_CONFIG = 'quick_demo'   # 快速测试（10轮）
```

### 步骤2: 检查数据路径

确认数据集路径正确（已在配置文件中设置）：
```
D:/Dataset/emg2pose_dataset_mini
```

### 步骤3: 运行训练

```bash
python main.py
```

完成！🎉

---

## 💡 配置选择建议

### 你在服务器上训练？ → `low_memory`
- ✅ 内存占用最低（~1GB）
- ✅ 使用梯度累积保持效果
- ✅ 不会被Killed

### 你在本地GPU上训练？ → `default`
- ✅ 平衡性能和速度
- ✅ 无需特殊优化
- ✅ 2-3小时完成100轮

### 你只是想快速测试？ → `quick_demo`
- ✅ 10轮训练，10分钟完成
- ✅ 验证代码能跑通
- ✅ 然后再用其他配置正式训练

### 你有高端GPU且追求极致？ → `high_performance`
- ⚠️ 需要8GB+显存
- ✅ 最佳性能
- ⏱️ 5小时，200轮

---

## ⚠️ 遇到问题？

### 问题：服务器上被Killed
**解决**: 使用 `low_memory` 配置

### 问题：GPU内存不足
**解决**: 
1. 使用 `low_memory`
2. 或手动降低 `batch_size`

### 问题：找不到数据集
**解决**: 修改配置文件中的 `dataset_path`

详细问题排查：查看 `README_TDS_LSTM.md` 或 `MEMORY_OPTIMIZATION.md`

---

## 📊 训练过程

训练时会显示：
```
Epoch: 0, Batch: 100/11014, Loss: 0.152341
Epoch: 0, Batch: 200/11014, Loss: 0.138052
...
```

训练完成后会生成：
```
logs/
  └── default_20251015_101510/
      ├── training_*.log          # 训练日志
      ├── config_*.yaml           # 配置快照
      └── experiment_report_*.md  # 实验报告

checkpoints/
  └── best_model_*.pth            # 最佳模型
```

---

## 🎯 下一步

- 查看训练日志：`logs/*/training_*.log`
- 查看TensorBoard：`tensorboard --logdir=logs`
- 详细文档：`README_TDS_LSTM.md`
- 内存优化：`MEMORY_OPTIMIZATION.md`

**祝训练顺利！** 🎉

