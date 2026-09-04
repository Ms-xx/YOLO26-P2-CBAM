# YOLO26-P2-CBAM

基于 YOLO26 的轻量化学习行为监测模型，用于在线教育场景下的**打哈欠（疲劳）检测**与**人眼距屏幕距离估计**。

## 论文

**Learning Behavior Monitoring Through Drowsiness and Screen-Distance Recognition**

## 改进点

在 YOLO26 基线基础上引入了三项改进：

1. **P2 检测头**：新增高分辨率检测分支（约 1/4 下采样），构建 P2–P5 四尺度检测结构，增强小目标感知能力。
2. **CBAM 注意力机制**：在 backbone 每个 C3k2 模块后插入 CBAM（通道注意力 + 空间注意力），共 4 个模块，抑制背景干扰、强化行为相关特征。
3. **WIoU 损失**：以 Wise-IoU（WIoU）替换 CIoU 作为边界框回归损失，自适应分配样本权重，提升定位精度与训练稳定性。

## 模型配置

- 模型文件：`ultralytics/cfg/models/26/yolo26-p2-CBAM.yaml`
- 检测尺度：P2 / P3 / P4 / P5（4 个检测头）
- 类别数：`nc = 4`

## 实验结果

| 数据集 | 指标 | 结果 |
| --- | --- | --- |
| SCB-dataset（公开） | mAP@0.5 | 63.2%（较基线 +4.2%） |
| SCB-dataset（公开） | Precision / Recall / F1 | 62.2% / 59.6% / 60.9% |
| 自建打哈欠数据集 | mAP@0.5 | 86.8% |
| 屏幕距离估计 | MAE | 2.12 cm |
| 推理速度 | 单帧 | 约 12.5 ms |

## 环境

- Python 3.8+
- PyTorch 1.12+
- CUDA 12.2（GPU 加速推荐，CPU 单帧约 180 ms）

## 快速开始

```bash
# 训练（两阶段：冻结 backbone → 全模型微调）
python train.py

# 推理
python test.py
```

> 训练 / 测试脚本中的路径需按实际数据与权重位置修改。