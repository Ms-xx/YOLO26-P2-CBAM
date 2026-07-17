import os
import sys

# 解决 MKL 冲突（保留）
os.environ["MKL_THREADING_LAYER"] = "GNU"

# 将当前目录添加到模块搜索路径最前面
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# 现在会优先导入当前目录下的 ultralytics（如果存在）
import torch

import ultralytics
from ultralytics import YOLO

print(f"ultralytics 路径: {ultralytics.__file__}")
print(f"ultralytics 版本: {ultralytics.__version__}")

# 测试 CUDA
print(f"CUDA 可用: {torch.cuda.is_available()}")
print(f"GPU 数量: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"当前设备: GPU {torch.cuda.current_device()}")


# ========== 阶段1: 冻结Backbone训练 ==========
model = YOLO("###/cfg/models/v8/yolo26-p2-CBAM.yaml")
model.load("###/yolov26x.pt")

# 手动冻结前10层
for i, (name, param) in enumerate(model.model.named_parameters()):
    if i < 10:  # 前10层参数禁用梯度更新（冻结）
        param.requires_grad = False
        print(f"冻结层 {i}: {name}")  # 可选：打印冻结的层名，便于验证
    else:  # 10层之后的参数允许梯度更新（训练）
        param.requires_grad = True


print(f"可训练参数: {sum(p.numel() for p in model.model.parameters() if p.requires_grad):,}")

# 使用 GPU 3，用字符串格式
model.train(
    data="###/ultralytics/cfg/datasets/data.yaml",
    epochs=150,
    lr0=0.001,
    lrf=0.01,
    name="stage1_freeze1",
    device="cuda:0,1,2,3",  # 因为 CUDA_VISIBLE_DEVICES='3'，所以这里 'cuda:0' 实际是 GPU 3
    batch=32,
    imgsz=512,
    workers=8,
    cache=True,
    # patience=20,
)

# ========== 阶段2: 解冻微调 ==========
model = YOLO("###/runs/detect/stage1_freeze1/weights/best.pt")

for param in model.model.parameters():
    param.requires_grad = True

model.train(
    data="###/ultralytics/cfg/datasets/data.yaml",
    epochs=200,
    lr0=0.0001,
    lrf=0.01,
    name="stage1_finetune1",
    device="cuda:0,1,2,3",  # 同样使用 GPU 3
    batch=32,
    imgsz=512,
    workers=8,
    cache=True,
    # patience=30,
)
