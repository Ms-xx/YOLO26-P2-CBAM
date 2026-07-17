import os
import sys

# 将当前目录添加到模块搜索路径的最前面
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# 现在会优先从当前目录导入 ultralytics（如果存在）
from ultralytics import YOLO

if __name__ == "__main__":
    pth_path = r"###/runs/detect/stage1_freeze1/weights/best.pt"

    test_path = r"###/data/CeShi_2"
    # Load a model
    # model = YOLO('yolov8n.pt')  # load an official model
    model = YOLO(pth_path)  # load a custom model

    # Predict with the model
    results = model(test_path, save=True, conf=0.5)  # predict on an image
