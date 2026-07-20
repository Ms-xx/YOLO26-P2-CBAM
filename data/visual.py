from pathlib import Path

import cv2
import numpy as np


def visualize_yolo_labels(image_dir, label_dir, output_dir="visualized", class_names=None):
    image_path = Path(image_dir)
    label_path = Path(label_dir)
    output_path = Path(output_dir)  # 缺少这行！

    # 删除重复的mkdir，只保留一行
    output_path.mkdir(parents=True, exist_ok=True)

    # 为每个类别生成随机颜色
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(80, 3), dtype=np.uint8).tolist()

    # 支持的图片格式
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    for img_file in image_path.iterdir():
        if img_file.suffix.lower() not in image_exts:
            continue

        # 读取图片
        img = cv2.imread(str(img_file))
        if img is None:
            print(f"无法读取图片: {img_file.name}")
            continue

        h, w = img.shape[:2]

        # 查找对应的标签文件
        label_file = label_path / f"{img_file.stem}.txt"
        if not label_file.exists():
            print(f"未找到标签: {label_file.name}")
            # 仍保存原图
            cv2.imwrite(str(output_path / img_file.name), img)
            continue

        # 读取并绘制标注
        with open(label_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue

                class_id, x_center, y_center, width, height = map(float, parts)
                class_id = int(class_id)

                # YOLO格式转像素坐标
                x1 = int((x_center - width / 2) * w)
                y1 = int((y_center - height / 2) * h)
                x2 = int((x_center + width / 2) * w)
                y2 = int((y_center + height / 2) * h)

                # 边界检查
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                # 获取颜色和类别名
                color = colors[class_id % len(colors)]
                label = class_names[class_id] if class_names and class_id < len(class_names) else f"Class {class_id}"

                # 绘制矩形框
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

                # 绘制标签背景
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(img, (x1, y1 - text_size[1] - 10), (x1 + text_size[0], y1), color, -1)

                # 绘制标签文字
                cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 保存结果
        output_file = output_path / img_file.name
        cv2.imwrite(str(output_file), img)
        print(f"已保存: {output_file.name}")


# 使用示例
if __name__ == "__main__":
    # 配置路径（建议使用绝对路径或检查当前工作目录）
    IMAGE_DIR = "###/data/task_all/image"
    LABEL_DIR = "###/data/task_all/label"
    OUTPUT_DIR = "###/data/task_all/visualized"

    CLASS_NAMES = [
        "Unclassified foreign objects",
        "Rockfall gauge infringement",
        "Personnel Intrusion",
        "Animal Intrusion",
        "Train Passing",
        "Background-Only Scene",
    ]

    visualize_yolo_labels(IMAGE_DIR, LABEL_DIR, OUTPUT_DIR, CLASS_NAMES)
