import os
import shutil
import random
from pathlib import Path
from collections import defaultdict

def split_dataset(image_dir, label_dir, output_dir='dataset', train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "比例之和必须等于1"
    
    random.seed(seed)
    
    # 创建输出目录结构
    splits = ['train', 'val', 'test']
    for split in splits:
        (Path(output_dir) / 'images' / split).mkdir(parents=True, exist_ok=True)
        (Path(output_dir) / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    # 获取所有图片文件
    image_path = Path(image_dir)
    label_path = Path(label_dir)
    
    image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.gif', '.tiff'}
    images = [f for f in image_path.iterdir() if f.suffix.lower() in image_exts]
    
    # 筛选有对应标签的图片
    valid_pairs = []
    for img in images:
        label_file = label_path / f"{img.stem}.txt"
        if label_file.exists():
            valid_pairs.append((img, label_file))
        else:
            print(f"警告: 未找到标签文件 {img.stem}.txt")
    
    if len(valid_pairs) == 0:
        print("错误: 未找到有效的图片-标签对!")
        return
    
    print(f"\n找到 {len(valid_pairs)} 对有效数据")
    
    # 随机打乱
    random.shuffle(valid_pairs)
    
    # 计算划分数量
    total = len(valid_pairs)
    train_num = int(total * train_ratio)
    val_num = int(total * val_ratio)
    test_num = total - train_num - val_num  # 避免舍入误差
    
    # 划分数据集
    train_data = valid_pairs[:train_num]
    val_data = valid_pairs[train_num:train_num + val_num]
    test_data = valid_pairs[train_num + val_num:]
    
    split_dict = {
        'train': train_data,
        'val': val_data,
        'test': test_data
    }
    
    # 复制文件到对应目录
    for split_name, data in split_dict.items():
        print(f"\n处理 {split_name} 集: {len(data)} 个样本")
        
        for img_file, label_file in data:
            # 复制图片
            dst_img = Path(output_dir) / 'images' / split_name / img_file.name
            shutil.copy2(img_file, dst_img)
            
            # 复制标签
            dst_label = Path(output_dir) / 'labels' / split_name / label_file.name
            shutil.copy2(label_file, dst_label)
    
    # 输出统计信息
    print(f"\n{'='*50}")
    print("数据集划分完成:")
    print(f"  训练集 (train): {len(train_data)} 张 ({len(train_data)/total*100:.1f}%)")
    print(f"  验证集 (val):   {len(val_data)} 张 ({len(val_data)/total*100:.1f}%)")
    print(f"  测试集 (test):  {len(test_data)} 张 ({len(test_data)/total*100:.1f}%)")
    print(f"  总计: {total} 张")
    print(f"{'='*50}")
    
    # 生成YOLO的data.yaml配置文件
    nc = count_classes(label_path)
    yaml_content = f"""# YOLO Dataset Configuration
path: {Path(output_dir).absolute()}  # dataset root dir
train: images/train
val: images/val
test: images/test

# Classes
nc: {nc}  # number of classes
names: []  # class names, e.g., ['person', 'car', 'dog']
"""
    
    yaml_path = Path(output_dir) / 'data.yaml'
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    print(f"\n已生成配置文件: {yaml_path}")

def count_classes(label_dir):
    """统计类别数量"""
    classes = set()
    label_path = Path(label_dir)
    
    for txt_file in label_path.glob('*.txt'):
        try:
            with open(txt_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        classes.add(int(parts[0]))
        except Exception as e:
            continue
    
    return len(classes) if classes else 0


# 使用示例
if __name__ == "__main__":
    # 配置路径（根据你的实际路径修改）
    IMAGE_DIR = "###/data/newShuJu/image"     # 图片文件夹
    LABEL_DIR = "###/data/newShuJu/label"      # YOLO标签文件夹
    OUTPUT_DIR = "###/dataset"           # 输出文件夹
    
    # 执行划分（8:1:1）
    split_dataset(
        image_dir=IMAGE_DIR,
        label_dir=LABEL_DIR,
        output_dir=OUTPUT_DIR,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        seed=42  # 随机种子，保证可复现
    )