import shutil
from pathlib import Path


def organize_files(source_dir, image_dir="image", label_dir="label"):
    source_path = Path(source_dir)

    # 创建目标文件夹
    img_path = source_path / image_dir
    lbl_path = source_path / label_dir
    img_path.mkdir(exist_ok=True)
    lbl_path.mkdir(exist_ok=True)

    # 支持的图片格式
    image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff", ".webp"}
    label_exts = {".txt"}

    moved_images = 0
    moved_labels = 0

    # 遍历源文件夹
    for file in source_path.iterdir():
        if file.is_file():
            ext = file.suffix.lower()

            if ext in image_exts:
                # 移动图片
                shutil.move(str(file), str(img_path / file.name))
                moved_images += 1
                print(f"图片: {file.name} -> {image_dir}/")

            elif ext in label_exts:
                # 移动标签
                shutil.move(str(file), str(lbl_path / file.name))
                moved_labels += 1
                print(f"标签: {file.name} -> {label_dir}/")

    print(f"\n完成！移动 {moved_images} 个图片文件，{moved_labels} 个标签文件")


# 使用示例
if __name__ == "__main__":
    source_folder = r"###/data/newShuJu"

    organize_files(source_folder)
