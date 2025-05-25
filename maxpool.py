import os
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np


def process_image(input_path, output_path, pool_size=2):
    """
    处理单张图片：
    1. 转为灰度图
    2. 调整大小为 224x224
    3. 最大池化
    4. 保存
    """
    try:
        # 1. 打开图片并转为灰度
        img = Image.open(input_path).convert('L')  # 'L' 表示灰度
        img = img.resize((224, 224))

        # 2. 转为张量（0-1 范围）
        img_array = np.array(img) / 255.0
        img_tensor = torch.tensor(img_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

        # 3. 最大池化
        pooled_tensor = F.max_pool2d(img_tensor, kernel_size=pool_size).squeeze()

        # 4. 转回 PIL 图像（0-255 范围）
        pooled_img_array = (pooled_tensor.numpy() * 255).clip(0, 255).astype(np.uint8)
        img_pil = Image.fromarray(pooled_img_array.squeeze())

        # 5. 保存
        img_pil.save(output_path)
        print(f"Save at {output_path}")
    except Exception as e:
        print(f"Error processing {input_path}: {e}")


def process_dataset(dataset_dir, output_dir, pool_size=2):
    """
    遍历数据集目录，处理所有图片并保持目录结构
    """
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                # 构建输入路径
                input_path = os.path.join(root, file)

                # 构建相对路径（用于保持目录结构）
                rel_path = os.path.relpath(input_path, dataset_dir)

                # 构建输出路径
                output_path = os.path.join(output_dir, rel_path)

                # 确保输出目录存在
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                # 处理图片
                process_image(input_path, output_path, pool_size)


if __name__ == '__main__':
    # 使用示例
    dataset_dir = "data/CT"  # 输入数据集目录
    output_dir = "data/MaxPool4/CT"  # 输出目录（可以修改为任意路径）

    # 处理数据集
    process_dataset(dataset_dir, output_dir, pool_size=4)
