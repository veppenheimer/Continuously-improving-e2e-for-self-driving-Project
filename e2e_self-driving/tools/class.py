import os
from pathlib import Path
from collections import Counter


def analyze_steering_angles(folder_path):
    # 检查文件夹是否存在
    target_dir = Path(folder_path)
    if not target_dir.exists():
        print(f"错误: 找不到路径 '{folder_path}'")
        return

    # 支持常见的图像后缀
    extensions = {'.jpg', '.jpeg', '.png', '.bmp'}

    angles = []
    error_count = 0

    print(f"正在扫描目录: {target_dir.absolute()} ...")

    # 遍历文件夹
    for file in target_dir.iterdir():
        if file.is_file() and file.suffix.lower() in extensions:
            try:
                # 文件名格式: 序号_转向角值.jpg
                # 1. 去掉后缀 -> 序号_转向角值
                # 2. 按 '_' 分割取最后一个元素 -> 转向角值
                angle_str = file.stem.split('_')[-1]

                # 转换为浮点数或整数以统一格式（防止 0.0 和 0 被视为不同类别）
                angle = float(angle_str)
                angles.append(angle)
            except (ValueError, IndexError):
                print(f"跳过格式不正确的文件: {file.name}")
                error_count += 1

    # 使用 Counter 统计频率
    counts = Counter(angles)

    # 打印结果
    print("\n--- 统计结果 ---")
    print(f"成功处理图像总数: {len(angles)}")
    print(f"不同角度类别总数: {len(counts)}")
    print(f"解析失败文件数: {error_count}")
    print("-" * 20)

    # 按角度排序并打印
    print(f"{'转向角':<15} | {'频次':<10}")
    print("-" * 28)
    for angle in sorted(counts.keys()):
        print(f"{angle:<15} | {counts[angle]:<10}")


if __name__ == "__main__":
    # 在这里输入你的图像文件夹路径
    # 如果脚本和图像在同一个文件夹，可以使用 '.'
    path_to_images = "E:\桌面\项目\e2e_self-driving\dataset\data"
    analyze_steering_angles(path_to_images)