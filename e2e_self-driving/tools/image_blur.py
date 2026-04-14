#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@文件        :augment_gaussian_blur.py
@说明        :对指定文件夹中所有图像进行高斯模糊增强，生成新的图像，并保持文件名中转向角数据不变。
         :每张图像生成5张模糊图，模糊核尺寸均匀覆盖设定范围。
@作者        :your_name
@日期        :2025-XX-XX
"""

import os
import cv2
import numpy as np

# 指定原始数据文件夹和目标文件夹
src_folder = "data/raw"        # 原始数据所在文件夹
dst_folder = "data/aug2"     # 增强后数据保存到此文件夹

# 如果目标文件夹不存在，则创建
if not os.path.exists(dst_folder):
    os.makedirs(dst_folder)

# 设置高斯模糊核尺寸（必须为奇数），示例：3x3 到 11x11
KERNEL_SIZES = [3, 5, 7, 9, 11]

# 获取源文件夹中所有的jpg文件
file_list = sorted([f for f in os.listdir(src_folder) if f.endswith('.jpg')])

if not file_list:
    print("没有找到jpg文件！")
    exit()

print(f"共找到 {len(file_list)} 张图片，开始进行高斯模糊增强...")

new_index = 0  # 新的索引起始值

for fname in file_list:
    # 文件名格式："{index}_{steering_angle}.jpg"
    parts = fname.split('_')
    if len(parts) < 2:
        print(f"文件名格式不正确，跳过: {fname}")
        continue
    # 保留原始标签不变
    steering_angle_str = parts[1].replace('.jpg', '')

    src_path = os.path.join(src_folder, fname)
    img = cv2.imread(src_path)
    if img is None:
        print(f"无法读取图像: {src_path}")
        continue

    # 统一调整为训练/推理时的尺寸 160×120
    img_resized = cv2.resize(img, (160, 120))

    # 针对每张图片生成5个模糊图像
    for k in KERNEL_SIZES:
        # 应用高斯模糊：kernel 大小 k x k，sigmaX=0（自动计算）
        img_blur = cv2.GaussianBlur(img_resized, (k, k), sigmaX=0)
        new_fname = f"{new_index}_{steering_angle_str}.jpg"
        dst_path = os.path.join(dst_folder, new_fname)
        cv2.imwrite(dst_path, img_blur)
        print(f"{new_fname} --> Gaussian kernel = {k}x{k}")
        new_index += 1

print("高斯模糊数据增强完成。")
