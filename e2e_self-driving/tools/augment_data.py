#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@文件        :augment_data.py
@说明        :自动驾驶数据集增强脚本
@版本        :1.0
'''

# ==================== 配置参数 ====================
INPUT_DIR = 'data/raw'              # 原始数据目录
OUTPUT_DIR = 'data/aug1'  # 增强后数据保存目录

AUGMENT_FACTOR = 30                          # 每张图像生成的增强图像数量
COMBINE_METHODS = (2, 6)                     # 每次增强方法组合数量范围
RANDOM_SEED = 42

ENABLED_METHODS = [1,2,3,4,5,6,7,8]

BRIGHTNESS_RANGE = (0.7, 1.3)
CONTRAST_RANGE = (0.8, 1.2)
BLUR_KERNEL_SIZES = [3, 5]
NOISE_STRENGTH_RANGE = (5, 20)
SHIFT_RANGE = (-0.1, 0.1)
ROTATION_ANGLE_RANGE = (-5, 5)
SHADOW_DARKNESS_RANGE = (0.6, 0.9)

SHIFT_ANGLE_FACTOR = 0.5
ROTATION_ANGLE_FACTOR = 0.3

METHOD_DESCRIPTIONS = {
    1: "亮度调整",
    2: "对比度调整",
    3: "模糊",
    4: "噪声",
    5: "水平翻转",
    6: "水平平移",
    7: "轻微旋转",
    8: "阴影"
}

# ==================== 导入库 ====================
import os
import cv2
import numpy as np
import argparse
import random
import re
from tqdm import tqdm

class DataAugmenter:
    def __init__(self, input_dir=INPUT_DIR, output_dir=OUTPUT_DIR,
                 augment_factor=AUGMENT_FACTOR, enabled_methods=ENABLED_METHODS):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.augment_factor = augment_factor
        self.enabled_methods = enabled_methods

        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)

        os.makedirs(output_dir, exist_ok=True)

        self.all_methods = {
            1: {"name": "brightness", "func": self.adjust_brightness},
            2: {"name": "contrast", "func": self.adjust_contrast},
            3: {"name": "blur", "func": self.add_blur},
            4: {"name": "noise", "func": self.add_noise},
            5: {"name": "horizontal_flip", "func": self.horizontal_flip},
            6: {"name": "shift", "func": self.horizontal_shift},
            7: {"name": "rotation", "func": self.slight_rotation},
            8: {"name": "shadow", "func": self.add_shadow}
        }

        self.augment_methods = {
            self.all_methods[m]["name"]: self.all_methods[m]["func"]
            for m in enabled_methods if m in self.all_methods
        }

        print("已启用的增强方法:")
        for m in enabled_methods:
            if m in METHOD_DESCRIPTIONS:
                print(f"  {m}: {METHOD_DESCRIPTIONS[m]}")

    # ---------------------- 增强方法 ----------------------
    def adjust_brightness(self, image, angular_vel):
        factor = random.uniform(*BRIGHTNESS_RANGE)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 2] *= factor
        hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR), angular_vel

    def adjust_contrast(self, image, angular_vel):
        factor = random.uniform(*CONTRAST_RANGE)
        mean = np.mean(image, axis=(0, 1), keepdims=True)
        enhanced = image * factor + mean * (1 - factor)
        return np.clip(enhanced, 0, 255).astype(np.uint8), angular_vel

    def add_blur(self, image, angular_vel):
        k = random.choice(BLUR_KERNEL_SIZES)
        return cv2.GaussianBlur(image, (k, k), 0), angular_vel

    def add_noise(self, image, angular_vel):
        strength = random.uniform(*NOISE_STRENGTH_RANGE)
        noise = np.random.randn(*image.shape) * strength
        noisy_image = image + noise
        return np.clip(noisy_image, 0, 255).astype(np.uint8), angular_vel

    def horizontal_flip(self, image, angular_vel):
        return cv2.flip(image, 1), -angular_vel

    def horizontal_shift(self, image, angular_vel):
        shift = random.uniform(*SHIFT_RANGE)
        height, width = image.shape[:2]
        pixels = int(width * shift)
        M = np.float32([[1, 0, pixels], [0, 1, 0]])
        shifted = cv2.warpAffine(image, M, (width, height), borderMode=cv2.BORDER_REPLICATE)
        adjusted_angular = angular_vel - shift * SHIFT_ANGLE_FACTOR
        return shifted, adjusted_angular

    def slight_rotation(self, image, angular_vel):
        angle = random.uniform(*ROTATION_ANGLE_RANGE)
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (width, height), borderMode=cv2.BORDER_REPLICATE)
        adjusted_angular = angular_vel - angle * np.pi / 180.0 * ROTATION_ANGLE_FACTOR
        return rotated, adjusted_angular

    def add_shadow(self, image, angular_vel):
        """增强方法8: 添加随机阴影"""
        height, width = image.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        x1, x2 = random.randint(0, width), random.randint(0, width)
        vertices = np.array([[(x1, 0), (x2, height), (0, height), (0, 0)]], dtype=np.int32)
        cv2.fillPoly(mask, vertices, 255)

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        darkness = random.uniform(*SHADOW_DARKNESS_RANGE)

        shadow_factor = 1 - (mask / 255.0) * (1 - darkness)
        hsv[:, :, 2] *= shadow_factor
        hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)

        hsv = hsv.astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), angular_vel

    # ---------------------- 核心逻辑 ----------------------
    def augment_image(self, image_path, index_start):
        filename = os.path.basename(image_path)
        match = re.match(r'^\d+_([-\d\.]+)\.jpg$', filename)
        if not match:
            print(f"警告: 无法从文件名解析角速度: {filename}")
            return [], index_start

        angular_vel = float(match.group(1))
        image = cv2.imread(image_path)
        if image is None:
            print(f"警告: 无法读取图像 {image_path}")
            return [], index_start

        results = []
        for _ in range(self.augment_factor):
            methods = random.sample(
                list(self.augment_methods.keys()),
                random.randint(*COMBINE_METHODS)
            )
            aug_img = image.copy()
            aug_ang = angular_vel
            for method in methods:
                aug_img, aug_ang = self.augment_methods[method](aug_img, aug_ang)

            new_fname = f"{index_start}_{aug_ang:.4f}.jpg"
            cv2.imwrite(os.path.join(self.output_dir, new_fname), aug_img)
            results.append(new_fname)
            index_start += 1

        return results, index_start

    def process_dataset(self):
        files = [f for f in os.listdir(self.input_dir)
                 if f.lower().endswith('.jpg') and os.path.isfile(os.path.join(self.input_dir, f))]
        if not files:
            print(f"❌ 未在目录 {self.input_dir} 中找到图像。")
            return

        print(f"共找到 {len(files)} 张图像，开始增强...")
        total = 0
        index_start = 0

        for fname in tqdm(files):
            path = os.path.join(self.input_dir, fname)
            _, index_start = self.augment_image(path, index_start)
            total = index_start

        print(f"✅ 增强完成，总生成图像数：{total}")

# ---------------------- CLI入口 ----------------------
def main():
    parser = argparse.ArgumentParser(description='自动驾驶数据集增强脚本')
    parser.add_argument('--input', type=str, default=INPUT_DIR, help='原始数据目录')
    parser.add_argument('--output', type=str, default=OUTPUT_DIR, help='增强图像保存目录')
    parser.add_argument('--factor', type=int, default=AUGMENT_FACTOR, help='每张图像生成数量')
    parser.add_argument('--methods', type=int, nargs='+', default=ENABLED_METHODS,
                        help='启用的方法编号列表 (1~8)')
    args = parser.parse_args()

    augmenter = DataAugmenter(args.input, args.output, args.factor, args.methods)
    augmenter.process_dataset()

if __name__ == '__main__':
    main()


