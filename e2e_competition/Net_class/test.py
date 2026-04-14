#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
单张图像测试：用原始真值转向角（文件名或 --angle，不做九档量化）与模型预测的九档转向角对比，
并输出绝对偏差与偏差率（相对 |真值| 的百分比）。
"""

import argparse
import os
import sys

import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image

from models import AutoDriveNet
from steering_config import class_to_angle


def build_transform():
    return transforms.Compose([
        transforms.Resize((120, 160)),
        transforms.ToTensor(),
    ])


def parse_angle_from_filename(image_path):
    """与数据命名一致：去掉扩展名后按 '_' 分割，取最后一段为浮点角。"""
    basename = os.path.basename(image_path)
    stem, _ = os.path.splitext(basename)
    token = stem.split('_')[-1]
    return float(token)


def load_tensor(image_path, transform):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f'无法读取图像: {image_path}')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img = Image.fromarray(img)
    return transform(img).unsqueeze(0)


def main():
    parser = argparse.ArgumentParser(description='单图分类推理，对比真值与预测转向角')
    parser.add_argument('image', help='待测图像路径')
    parser.add_argument(
        '--checkpoint', '-c', default='ve2.pth',
        help='训练得到的权重路径（默认 ve2.pth）')
    parser.add_argument(
        '--angle', type=float, default=None,
        help='手动指定真值转向角（弧度/标度与数据一致；不设则从文件名末段解析）')
    args = parser.parse_args()

    image_path = os.path.abspath(args.image)
    if not os.path.isfile(image_path):
        print(f'错误：文件不存在 {image_path}', file=sys.stderr)
        sys.exit(1)

    if args.angle is not None:
        raw_angle = float(args.angle)
        angle_source = '命令行 --angle'
    else:
        try:
            raw_angle = parse_angle_from_filename(image_path)
            angle_source = '文件名'
        except (ValueError, IndexError) as e:
            print(
                '错误：无法从文件名解析转向角，请使用 --angle 指定真值。\n'
                f'详情: {e}',
                file=sys.stderr)
            sys.exit(1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AutoDriveNet().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model'], strict=True)
    model.eval()

    transform = build_transform()
    x = load_tensor(image_path, transform).to(device)

    with torch.no_grad():
        logits = model(x)
        pred_class = int(logits.argmax(dim=1).item())
    pred_angle = class_to_angle(pred_class)

    abs_err = abs(pred_angle - raw_angle)
    abs_gt = abs(raw_angle)
    # 偏差率：|推理-真值| / |真值| × 100%；真值为 0 时分母无定义
    if abs_gt > 1e-12:
        deviation_rate_pct = (abs_err / abs_gt) * 100.0
        rate_str = f'{deviation_rate_pct:.4f}%'
    else:
        rate_str = '未定义（真值为 0，无法相对 |真值| 归一化）'

    print(f'图像路径: {image_path}')
    print(f'真值来源: {angle_source}')
    print(f'真值转向角（原始）: {raw_angle}')
    print(f'推理转向角（模型九档）: {pred_angle}  [预测类别 {pred_class}]')
    print(f'绝对偏差: {abs_err}')
    print(f'偏差率（|推理-真值|/|真值|×100%）: {rate_str}')


if __name__ == '__main__':
    main()
