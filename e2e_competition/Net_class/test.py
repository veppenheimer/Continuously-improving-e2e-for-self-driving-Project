#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
鍗曞紶鍥惧儚娴嬭瘯锛氱敤鍘熷鐪熷€艰浆鍚戣锛堟枃浠跺悕鎴?--angle锛屼笉鍋氫節妗ｉ噺鍖栵級涓庢ā鍨嬮娴嬬殑杩炵画瑙ｇ爜杞悜瑙掑姣旓紝
骞惰緭鍑虹粷瀵瑰亸宸笌鍋忓樊鐜囷紙鐩稿 |鐪熷€紎 鐨勭櫨鍒嗘瘮锛夈€?
"""

import argparse
import os
import sys

import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image

from models import AutoDriveNet
from steering_config import MAX_DELTA, decode_output, split_output


def build_transform():
    return transforms.Compose([
        transforms.Resize((120, 160)),
        transforms.ToTensor(),
    ])


def parse_angle_from_filename(image_path):
    basename = os.path.basename(image_path)
    stem, _ = os.path.splitext(basename)
    token = stem.split('_')[-1]
    return float(token)


def load_tensor(image_path, transform):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f'鏃犳硶璇诲彇鍥惧儚: {image_path}')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img = Image.fromarray(img)
    return transform(img).unsqueeze(0)


def main():
    parser = argparse.ArgumentParser(description='鍗曞浘鍒嗙被鎺ㄧ悊锛屽姣旂湡鍊间笌棰勬祴杞悜瑙?')
    parser.add_argument('image', help='寰呮祴鍥惧儚璺緞')
    parser.add_argument(
        '--checkpoint', '-c', default='ve2.pth',
        help='璁粌寰楀埌鐨勬潈閲嶈矾寰勶紙榛樿 ve2.pth锛?')
    parser.add_argument(
        '--angle', type=float, default=None,
        help='鎵嬪姩鎸囧畾鐪熷€艰浆鍚戣锛堝姬搴?鏍囧害涓庢暟鎹竴鑷达紱涓嶈鍒欎粠鏂囦欢鍚嶆湯娈佃В鏋愶級')
    args = parser.parse_args()

    image_path = os.path.abspath(args.image)
    if not os.path.isfile(image_path):
        print(f'閿欒锛氭枃浠朵笉瀛樺湪 {image_path}', file=sys.stderr)
        sys.exit(1)

    if args.angle is not None:
        raw_angle = float(args.angle)
        angle_source = '鍛戒护琛?--angle'
    else:
        try:
            raw_angle = parse_angle_from_filename(image_path)
            angle_source = '鏂囦欢鍚?'
        except (ValueError, IndexError) as e:
            print(
                '閿欒锛氭棤娉曚粠鏂囦欢鍚嶈В鏋愯浆鍚戣锛岃浣跨敤 --angle 鎸囧畾鐪熷€笺€俓n'
                f'璇︽儏: {e}',
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
        output = model(x)
        logits, raw_delta = split_output(output)
        pred_class = int(logits.argmax(dim=1).item())
        pred_angle = float(decode_output(output).reshape(-1)[0].item())
        pred_delta = float((torch.tanh(raw_delta) * MAX_DELTA).reshape(-1)[0].item())

    abs_err = abs(pred_angle - raw_angle)
    abs_gt = abs(raw_angle)
    if abs_gt > 1e-12:
        deviation_rate_pct = (abs_err / abs_gt) * 100.0
        rate_str = f'{deviation_rate_pct:.4f}%'
    else:
        rate_str = '鏈畾涔夛紙鐪熷€间负 0锛屾棤娉曠浉瀵?|鐪熷€紎 褰掍竴鍖栵級'

    print(f'鍥惧儚璺緞: {image_path}')
    print(f'鐪熷€兼潵婧? {angle_source}')
    print(f'鐪熷€艰浆鍚戣锛堝師濮嬶級: {raw_angle}')
    print(f'鎺ㄧ悊杞悜瑙掞紙杩炵画瑙ｇ爜锛? {pred_angle}  [棰勬祴绫诲埆 {pred_class}, residual {pred_delta:.6f}]')
    print(f'缁濆鍋忓樊: {abs_err}')
    print(f'鍋忓樊鐜囷紙|鎺ㄧ悊-鐪熷€紎/|鐪熷€紎脳100%锛? {rate_str}')


if __name__ == '__main__':
    main()
