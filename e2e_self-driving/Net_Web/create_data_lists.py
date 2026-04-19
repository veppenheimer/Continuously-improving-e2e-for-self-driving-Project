#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""本地生成 train.txt / val.txt / test.txt（网页上传 ZIP 时由服务端自动完成，此脚本供离线使用）。"""

import os
import random


def create_data_list(dataset_path, file_list, mode="train"):
    with open(os.path.join(dataset_path, f"{mode}.txt"), "w", encoding="utf-8") as f:
        for img_path, angle in file_list:
            f.write(f"{img_path} {angle}\n")
    print(f"{mode}.txt 已生成")


def get_file_list(path, file_list, ext=None):
    if os.path.isfile(path):
        if ext is None or path.lower().endswith(ext.lower()):
            file_list.append(path)
        return file_list

    if os.path.isdir(path):
        for entry in os.listdir(path):
            get_file_list(os.path.join(path, entry), file_list, ext)
    return file_list


def split_three_way(file_list, train_ratio=0.7, val_ratio=0.15):
    n_total = len(file_list)
    if n_total < 3:
        raise RuntimeError("至少需要 3 张图像，才能生成 train/val/test 三个列表。")

    n_train = max(1, int(n_total * train_ratio))
    n_val = max(1, int(n_total * val_ratio))
    n_test = n_total - n_train - n_val

    if n_test < 1:
        deficit = 1 - n_test
        take_from_train = min(deficit, max(0, n_train - 1))
        n_train -= take_from_train
        deficit -= take_from_train
        take_from_val = min(deficit, max(0, n_val - 1))
        n_val -= take_from_val
        deficit -= take_from_val
        if deficit > 0:
            raise RuntimeError("数据量不足，无法同时保留 train/val/test 三个非空划分。")
        n_test = n_total - n_train - n_val

    train_list = file_list[:n_train]
    val_list = file_list[n_train:n_train + n_val]
    test_list = file_list[n_train + n_val:]
    return train_list, val_list, test_list


def main():
    org_img_folder = "./data/1_1"
    train_ratio = 0.7
    val_ratio = 0.15
    jpg_list = get_file_list(org_img_folder, [], "jpg")
    print("本次执行检索到 " + str(len(jpg_list)) + " 个 jpg 文件\n")

    file_list = []
    for jpg_path in jpg_list:
        cur_dir = os.path.dirname(jpg_path)
        basename = os.path.basename(jpg_path)
        angle = (basename[:-4]).split("_")[-1]
        img_path = os.path.join(cur_dir, basename).replace("\\", "/")
        file_list.append((img_path, angle))

    random.seed(256)
    random.shuffle(file_list)
    train_list, val_list, test_list = split_three_way(file_list, train_ratio=train_ratio, val_ratio=val_ratio)
    create_data_list(org_img_folder, train_list, mode="train")
    create_data_list(org_img_folder, val_list, mode="val")
    create_data_list(org_img_folder, test_list, mode="test")


if __name__ == "__main__":
    main()