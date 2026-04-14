#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""本地生成 train.txt / val.txt（网页上传 ZIP 时由服务端自动完成，此脚本供离线使用）。"""

import os
import random


def creat_data_list(dataset_path, file_list, mode="train"):
    with open(os.path.join(dataset_path, (mode + ".txt")), "w", encoding="utf-8") as f:
        for (imgpath, angle) in file_list:
            f.write(imgpath + " " + str(angle) + "\n")
    print(mode + ".txt 已生成")


def getFileList(dir, Filelist, ext=None):
    newDir = dir
    if os.path.isfile(dir):
        if ext is None:
            Filelist.append(dir)
        else:
            if ext in dir[-3:]:
                Filelist.append(dir)
    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            newDir = os.path.join(dir, s)
            getFileList(newDir, Filelist, ext)
    return Filelist


def main():
    org_img_folder = "./data/1_1"
    train_ratio = 0.8
    jpglist = getFileList(org_img_folder, [], "jpg")
    print("本次执行检索到 " + str(len(jpglist)) + " 个jpg文件\n")

    file_list = []
    for jpgpath in jpglist:
        curDataDir = os.path.dirname(jpgpath)
        basename = os.path.basename(jpgpath)
        angle = (basename[:-4]).split("_")[-1]
        imgPath = os.path.join(curDataDir, basename).replace("\\", "/")
        file_list.append((imgPath, angle))

    random.seed(256)
    random.shuffle(file_list)
    train_num = int(len(file_list) * train_ratio)
    train_list = file_list[0:train_num]
    val_list = file_list[train_num:]
    creat_data_list(org_img_folder, train_list, mode="train")
    creat_data_list(org_img_folder, val_list, mode="val")


if __name__ == "__main__":
    main()
