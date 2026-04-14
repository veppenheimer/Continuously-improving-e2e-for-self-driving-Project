#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""原 Net 脚本式训练入口（无 HTTP）。需要本地已准备含 train.txt / val.txt 的 data_folder。"""

import torch.backends.cudnn as cudnn
import torch
from torch import nn
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

from models import AutoDriveNet
from datasets import AutoDriveDataset
from utils import *


def main():
    data_folder = "data/simulate/data"
    checkpoint = None
    batch_size = 32
    start_epoch = 1
    epochs = 100
    lr = 1e-4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("使用 GPU:", torch.cuda.get_device_name(0))
    else:
        print("当前使用 CPU")
    ngpu = 1
    cudnn.benchmark = True
    writer = SummaryWriter()

    model = AutoDriveNet()
    optimizer = torch.optim.Adam(
        params=filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
    )

    model = model.to(device)
    criterion = nn.MSELoss().to(device)

    if checkpoint is not None:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint["epoch"] + 1
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])

    if torch.cuda.is_available() and ngpu > 1:
        model = nn.DataParallel(model, device_ids=list(range(ngpu)))

    transformations = transforms.Compose(
        [
            transforms.Resize((120, 160)),
            transforms.ToTensor(),
        ]
    )

    train_dataset = AutoDriveDataset(data_folder, mode="train", transform=transformations)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        loss_epoch = AverageMeter()

        for i, (imgs, labels) in enumerate(train_loader):
            print(f"Batch {i}: imgs.shape = {imgs.shape}")
            imgs = imgs.to(device)
            labels = labels.to(device)
            pre_labels = model(imgs)
            loss = criterion(pre_labels, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_epoch.update(loss.item(), imgs.size(0))
            print("第 " + str(i) + " 个batch训练结束")

        del imgs, labels, pre_labels

        writer.add_scalar("MSE_Loss", loss_epoch.avg, epoch)
        print("epoch:" + str(epoch) + "  MSE_Loss:" + str(loss_epoch.avg))

        model_state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()

        torch.save(
            {
                "model": model_state_dict,
                "optimizer": optimizer.state_dict(),
            },
            "ve2.pth",
        )

    writer.close()


if __name__ == "__main__":
    main()
