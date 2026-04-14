#!/usr/bin/env python
# -*- encoding: utf-8 -*-

# 导入torch库
import os
import torch.backends.cudnn as cudnn
import torch
from torch import nn
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

# 导入自定义库
from models import AutoDriveNet
from datasets import AutoDriveDataset
from utils import *


def main():
    """
    训练（九类转向角分类）.
    """
    # 数据集路径
    data_folder = os.getenv("VENET_DATA_FOLDER", "data/simulate/data")

    # 学习参数
    checkpoint = None  # 预训练模型路径，如果不存在则为None
    batch_size = int(os.getenv("VENET_BATCH_SIZE", "32"))  # 批大小
    start_epoch = 1  # 轮数起始位置
    epochs = int(os.getenv("VENET_EPOCHS", "100"))  # 迭代轮数
    lr = float(os.getenv("VENET_LR", "1e-4"))  # 学习率

    # 设备参数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("使用 GPU:", torch.cuda.get_device_name(0))
    else:
        print("当前使用 CPU")
    ngpu = 1  # 用来运行的gpu数量
    cudnn.benchmark = True  # 对卷积进行加速
    log_dir = os.getenv("VENET_LOG_DIR", "")
    writer = SummaryWriter(log_dir=log_dir if log_dir else None)  # 实时监控

    # 初始化模型
    model = AutoDriveNet()

    # 初始化优化器
    optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad,
                                               model.parameters()),
                                 lr=lr)

    # 迁移至默认设备进行训练
    model = model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)

    # 加载预训练模型
    if checkpoint is not None:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    # 单机多卡训练
    if torch.cuda.is_available() and ngpu > 1:
        model = nn.DataParallel(model, device_ids=list(range(ngpu)))

    # 定制化的dataloader
    transformations = transforms.Compose([
        transforms.Resize((120, 160)),
        transforms.ToTensor(),  # 通道置前并且将0-255RGB值映射至0-1
    ])

    train_dataset = AutoDriveDataset(data_folder,
                                     mode='train',
                                     transform=transformations)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=0,
                                               pin_memory=True)

    # 开始逐轮训练
    for epoch in range(start_epoch, epochs + 1):

        model.train()  # 训练模式：允许使用批样本归一化
        loss_epoch = AverageMeter()
        acc_epoch = AverageMeter()
        n_iter = len(train_loader)

        # 按批处理
        for i, (imgs, labels) in enumerate(train_loader):
            print(f"Batch {i}: imgs.shape = {imgs.shape}")
            imgs = imgs.to(device)
            labels = labels.to(device)

            pre_logits = model(imgs)
            loss = criterion(pre_logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                pred = pre_logits.argmax(dim=1)
                acc = (pred == labels).float().mean().item()

            loss_epoch.update(loss.item(), imgs.size(0))
            acc_epoch.update(acc, imgs.size(0))

            print("第 " + str(i) + " 个batch训练结束")

        del imgs, labels, pre_logits

        writer.add_scalar('CE_Loss', loss_epoch.avg, epoch)
        writer.add_scalar('Train_Acc', acc_epoch.avg, epoch)
        print('epoch:' + str(epoch) + '  CE_Loss:' + str(loss_epoch.avg)
              + '  Train_Acc:' + str(acc_epoch.avg))

        model_state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()

        output_dir = os.getenv("VENET_OUTPUT_DIR", ".")
        save_name = os.getenv("VENET_SAVE_NAME", "ve2.pth")
        os.makedirs(output_dir, exist_ok=True)
        torch.save({
            'model': model_state_dict,
            'optimizer': optimizer.state_dict(),
        }, os.path.join(output_dir, save_name))

    writer.close()


if __name__ == '__main__':
    '''
    程序入口
    '''
    main()
