#!/usr/bin/env python
# -*- encoding: utf-8 -*-


class AverageMeter:
    """跟踪平均值、当前值和总和。"""

    def __init__(self, name="Metric", fmt=":6.4f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        val = float(val)
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} (avg:{avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)
