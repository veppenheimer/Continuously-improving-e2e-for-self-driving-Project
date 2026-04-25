#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@文件        :trtCrun.py
@说明        :极简版ROS节点，使用TensorRT模型实时推理并控制小车
            ? 异步多线程摄像头采集
            ? 零拷贝主机内存
            ? 保持原有推理与控制逻辑不变
"""

import threading
import rospy
from geometry_msgs.msg import Twist
import cv2, time, numpy as np, tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

# ---------- 配置 ----------
ENGINE_PATH  = "../results/ve_0524x_simplified_fp16.engine"
CAMERA_INDEX = 0
LINEAR_SPEED = 0.5
INPUT_SHAPE  = (1, 3, 120, 160)
# ---------------------------

class CameraCapture(threading.Thread):
    """异步摄像头采集线程，只保留最新帧"""
    def __init__(self, index):
        super().__init__(daemon=True)
        self.cap = cv2.VideoCapture(index)
        self.lock = threading.Lock()
        self.frame = None
        self.running = True

    def run(self):
        while self.running:
            ret, frm = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frm
            else:
                time.sleep(0.01)

    def get_frame(self):
        with self.lock:
            return self.frame

    def stop(self):
        self.running = False
        self.cap.release()

def load_engine(path):
    logger = trt.Logger(trt.Logger.WARNING)
    with open(path, "rb") as f, trt.Runtime(logger) as rt:
        return rt.deserialize_cuda_engine(f.read())

def preprocess(frame, host_mem):
    """BGR→HSV→归一化→CHW→NCHW，写入 host_mem（零拷贝缓冲）"""
    # 上传图像到GPU
    gpu_img = cv2.cuda_GpuMat()
    gpu_img.upload(frame)

    # 调整尺寸
    gpu_img = cv2.cuda.resize(gpu_img, (INPUT_SHAPE[3], INPUT_SHAPE[2]))

    # 颜色空间转换 BGR → HSV
    gpu_img = cv2.cuda.cvtColor(gpu_img, cv2.COLOR_BGR2HSV)

    # 下载回CPU并进行归一化、格式转换
    img = gpu_img.download()
    arr = np.asarray(img, dtype=np.float32) * (1.0 / 255.0)
    host_mem[:] = arr.transpose(2, 0, 1)[None, ...]


def main():
    rospy.init_node("tensorrt_controller", anonymous=True)
    pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
    rate = rospy.Rate(20)

    # 加载 TensorRT 引擎
    engine = load_engine(ENGINE_PATH)
    ctx    = engine.create_execution_context()

    # 分配页锁定、可映射主机内存（零拷贝）
    host_in = cuda.pagelocked_empty(INPUT_SHAPE, dtype=np.float32,
                                     mem_flags=cuda.host_alloc_flags.DEVICEMAP)
    devptr_in = host_in.base.get_device_pointer()

    host_out = cuda.pagelocked_empty((1,), dtype=np.float32,
                                      mem_flags=cuda.host_alloc_flags.DEVICEMAP)
    devptr_out = host_out.base.get_device_pointer()

    stream = cuda.Stream()

    # 启动摄像头采集线程
    cam_thread = CameraCapture(CAMERA_INDEX)
    cam_thread.start()

    try:
        while not rospy.is_shutdown():
            start_total = time.time()  # 总耗时开始

            frame = cam_thread.get_frame()
            if frame is None:
                rate.sleep()
                continue

            # 预处理写入零拷贝缓冲
            preprocess(frame, host_in)

            # 推理：直接在映射缓冲上执行
            start_infer = time.time()  # 推理耗时开始
            ctx.execute_async_v2([int(devptr_in), int(devptr_out)], stream.handle)
            stream.synchronize()
            end_infer = time.time()    # 推理耗时结束

            # 发布 Twist，与原逻辑一致
            twist = Twist()
            twist.linear.x  = LINEAR_SPEED
            twist.angular.z = float(host_out[0])
            pub.publish(twist)

            end_total = time.time()  # 总耗时结束

            print("推理耗时: %.2f ms | 单帧总耗时: %.2f ms" % (
                (end_infer - start_infer) * 1000.0,
                (end_total - start_total) * 1000.0
            ))

            rate.sleep()
    finally:
        cam_thread.stop()


if __name__ == "__main__":
    main()
