#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@文件        :trtCrun_pipeline.py
@说明        :ROS节点 + 流水线多线程结构 + TensorRT 零拷贝推理
"""

import threading
import queue
import rospy
from geometry_msgs.msg import Twist
import cv2, time, numpy as np, tensorrt as trt
import pycuda.driver as cuda
from pycuda.tools import context_dependent_memoize

# ---------- 配置 ----------
ENGINE_PATH  = "../results/ve_0524x_simplified_fp16.engine"
CAMERA_INDEX = 0
LINEAR_SPEED = 0.45
INPUT_SHAPE  = (1, 3, 120, 160)
# ---------------------------

def load_engine(path):
    logger = trt.Logger(trt.Logger.WARNING)
    with open(path, "rb") as f, trt.Runtime(logger) as rt:
        return rt.deserialize_cuda_engine(f.read())

def preprocess(frame, host_mem):
    img = cv2.resize(frame, (INPUT_SHAPE[3], INPUT_SHAPE[2]))
    cv2.cvtColor(img, cv2.COLOR_BGR2HSV, img)
    arr = np.asarray(img, dtype=np.float32) * (1.0 / 255.0)
    host_mem[:] = arr.transpose(2, 0, 1)[None, ...]

def camera_thread(cap, frame_queue):
    while True:
        ret, frame = cap.read()
        if ret:
            if not frame_queue.full():
                frame_queue.put(frame)
        else:
            time.sleep(0.01)

def infer_thread(engine, ctx, frame_queue, result_queue, cuda_ctx):
    # 在当前线程 push 主 CUDA 上下文
    cuda_ctx.push()

    # 分配零拷贝主机内存
    host_in = cuda.pagelocked_empty(INPUT_SHAPE, dtype=np.float32,
                                     mem_flags=cuda.host_alloc_flags.DEVICEMAP)
    host_out = cuda.pagelocked_empty((1,), dtype=np.float32,
                                      mem_flags=cuda.host_alloc_flags.DEVICEMAP)

    devptr_in = host_in.base.get_device_pointer()
    devptr_out = host_out.base.get_device_pointer()
    stream = cuda.Stream()

    while True:
        frame = frame_queue.get()
        start_infer = time.time()  # 推理开始时间

        preprocess(frame, host_in)
        ctx.execute_async_v2([int(devptr_in), int(devptr_out)], stream.handle)
        stream.synchronize()
        result = float(host_out[0])

        end_infer = time.time()  # 推理结束时间
        print("推理耗时: %.2f ms" % ((end_infer - start_infer) * 1000.0))

        if not result_queue.full():
            result_queue.put((result, start_infer))

def control_thread(result_queue, pub):
    rate = rospy.Rate(20)
    while not rospy.is_shutdown():
        if not result_queue.empty():
            angular, start_time = result_queue.get()
            twist = Twist()
            twist.linear.x = LINEAR_SPEED
            twist.angular.z = angular
            pub.publish(twist)

            end_time = time.time()
            print("单帧总耗时: %.2f ms" % ((end_time - start_time) * 1000.0))
        rate.sleep()

def main():
    # 手动初始化 CUDA
    cuda.init()
    device = cuda.Device(0)
    main_ctx = device.make_context()

    rospy.init_node("tensorrt_controller_pipeline", anonymous=True)
    pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)

    # 加载 TensorRT 引擎
    engine = load_engine(ENGINE_PATH)
    ctx = engine.create_execution_context()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        return

    frame_queue = queue.Queue(maxsize=1)
    result_queue = queue.Queue(maxsize=1)

    # 启动线程
    t1 = threading.Thread(target=camera_thread,
                          args=(cap, frame_queue),
                          daemon=True)
    t2 = threading.Thread(target=infer_thread,
                          args=(engine, ctx, frame_queue, result_queue, main_ctx),
                          daemon=True)
    t3 = threading.Thread(target=control_thread,
                          args=(result_queue, pub),
                          daemon=True)

    t1.start()
    t2.start()
    t3.start()

    rospy.spin()

    # 清理
    cap.release()
    main_ctx.pop()
    main_ctx.detach()

if __name__ == "__main__":
    main()
