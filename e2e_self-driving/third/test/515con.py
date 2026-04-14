#!/usr/bin/env python
# -*- coding: utf-8 -*-
import threading
import rospy
from geometry_msgs.msg import Twist
from yolo_msgs.msg import YoloSign
import numpy as np
import cv2
import time
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from collections import deque

# ---------- 配置 ----------
ENGINE_PATH  = "../results/ve_0525x_simplified_fp16.engine"
CAMERA_INDEX = 0
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


class MotionController:

    def __init__(self):
        rospy.init_node("tensorrt_controller", anonymous=True)

        self.twist_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        rospy.Subscriber('/yolo_sign', YoloSign, self.sign_callback)

        # 交通标志相关变量
        self.sign_queue = deque()
        self.current_action = None
        self.current_sign = None
        self.prev_processed_sign = None
        self.action_start_time = 0.0
        self.action_duration = 0.0
        self.limit_handled = False
        self.delay_before_red = 7.2

        # 控制参数
        self.default_linear_speed = 0.2
        self.slow_linear_speed = 0.12
        self.turn_angular_speed = 0.5
        self.stop1_duration = 2.0
        self.stop2_duration = 100.0
        self.stop_duration_person = 1.0
        self.slow_duration = 40.0
        self.turn_duration = 3.0

        # 加载 TensorRT 引擎
        rospy.loginfo("Loading TensorRT engine from %s ...", ENGINE_PATH)
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with open(ENGINE_PATH, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        rospy.loginfo("TensorRT engine loaded.")
        self.context = self.engine.create_execution_context()

        # 分配零拷贝内存
        self.host_in = cuda.pagelocked_empty(INPUT_SHAPE, dtype=np.float32,
                                             mem_flags=cuda.host_alloc_flags.DEVICEMAP)
        self.devptr_in = self.host_in.base.get_device_pointer()

        self.host_out = cuda.pagelocked_empty((1,), dtype=np.float32,
                                              mem_flags=cuda.host_alloc_flags.DEVICEMAP)
        self.devptr_out = self.host_out.base.get_device_pointer()

        self.stream = cuda.Stream()

        # 启动异步摄像头采集线程
        self.cam_thread = CameraCapture(CAMERA_INDEX)
        self.cam_thread.start()

        self.rate = rospy.Rate(20)

    def clear_sign_file(self):
        path = '/opt/nvidia/deepstream/deepstream-6.0/sources/DeepStream-Yolo/nvdsinfer_custom_impl_Yolo/yolosign.txt'
        try:
            with open(path, 'w'):
                pass
            rospy.loginfo("Cleared sign file: %s", path)
        except Exception as e:
            rospy.logerr("Failed to clear sign file %s: %s", path, e)

    def sign_callback(self, msg):
        if msg.sign_type == 'limit' and self.limit_handled:
            return
        if self.current_sign == 'red' and msg.sign_type == 'green':
            self.current_action = None
            self.current_sign = None
            self.prev_processed_sign = msg.sign_type
            self.clear_sign_file()
            return
        if self.current_sign == 'limit' and msg.sign_type == 'remove_limit':
            self.current_action = None
            self.current_sign = None
            self.prev_processed_sign = msg.sign_type
            self.clear_sign_file()
            return
        rospy.loginfo("Queued sign: %s", msg.sign_type)
        self.sign_queue.append(msg.sign_type)

    def preprocess(self, frame):
        """图像预处理并写入零拷贝 host_in"""
        img = cv2.resize(frame, (INPUT_SHAPE[3], INPUT_SHAPE[2]))
        cv2.cvtColor(img, cv2.COLOR_BGR2HSV, img)
        arr = np.asarray(img, dtype=np.float32) * (1.0 / 255.0)
        self.host_in[:] = arr.transpose(2, 0, 1)[None, ...]

    def infer(self):
        self.context.execute_async_v2([int(self.devptr_in), int(self.devptr_out)],
                                      self.stream.handle)
        self.stream.synchronize()
        return float(self.host_out[0])

    def spin(self):
        while not rospy.is_shutdown():
            twist = Twist()
            now = rospy.get_time()

            frame = self.cam_thread.get_frame()
            if frame is None:
                self.rate.sleep()
                continue

            if self.current_action is None and self.sign_queue:
                sign = self.sign_queue.popleft()
                if sign == 'red' and self.prev_processed_sign != 'remove_limit':
                    continue
                if sign == 'red':
                    action = 'delay_red'; duration = self.delay_before_red
                elif sign == 'limit':
                    action = 'slow'; duration = self.slow_duration
                    self.limit_handled = True
                elif sign in ('green', 'remove_limit'):
                    continue
                elif sign == 'crossing_walk':
                    action = 'stop1'; duration = self.stop1_duration
                elif sign == 'person':
                    action = 'person'; duration = self.stop_duration_person
                else:
                    rospy.logwarn("Unknown sign %s", sign)
                    continue
                self.current_action = action
                self.current_sign = sign
                self.action_start_time = now
                self.action_duration = duration

            if self.current_action is not None:
                elapsed = now - self.action_start_time
                if elapsed >= self.action_duration:
                    self.current_action = None
                    self.current_sign = None
                    continue
                if self.current_action == 'stop1' or self.current_action == 'person':
                    twist.linear.x = 0.0
                elif self.current_action == 'slow':
                    twist.linear.x = self.slow_linear_speed
                elif self.current_action == 'delay_red':
                    twist.linear.x = 0.0
                else:
                    twist.linear.x = self.default_linear_speed
                twist.angular.z = 0.0
            else:
                self.preprocess(frame)
                angle = self.infer()
                twist.linear.x = self.default_linear_speed
                twist.angular.z = angle

            self.twist_pub.publish(twist)
            self.rate.sleep()

        self.cam_thread.stop()

if __name__ == "__main__":
    controller = MotionController()
    controller.spin()
