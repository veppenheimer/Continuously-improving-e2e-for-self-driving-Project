#!/usr/bin/env python
# -*- coding: utf-8 -*-
import threading
import queue
import rospy
from geometry_msgs.msg import Twist
from yolo_msgs.msg import YoloSign
import cv2, numpy as np, tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from collections import deque
import random

# ---------- 配置 ----------
ENGINE_PATH  = rospy.get_param('~engine_path', './ve_simplified_fp16.engine')
CAMERA_INDEX = rospy.get_param('~camera_index', 0)
INPUT_SHAPE  = (1, 3, 120, 160)
# ---------------------------

class CameraCapture(threading.Thread):
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
                rospy.sleep(0.01)

    def get_frame(self):
        with self.lock:
            return self.frame

    def stop(self):
        self.running = False
        self.cap.release()

class AsyncPublisher(threading.Thread):
    def __init__(self, topic, msg_type, queue_size=1):
        super().__init__(daemon=True)
        self.publisher = rospy.Publisher(topic, msg_type, queue_size=queue_size)
        self.queue = queue.Queue(maxsize=5)
        self.running = True

    def run(self):
        while self.running and not rospy.is_shutdown():
            try:
                msg = self.queue.get(timeout=0.1)
                self.publisher.publish(msg)
            except queue.Empty:
                continue

    def send(self, msg):
        if self.queue.full():
            try:
                self.queue.get_nowait()
            except queue.Empty:
                pass
        self.queue.put_nowait(msg)

    def stop(self):
        self.running = False

def wait_for_user_start():
    rospy.loginfo("模型加载完毕。请在键盘上按 'r' 键开始运动……")
    try:
        user_input = input("Press 'r' to start: ")
        while user_input.strip().lower() != 'r':
            user_input = input("Invalid input. Press 'r' to start: ")
    except EOFError:
        rospy.logwarn("无法读取输入，直接开始（可能是在后台运行）")

class MotionController:
    def __init__(self):
        rospy.init_node('motion_controller_node', anonymous=False)
        rospy.loginfo("Loading TensorRT engine from %s ...", ENGINE_PATH)

        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with open(ENGINE_PATH, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        rospy.loginfo("TensorRT engine loaded.")
        self.context = self.engine.create_execution_context()

        self.host_in = cuda.pagelocked_empty(INPUT_SHAPE, dtype=np.float32, mem_flags=cuda.host_alloc_flags.DEVICEMAP)
        self.devptr_in = self.host_in.base.get_device_pointer()
        self.host_out = cuda.pagelocked_empty((1,), dtype=np.float32, mem_flags=cuda.host_alloc_flags.DEVICEMAP)
        self.devptr_out = self.host_out.base.get_device_pointer()
        self.stream = cuda.Stream()

        self.async_pub = AsyncPublisher('/cmd_vel', Twist)
        self.async_pub.start()
        rospy.Subscriber('/yolo_sign', YoloSign, self.sign_callback)

        self.cam_thread = CameraCapture(CAMERA_INDEX)
        self.cam_thread.start()

        self.sign_queue = deque()
        self.current_action = None
        self.current_sign = None
        self.prev_processed_sign = None
        self.action_start_time = 0.0
        self.action_duration = 0.0
        self.limit_handled = False
        self.crosswalk_handled = False

        self.delay_before_red = 2.75

        self.default_linear_speed = rospy.get_param('~default_linear_speed', 0.5)
        self.slow_linear_speed    = rospy.get_param('~slow_linear_speed', 0.42)
        self.stop1_duration       = rospy.get_param('~stop1_duration', 2.0)
        self.stop2_duration       = rospy.get_param('~stop2_duration', 100.0)
        self.stop_duration_person = rospy.get_param('~stop_duration_person', 1.5)
        self.slow_duration        = rospy.get_param('~slow_duration', 40.0)

        self.rate = rospy.Rate(30)
        self.init_motions = deque([
           # 右
            # {"duration": 0.6, "linear": 0.5, "angular": 0.0},
            # {"duration": 0.79, "linear": 0.5, "angular": -2.6},
            # {"duration": 0.10, "linear": 0.5, "angular": 0.0},
            # {"duration": 2.6, "linear": 0.5, "angular": 1.47},
            # {"duration": 0.33, "linear": 0.5, "angular": 0.0},
            # {"duration": 0.86, "linear": 0.5, "angular": -2.2},
            # {"duration": 1.5, "linear": 0.0, "angular": 0.0},
            # 左
            {"duration": 0.45, "linear": 0.5, "angular": 0.0},
            {"duration": 0.78, "linear": 0.5, "angular": 2.6},
            {"duration": 0.08, "linear": 0.5, "angular": 0.0},
            {"duration": 2.54, "linear": 0.5, "angular": -1.48},
            {"duration": 0.40, "linear": 0.5, "angular": 0.0},
            {"duration": 0.91, "linear": 0.5, "angular": 1.8},
            {"duration": 1.5, "linear": 0.0, "angular": 0.0},
            # {"duration": 0.49, "linear": 0.5, "angular": 0.0},

        ])
        self.init_start_time = None
        self.initializing = True

        self.interrupted_action = None
        self.interrupted_sign = None
        self.interrupted_duration = None

        # 新增标志与person延迟轨迹
        self.green_handled = False
        self.person_delay_active = False
        self.person_delay_start_time = None
        self.person_delay_current_motion = None
        self.person_delay_motions = self.generate_example_person_delay()

    def generate_example_person_delay(self):
        return deque([
            {"duration": 2.7, "linear": 0.5, "angular": 1.51},
            {"duration": 0.65, "linear": 0.5, "angular": -1.62},
            {"duration": 0.9, "linear": 0.5, "angular": 1.72},
            {"duration": 1.0, "linear": 0.5, "angular": 0.0},
        ])

    def clear_sign_file(self):
        path = '/opt/nvidia/deepstream/deepstream-6.0/sources/DeepStream-Yolo/nvdsinfer_custom_impl_Yolo/yolosign.txt'
        try:
            with open(path, 'w'): pass
            rospy.loginfo("Cleared sign file: %s", path)
        except Exception as e:
            rospy.logerr("Failed to clear sign file %s: %s", path, e)

    def sign_callback(self, msg):
        if msg.sign_type == 'green':
            rospy.loginfo("Green light detected.")
            self.green_handled = True
            self.clear_sign_file()
            return

        if msg.sign_type == 'person':
            rospy.loginfo("Person detected.")
            if self.green_handled:
                rospy.loginfo("Green light was handled, setting person_delay after person stop.")
            if self.current_action != 'person':
                self.interrupted_action = self.current_action
                self.interrupted_sign = self.current_sign
                self.interrupted_duration = self.action_duration - (rospy.get_time() - self.action_start_time)
                self.current_action = 'person'
                self.current_sign = 'person'
                self.action_start_time = rospy.get_time()
                self.action_duration = self.stop_duration_person
            else:
                self.action_start_time = rospy.get_time()
                self.clear_sign_file()
                rospy.loginfo("Refreshing person stop timer.")
            return

        if msg.sign_type == 'limit' and self.limit_handled:
            return
        if msg.sign_type == 'crossing_walk' and self.crosswalk_handled:
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
        img = cv2.resize(frame, (INPUT_SHAPE[3], INPUT_SHAPE[2]))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        arr = hsv.astype(np.float32) * (1.0 / 255.0)
        self.host_in[:] = arr.transpose(2, 0, 1)[None, ...]

    def infer_trt(self):
        self.context.execute_async_v2([int(self.devptr_in), int(self.devptr_out)], self.stream.handle)
        self.stream.synchronize()
        return float(self.host_out[0])

    def spin(self):
        while not rospy.is_shutdown():
            twist = Twist()
            now = rospy.get_time()

            if self.person_delay_active:
                if not self.person_delay_current_motion and self.person_delay_motions:
                    self.person_delay_current_motion = self.person_delay_motions.popleft()
                    self.person_delay_start_time = now

                if self.person_delay_current_motion:
                    elapsed_delay = now - self.person_delay_start_time
                    if elapsed_delay < self.person_delay_current_motion['duration']:
                        twist.linear.x = self.person_delay_current_motion['linear']
                        twist.angular.z = self.person_delay_current_motion['angular']
                        self.async_pub.send(twist)
                        self.rate.sleep()
                        continue
                    else:
                        self.person_delay_current_motion = None
                        continue
                else:
                    if not self.person_delay_motions:
                        rospy.loginfo("Person delay trajectory completed.")
                        self.person_delay_active = False
                        self.green_handled = False
                        if self.interrupted_action:
                            rospy.loginfo("Resuming interrupted action: %s", self.interrupted_action)
                            self.current_action = self.interrupted_action
                            self.current_sign = self.interrupted_sign
                            self.action_duration = self.interrupted_duration
                            self.action_start_time = rospy.get_time()
                            self.interrupted_action = None
                            self.interrupted_sign = None
                            self.interrupted_duration = None
                        continue

            if self.initializing:
                if self.init_motions:
                    current_motion = self.init_motions[0]
                    if self.init_start_time is None:
                        self.init_start_time = now
                    elapsed = now - self.init_start_time
                    if elapsed < current_motion['duration']:
                        twist.linear.x = current_motion['linear']
                        twist.angular.z = current_motion['angular']
                        self.async_pub.send(twist)
                        self.rate.sleep()
                        continue
                    else:
                        self.init_motions.popleft()
                        self.init_start_time = None
                        continue
                else:
                    self.initializing = False

            if self.current_action is None and self.sign_queue:
                sign = self.sign_queue.popleft()
                if sign == 'red':
                    action = 'delay_red'; duration = self.delay_before_red
                elif sign == 'limit':
                    action = 'delay_limit'; duration = 0.01
                    self.limit_handled = True
                elif sign == 'crossing_walk' and not self.crosswalk_handled:
                    action = 'stop1'; duration = self.stop1_duration
                else:
                    continue
                self.current_action = action
                self.current_sign = sign
                self.action_duration = duration
                self.action_start_time = now

            if self.current_action:
                elapsed = now - self.action_start_time

                if self.current_action == 'person':
                    if elapsed < self.action_duration:
                        twist.linear.x = twist.angular.z = 0.0
                        self.async_pub.send(twist)
                        self.rate.sleep()
                        continue
                    else:
                        rospy.loginfo("Person stop completed.")
                        self.clear_sign_file()
                        self.current_action = None
                        self.current_sign = None
                        if self.green_handled:
                            rospy.loginfo("Starting person_delay trajectory after person stop.")
                            self.person_delay_active = True
                            self.person_delay_current_motion = None
                            continue
                        if self.interrupted_action:
                            rospy.loginfo("Resuming interrupted action: %s", self.interrupted_action)
                            self.current_action = self.interrupted_action
                            self.current_sign = self.interrupted_sign
                            self.action_duration = self.interrupted_duration
                            self.action_start_time = rospy.get_time()
                            self.interrupted_action = None
                            self.interrupted_sign = None
                            self.interrupted_duration = None
                        continue

                if self.current_action == 'delay_limit':
                    if elapsed < self.action_duration:
                        frame = self.cam_thread.get_frame()
                        if frame is not None:
                            self.preprocess(frame)
                            ang_z = self.infer_trt()
                            twist.linear.x = self.default_linear_speed
                            twist.angular.z = ang_z
                        else:
                            twist.linear.x = twist.angular.z = 0.0
                        self.async_pub.send(twist)
                        self.rate.sleep()
                        continue
                    else:
                        self.current_action = 'slow'
                        self.action_duration = self.slow_duration
                        self.action_start_time = now
                        continue

                if self.current_action == 'delay_red':
                    if elapsed < self.action_duration:
                        frame = self.cam_thread.get_frame()
                        if frame is not None:
                            self.preprocess(frame)
                            ang_z = self.infer_trt()
                            twist.linear.x = self.default_linear_speed
                            twist.angular.z = ang_z
                        else:
                            twist.linear.x = twist.angular.z = 0.0
                        self.async_pub.send(twist)
                        self.rate.sleep()
                        continue
                    else:
                        self.current_action = 'stop2'
                        self.action_duration = self.stop2_duration
                        self.action_start_time = now
                        continue

                if self.current_action in ('stop1', 'stop2'):
                    if elapsed < self.action_duration:
                        twist.linear.x = twist.angular.z = 0.0
                        self.async_pub.send(twist)
                        self.rate.sleep()
                        continue
                    else:
                        self.clear_sign_file()
                        self.prev_processed_sign = self.current_sign
                        if self.current_sign == 'crossing_walk':
                            self.crosswalk_handled = True
                        self.current_action = None
                        self.current_sign = None
                        continue

                if self.current_action == 'slow':
                    if elapsed < self.action_duration:
                        frame = self.cam_thread.get_frame()
                        if frame is not None:
                            self.preprocess(frame)
                            ang_z = self.infer_trt() * 0.85
                        else:
                            ang_z = 0.0
                        twist.linear.x = self.slow_linear_speed
                        twist.angular.z = ang_z
                        self.async_pub.send(twist)
                        self.rate.sleep()
                        continue
                    else:
                        self.clear_sign_file()
                        self.prev_processed_sign = self.current_sign
                        self.current_action = None
                        self.current_sign = None
                        continue



            frame = self.cam_thread.get_frame()
            if frame is None:
                self.rate.sleep()
                continue

            try:
                self.preprocess(frame)
                ang_z = self.infer_trt()
                twist.linear.x = self.default_linear_speed
                twist.angular.z = ang_z
            except Exception as e:
                rospy.logerr("推理或发布失败: %s", e)
                twist.linear.x = twist.angular.z = 0.0

            self.async_pub.send(twist)
            self.rate.sleep()

    def shutdown(self):
        self.cam_thread.stop()
        self.async_pub.stop()

if __name__ == '__main__':
    node = MotionController()
    wait_for_user_start()
    try:
        node.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        node.shutdown()
