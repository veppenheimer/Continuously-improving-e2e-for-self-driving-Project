#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import cv2
import time
import os
from geometry_msgs.msg import Twist
import threading

def data_collector():
    rospy.init_node('data_collector_motion', anonymous=True)
    cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
    rate = rospy.Rate(50)

    # ✅ 优化1：禁用自动缓冲（减少帧延迟）
    cap = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 最小缓冲
    cap.set(cv2.CAP_PROP_FPS, 30)        # 设定帧率

    if not cap.isOpened():
        rospy.logerr("无法打开摄像头")
        return

    save_dir = "data_log"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    phases = [
###################
   # 路径规划（录制数据）
###################
        # 起点到人行道（左转）
     #   {"duration": 0.45, "linear": 0.5, "angular": 0.0},
     #    {"duration": 0.78, "linear": 0.5, "angular": 2.6},
     #    {"duration": 0.08, "linear": 0.5, "angular": 0.0},
     ##    {"duration": 2.54, "linear": 0.5, "angular": -1.51},
          {"duration": 0.15, "linear": 0.5, "angular": 0.0},
         {"duration": 0.91, "linear": 0.5, "angular": 1.8},
          {"duration": 0.49, "linear": 0.5, "angular": 0.0},

        # 起点到人行道（右转）
        # {"duration": 0.45, "linear": 0.5, "angular": 0.0},
        # {"duration": 0.79, "linear": 0.5, "angular": -2.6},
        # {"duration": 0.12, "linear": 0.5, "angular": 0.0},
        # {"duration": 2.6, "linear": 0.5, "angular": 1.47},
         {"duration": 0.15, "linear": 0.5, "angular": 0.0},
         {"duration": 0.86, "linear": 0.5, "angular": -2.2},
         {"duration": 0.49, "linear": 0.5, "angular": 0.0},


        # 人行道到环岛
        # {"duration": 2.3, "linear": 0.5, "angular": 0.0},
        # {"duration": 2.9, "linear": 0.5, "angular": 1.511},
        # {"duration": 0.14, "linear": 0.5, "angular": 0.0},
        # {"duration": 0.77, "linear": 0.5, "angular": 0.0},
       #
       #
       #  # 环岛到红灯
       #  # 3.1
       #  {"duration": 1.548, "linear": 0.5, "angular": 0},
       #  {"duration": 1.382, "linear": 0.5, "angular": -1.60},
       #  {"duration": 0.98, "linear": 0.5, "angular": 0.0},
       #  # 3.2
       #  {"duration": 2.4, "linear": 0.5, "angular": -1.50},
       #  # {"duration": 0.1, "linear": 0.5, "angular": 0.0},
       #  {"duration": 2.2, "linear": 0.5, "angular": -1.56},
       #  {"duration": 1.0, "linear": 0.5, "angular": 0.0},
       #
       #
       #  # 红灯到驶出红色匝道
       #  {"duration": 0.32, "linear": 0.5, "angular": 0.0},
       #  {"duration": 0.93, "linear":  0.5, "angular": -1.5},
       #  {"duration": 0.93, "linear":  0.5, "angular": 1.64},
       #  {"duration": 0.57, "linear": 0.5, "angular": 0.0},#时间可能需要修改
       #  {"duration": 0.94, "linear": 0.5, "angular": 1.5},
       #  {"duration": 0.94, "linear": 0.5, "angular": -1.5},

       #  # 最后弯道到终点
       # {"duration": 2.7, "linear": 0.5, "angular": 1.51},
       # {"duration": 0.65, "linear": 0.5, "angular": -1.62},
       # {"duration": 0.9 ,"linear": 0.5, "angular": 1.72},
      # {"duration": 1.2, "linear": 0.5, "angular": 0},
    ]

    twist_msg = Twist()
    pic_index = 0
    rospy.loginfo("开始数据采集")

    for phase in phases:
        phase_start = time.time()
        while not rospy.is_shutdown() and (time.time() - phase_start < phase["duration"]):
            twist_msg.linear.x = phase["linear"]
            twist_msg.angular.z = phase["angular"]
            cmd_pub.publish(twist_msg)

            ret, frame = cap.read()
            if not ret:
                rospy.logerr("采集图像失败")
                continue  # 跳过该帧

            # ✅ 优化2：使用线程异步保存图像，避免阻塞主线程
            threading.Thread(
                target=cv2.imwrite,
                args=(os.path.join(save_dir, f"{pic_index}_{twist_msg.angular.z:.4f}.jpg"), frame)
            ).start()

            pic_index += 1
            rate.sleep()  # 控制采集频率

    # 停止小车
    twist_msg.linear.x = 0
    twist_msg.angular.z = 0
    cmd_pub.publish(twist_msg)

    cap.release()
    rospy.loginfo("数据采集完成")

if __name__ == '__main__':
    try:
        data_collector()
    except rospy.ROSInterruptException:
        pass