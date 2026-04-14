#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import cv2
import os
import time
from geometry_msgs.msg import Twist

class DataCollector:
    def __init__(self):
        rospy.init_node('data_collector', anonymous=True)

        # 存储目录
        self.save_dir = "collected_data_keyboard"
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # 打开摄像头
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            rospy.logerr("无法打开摄像头")
            exit(0)

        # 当前角速度
        self.angular_z = 0.0
        self.index = 0

        # 订阅cmd_vel
        rospy.Subscriber("/cmd_vel", Twist, self.cmd_vel_callback)


    def cmd_vel_callback(self, msg):
        self.angular_z = msg.angular.z

    def run(self):
        rate = rospy.Rate(10)  # 10Hz
        while not rospy.is_shutdown():
            ret, frame = self.cap.read()
            if not ret:
                rospy.logwarn("无法读取图像")
                continue

            # 显示画面
            cv2.imshow("camera", frame)

            # 保存图像
            filename = "{:d}_{:.4f}.jpg".format(self.index, self.angular_z)
            save_path = os.path.join(self.save_dir, filename)
            cv2.imwrite(save_path, frame)
            rospy.loginfo("保存图片：{}".format(save_path))
            self.index += 1

            rate.sleep()

        self.cap.release()
        cv2.destroyAllWindows()

def main():
    collector = DataCollector()
    collector.run()

if __name__ == "__main__":
    main()
