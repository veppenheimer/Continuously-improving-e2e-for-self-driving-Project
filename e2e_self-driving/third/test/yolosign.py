#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
from yolo_msgs.msg import YoloSign
# string sign_type
# float32 area
# time stamp


def main():
    rospy.init_node('yolo_sign_publisher', anonymous=True)
    pub = rospy.Publisher('/yolo_sign', YoloSign, queue_size=10)
    rate = rospy.Rate(50)  # 10 Hz

    # 路径根据实际部署环境调整
    file_path = '/opt/nvidia/deepstream/deepstream-6.0/sources/DeepStream-Yolo/nvdsinfer_custom_impl_Yolo/yolosign.txt'
    #清空上次残留信息

    # 记录各类别上次发布时刻
    last_pub_time = {}
    filter_duration = rospy.Duration(5.0)  # 其他类别间隔 5 秒
    filter_duration1 = rospy.Duration(0.5)  # 玩偶检测间隔 0.5 秒
    rospy.loginfo('YOLO 识别发布节点启动，监听文件: %s', file_path)
    while not rospy.is_shutdown():
        try:
            with open(file_path, 'r') as f:
                sign_line = f.readline().strip()
                area_line = f.readline().strip()
        except Exception as e:
            rospy.logwarn('读取 YOLO 文件失败: %s', e)
            rate.sleep()
            continue

        # 仅当读取到有效内容时才考虑发布
        if sign_line and area_line:
            try:
                area = float(area_line)
                sign_type = sign_line
                now = rospy.Time.now()
                if sign_type == '0':
                    if area > 15000: #4700 0.2
                        sign_type = 'crossing_walk'
                elif sign_type == '1':
                    if area > 10:    #3500 0.2
                        sign_type = 'limit'
                elif sign_type == '2':
                    if area > 6500:
                        sign_type = 'remove_limit'
                elif sign_type == '3':
                    if area > 270:
                        sign_type = 'red'
                elif sign_type == '4':
                    sign_type = 'green'
                else:
                    pass
            except ValueError:
                rospy.logwarn('YOLO 文件内容格式错误: sign="%s", area="%s"', sign_line, area_line)
                rate.sleep()
                continue
            last_time = last_pub_time.get(sign_type)
            #人偶0.5 s 内发布一次
            if sign_type == 'person':
                if last_time is None or (now - last_time) > filter_duration1:
                    msg = YoloSign()
                    msg.sign_type = sign_type
                    msg.area = area
                    msg.stamp = now
                    pub.publish(msg)
                    last_pub_time[sign_type] = now
                    rospy.loginfo('发布 YOLO 标志: %s, 面积: %.2f', sign_type, area)
            # 其它类别过滤：5 秒内只发布一次
            else:
                if last_time is None or (now - last_time) > filter_duration:
                    msg = YoloSign()
                    msg.sign_type = sign_type
                    msg.area = area
                    msg.stamp = now
                    pub.publish(msg)
                    last_pub_time[sign_type] = now
                    rospy.loginfo('发布 YOLO 标志: %s, 面积: %.2f', sign_type, area)

        rate.sleep()

    rospy.loginfo('YOLO 发布节点退出')


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
