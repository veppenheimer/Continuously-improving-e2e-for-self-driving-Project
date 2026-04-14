#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import json
import rospy
from std_msgs.msg import Header
from yolo_msgs.msg import Detection
from collections import deque  # 用于数据缓冲

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst

label_list = []
data_buffer = deque(maxlen=5)  # 全局数据缓冲区

def load_labels(path):
    try:
        with open(path, 'r') as f:
            label_list.extend(line.strip() for line in f)
        rospy.loginfo("Loaded %d labels from %s", len(label_list), path)
    except Exception as e:
        rospy.logerr("Label loading failed: %s", e)

def parse_json_with_buffer(raw_data):
    try:
        # 容忍非UTF-8字符，用占位符替换
        decoded = raw_data.decode('utf-8', errors='replace')
        msg = json.loads(decoded)
        return msg
    except json.JSONDecodeError as e:
        rospy.logdebug("JSON incomplete, waiting more data...")
        return None
    except Exception as e:
        rospy.logwarn("JSON parse error: %s", e)
        return None

def on_new_sample(appsink, publisher):
    global data_buffer
    sample = appsink.emit("pull-sample")
    if not sample:
        return Gst.FlowReturn.OK

    buf = sample.get_buffer()
    ok, map_info = buf.map(Gst.MapFlags.READ)
    if not ok:
        return Gst.FlowReturn.OK

    data = bytes(map_info.data)  # 确保数据不可变
    buf.unmap(map_info)

    data_buffer.append(data)
    raw = b''.join(data_buffer)

    msg = parse_json_with_buffer(raw)
    if not msg:
        return Gst.FlowReturn.OK  # 等待更多数据

    data_buffer.clear()  # 解析成功后清空缓冲

    try:
        detections = msg.get('payload', {}).get('detection', [])
        for item in detections:
            det = Detection()
            det.header = Header(stamp=rospy.Time.now())
            det.class_name = item.get('class-name', 'unknown')
            bbox = item.get('bbox', {})
            xmin = bbox.get('xmin', 0)
            ymin = bbox.get('ymin', 0)
            xmax = bbox.get('xmax', 0)
            ymax = bbox.get('ymax', 0)
            det.area = max(0, xmax - xmin) * max(0, ymax - ymin)
            publisher.publish(det)
    except KeyError as e:
        rospy.logwarn("Missing key in JSON: %s", e)

    return Gst.FlowReturn.OK

def main():
    rospy.init_node('deepstream_yolo_publisher', anonymous=False)
    pub = rospy.Publisher('yolo_detections', Detection, queue_size=10)

    base = "/opt/nvidia/deepstream/deepstream-6.0/sources/DeepStream-Yolo"
    labels_file = os.path.join(base, "labels.txt")
    infer_config = os.path.join(base, "config_infer_primary_yoloV5.txt")
    msgconv_config = os.path.join(base, "msgconv_config.txt")

    load_labels(labels_file)

    Gst.init(None)

    # 摄像头配置（根据实际设备调整）
    src_device = "/dev/video0"
    src_caps = "video/x-raw,format=YUY2,width=640,height=480,framerate=30/1"

    # GStreamer 管道（关键修改点）
    pipeline_str = (
        f"v4l2src device={src_device} ! {src_caps} ! "
        "nvvideoconvert ! video/x-raw(memory:NVMM),format=NV12 ! "
        "mux.sink_0 "
        "nvstreammux name=mux batch-size=1 width=640 height=640 ! "  # 匹配YOLO输入尺寸
        f"nvinfer config-file-path={infer_config} ! "
        f"nvmsgconv config={msgconv_config} payload-type=0 ! "  # 确保payload-type=0
        "tee name=t ! "
        "queue ! nvegltransform ! nveglglessink sync=false t. ! "
        "queue ! appsink name=dsappsink emit-signals=true drop=true sync=false"
    )

    pipeline = Gst.parse_launch(pipeline_str)
    appsink = pipeline.get_by_name("dsappsink")
    appsink.connect("new-sample", on_new_sample, pub)

    pipeline.set_state(Gst.State.PLAYING)
    rospy.loginfo("DeepStream YOLO Publisher started.")
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down...")
    finally:
        pipeline.set_state(Gst.State.NULL)

if __name__ == "__main__":
    main()