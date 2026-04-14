# camera_utils.py
import cv2
import time

def open_mt2602u(index=0, delay=0.2):
    """
    打开 MT2602U 摄像头并一次性设置好所有参数，
    返回已经配置好的 VideoCapture 对象。
    """
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开摄像头 (index={index})")

    # 等待摄像头初始化
    time.sleep(delay)

    # 按照前面探测出的映射值逐项设置
    cap.set(cv2.CAP_PROP_BRIGHTNESS,      -9)    # 亮度
    cap.set(cv2.CAP_PROP_CONTRAST,         8)     # 对比度
    cap.set(cv2.CAP_PROP_HUE,              40)    # 色调
    cap.set(cv2.CAP_PROP_SATURATION,       121)   # 饱和度
    cap.set(cv2.CAP_PROP_SHARPNESS,        6)     # 清晰度/锐度
    cap.set(cv2.CAP_PROP_GAMMA,            92)    # 伽玛
    cap.set(cv2.CAP_PROP_WHITE_BALANCE_BLUE_U, 6500)  # 白平衡 (Blue U通道)
    cap.set(cv2.CAP_PROP_BACKLIGHT,        1)     # 逆光补偿
    cap.set(cv2.CAP_PROP_GAIN,             0)     # 增益

    # 关闭自动曝光并固定曝光值
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    cap.set(cv2.CAP_PROP_EXPOSURE,     -6)

    return cap
