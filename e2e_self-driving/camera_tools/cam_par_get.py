import cv2
import time

# 先定义我们关心的“参数与 OpenCV 常量”的映射字典
PROP_MAPPING = {
    "Brightness":           cv2.CAP_PROP_BRIGHTNESS,         # AMCap“亮度”
    "Contrast":             cv2.CAP_PROP_CONTRAST,           # AMCap“对比度”
    "Hue":                  cv2.CAP_PROP_HUE,                # AMCap“色调”
    "Saturation":           cv2.CAP_PROP_SATURATION,         # AMCap“饱和度”
    "Sharpness":            cv2.CAP_PROP_SHARPNESS,          # AMCap“清晰度”
    "Gamma":                cv2.CAP_PROP_GAMMA,              # AMCap“伽玛”
    "WhiteBalance_U (Blue)":cv2.CAP_PROP_WHITE_BALANCE_BLUE_U,   # AMCap“白平衡”（U 通道）
    "WhiteBalance_V (Red)": cv2.CAP_PROP_WHITE_BALANCE_RED_V,    # AMCap“白平衡”（V 通道）
    "Backlight":            cv2.CAP_PROP_BACKLIGHT,          # AMCap“逆光补偿”
    "Gain":                 cv2.CAP_PROP_GAIN,               # AMCap“增益”
    "AutoExposure":         cv2.CAP_PROP_AUTO_EXPOSURE,      # AMCap 里关闭自动曝光后再给几十
    "Exposure":             cv2.CAP_PROP_EXPOSURE,           # AMCap“曝光”
}

# 你也可以根据需要补充更多 CAP_PROP，如 CAP_PROP_BRIGHTNESS、CAP_PROP_CONTRAST 等
# ----------------------------------------------------------------------------

def print_prop_range_and_value(cap, name, prop_id):
    """
    通过 set 和 get 探测该 prop_id
    以及是否支持、当前返回范围、默认值。
    """
    # 先 get 一下驱动里“默认值”
    default_val = cap.get(prop_id)
    print(f"[{name}] 当前默认值: {default_val}")

    # 如果是 Exposure，需要先关闭自动曝光，否则 set 无效
    if name == "Exposure":
        ok_ae = cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # 0.25 常表示“手动模式”
        ae_val = cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)
        print(f"  → 切换 AutoExposure 手动模式: 目标=0.25, 实际={ae_val}, 成功={ok_ae}")

    # 测试设置一个典型的“中间值”或“AMCap 给你的值”
    test_vals = {
        "Brightness": -9,
        "Contrast": 8,
        "Hue": 40,
        "Saturation": 121,
        "Sharpness": 6,
        "Gamma": 92,
        "WhiteBalance_U (Blue)": 6500,
        "WhiteBalance_V (Red)": 6500,
        "Backlight": 1,
        "Gain": 0,
        "Exposure": -6,
    }
    if name in test_vals:
        target = test_vals[name]
        ok = cap.set(prop_id, target)
        actual = cap.get(prop_id)
        print(f"  → Set {name}: 目标={target}, 实际={actual}, 成功={ok}")
    else:
        print(f"  → 没有提供测试值，跳过 set 测试")

    print("-" * 50)

# ----------------------------------------------------------------------------

def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # DirectShow 后端更靠谱地控制 UVC 摄像头
    if not cap.isOpened():
        print("❌ 无法打开摄像头，请确认设备索引是否正确或驱动是否正常。")
        return

    # 先让摄像头“热身”几百毫秒，以便 UVC 控制命令能生效
    time.sleep(0.3)

    for name, pid in PROP_MAPPING.items():
        print_prop_range_and_value(cap, name, pid)

    cap.release()

if __name__ == "__main__":
    main()
