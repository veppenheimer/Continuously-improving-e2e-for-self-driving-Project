# example_script.py
import cv2
from camera_utils import open_mt2602u

def main():
    cap = open_mt2602u(index=0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("摄像头读取失败")
            break

        cv2.imshow("实时预览", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # 按 ESC 退出
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
