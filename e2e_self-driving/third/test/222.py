import cv2

from camera_utils import open_mt2602u

if __name__ == "__main__":
    cap=open_mt2602u()
    while True:
        ret,frame = cap.read()
        if not ret:
            print("shibai")
            break
        cv2.imshow('shexiangtou',frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    cap.release()
    cv2.destroyAllwindows()
        