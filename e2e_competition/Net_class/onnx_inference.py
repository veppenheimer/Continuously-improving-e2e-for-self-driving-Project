import onnxruntime as ort
import numpy as np
import cv2

from steering_config import STEERING_CLASSES, class_to_angle

onnx_path = "0606x_static_quantized_model.onnx"
img_path = "image_test/2_0.0000.jpg"

img = cv2.imread(img_path)
img = cv2.resize(img, (160, 120))
img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

img = img.astype(np.float32) / 255.0
img = np.transpose(img, (2, 0, 1))
img = np.expand_dims(img, axis=0)

ort_session = ort.InferenceSession(onnx_path)

ort_inputs = {ort_session.get_inputs()[0].name: img}
ort_output = ort_session.run(None, ort_inputs)

logits = ort_output[0][0]
cls = int(np.argmax(logits))
angle = class_to_angle(cls)

print(f"ONNX logits shape: {np.array(logits).shape}")
print(f"预测类别索引: {cls}, 对应转向角: {angle} (档位表: {STEERING_CLASSES})")
