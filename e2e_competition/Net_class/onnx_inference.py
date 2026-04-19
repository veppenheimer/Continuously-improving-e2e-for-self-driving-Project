import onnxruntime as ort
import numpy as np
import cv2

from steering_config import MAX_DELTA, STEERING_CLASSES, decode_output, split_output

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

output = ort_output[0][0]
logits, raw_delta = split_output(output)
cls = int(np.argmax(logits))
angle = float(np.asarray(decode_output(output)).reshape(-1)[0])
delta = float(np.tanh(raw_delta) * MAX_DELTA)

print(f"ONNX output shape: {np.array(output).shape}")
print(f"棰勬祴绫诲埆绱㈠紩: {cls}, 杩炵画杞悜瑙? {angle}, residual {delta:.6f} (妗ｄ綅琛? {STEERING_CLASSES})")
