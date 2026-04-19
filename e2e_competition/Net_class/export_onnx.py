import cv2
import torch
import torch.onnx

from models import AutoDriveNet

print("Starting ONNX export...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

checkpoint_path = "./ve2.pth"
print(f"Loading checkpoint: {checkpoint_path}")

try:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = AutoDriveNet().to(device)
    model.load_state_dict(checkpoint["model"], strict=False)
    model.eval()
    print("Model loaded.")
except Exception as e:
    print(f"Failed to load model: {e}")
    raise SystemExit(1)

img_path = "./results/2.jpg"
print(f"Loading sample image: {img_path}")

try:
    img = cv2.imread(img_path)
    img = cv2.resize(img, (160, 120))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img = torch.from_numpy(img.copy()).float() / 255.0
    img = img.permute(2, 0, 1).unsqueeze(0).to(device)
    print("Sample image prepared.")
except Exception as e:
    print(f"Failed to prepare sample image: {e}")
    raise SystemExit(1)

onnx_path = "results/ve.onnx"
print(f"Exporting ONNX to: {onnx_path}")

try:
    torch.onnx.export(
        model,
        img,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    print("ONNX export finished. Output keeps 9 logits plus 1 residual head.")
except Exception as e:
    print(f"Failed to export ONNX: {e}")
