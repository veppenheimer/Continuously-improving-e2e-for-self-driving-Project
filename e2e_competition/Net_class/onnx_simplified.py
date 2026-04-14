from onnxsim import simplify
import onnx

onnx_path = "results/ve.onnx"
onnx_simplified_path = "results/ve_simplified.onnx"

onnx_model = onnx.load(onnx_path)

model_simp, check = simplify(onnx_model)

if check:
    print("✅ ONNX 模型优化成功！")
    onnx.save(model_simp, onnx_simplified_path)
else:
    print("❌ ONNX 模型优化失败，请检查模型结构！")
