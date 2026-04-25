# 3-frame Temporal Steering Project

这是从当前工程中整理出来的独立 3-frame 转向角预测项目包，目标是把模型、数据处理、标签创建、训练、测试和可视化入口集中在一个文件夹内。

## 项目结构

- `train.py`：3-frame 模型训练入口，默认 `VENET_MODEL_VARIANT=temporal3`。
- `models.py` / `steering_models.py`：模型结构，核心是 `AutoDriveNetTemporal` + `RegressionTemporalSteeringNet`。
- `datasets.py`：支持 `num_frames=3`、`frame_stride=1` 的时序堆叠数据集。
- `steering_preprocess.py`：图像预处理、角度字典、标签编码工具。
- `steering_augmentations.py` / `augmentations.py`：训练和评估增强。
- `create_data_lists.py`：旧版简单列表生成脚本，保留作参考。
- `scripts/prepare_real_dataset.py`：推荐使用的数据整理与标签创建脚本。
- `scripts/prepare_formal_view_run_dataset.py`：按 train/val/test run 显式划分数据集。
- `scripts/augment_dataset_offline.py`：离线安全增强数据集。
- `scripts/compare_regression_models.py`：测试/对比 checkpoint，自动读取 `numFrames` 和 `frameStride`。
- `scripts/visualize_temporal3_model.py`：导出 3-frame 模型结构说明。
- `export_onnx.py` / `onnx_inference.py`：导出和测试 ONNX，已适配三帧 9 通道输入。
- `docs/`：历史复盘和 3-frame 网络工作流说明。
- `examples/`：可直接修改路径后运行的 PowerShell 示例。

## 数据命名约定

默认数据脚本会从文件名解析标签，推荐命名格式：

```text
000001_xxx_0.12.jpg
000002_xxx_0.10.jpg
000003_xxx_-0.05.jpg
```

其中：

- 文件名前缀数字用于 3-frame 时序堆叠，例如当前帧 `000003` 会尝试读取 `000001/000002/000003`。
- 文件名最后一个 `_` 后面的数字作为转向角标签。
- 缺少历史帧时，数据集会回退使用当前帧，保证序列开头也能训练和推理。

## 1. 创建数据集和标签

首次使用建议先安装依赖：

```powershell
cd E:\桌面\深度\temporal3_project
python -m pip install -r .\requirements.txt
```

推荐用 `prepare_real_dataset.py`，它会复制图片、解析角度、生成 `train_clean.txt` / `val_clean.txt` / `test_clean.txt` 等列表和 `dataset_summary.json`。

```powershell
cd E:\桌面\深度\temporal3_project
python .\scripts\prepare_real_dataset.py `
  --src E:\桌面\data `
  --dst-root .\dataset `
  --name temporal3_data `
  --label-shift 0
```

如果希望用未来帧标签训练，可把 `--label-shift` 改成正整数，例如 `1` 或 `2`。

## 2. 训练 3-frame 模型

```powershell
cd E:\桌面\深度\temporal3_project
$env:VENET_DATA_FOLDER = ".\dataset\temporal3_data"
$env:VENET_MODEL_VARIANT = "temporal3"
$env:VENET_NUM_FRAMES = "3"
$env:VENET_FRAME_STRIDE = "1"
$env:VENET_OUTPUT_DIR = ".\runs\temporal3"
$env:VENET_SAVE_NAME = "temporal3.pth"
$env:VENET_EPOCHS = "80"
python .\train.py
```

训练完成后默认输出：

- `runs/temporal3/temporal3.pth`
- `runs/temporal3/best_temporal3.pth`
- `runs/temporal3/training_summary.json`

## 3. 测试 checkpoint

```powershell
cd E:\桌面\深度\temporal3_project
python .\scripts\compare_regression_models.py `
  --model temporal3=.\runs\temporal3\best_temporal3.pth `
  --flat-dataset test=.\dataset\temporal3_data `
  --output-dir .\output\temporal3_eval
```

该脚本会根据 checkpoint 里的 `numFrames` / `frameStride` 自动用三帧输入推理。

## 4. 可视化模型结构

```powershell
cd E:\桌面\深度\temporal3_project
python .\scripts\visualize_temporal3_model.py `
  --checkpoint .\runs\temporal3\best_temporal3.pth `
  --output-dir .\output\model_visualization\temporal3_mobilenet_v2
```

## 5. 导出和测试 ONNX

```powershell
cd E:\桌面\深度\temporal3_project
python .\export_onnx.py --ckpt .\runs\temporal3\best_temporal3.pth --out .\output\temporal3.onnx
python .\onnx_inference.py --onnx .\output\temporal3.onnx --image .\dataset\temporal3_data\000003_xxx_0.10.jpg --num-frames 3 --frame-stride 1
```

## 当前默认值

- `modelVariant`: `temporal3`
- `numFrames`: `3`
- `frameStride`: `1`
- 默认训练数据目录：`dataset/temporal3_data`
- 默认输出目录：`runs/temporal3`
