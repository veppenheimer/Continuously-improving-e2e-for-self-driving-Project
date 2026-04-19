# 回归模型安全数据增强与非均匀采样说明

本文档适用于 `e2e_self-driving/Net` 的端到端转向角回归训练链路。

## 设计目标

- 训练集使用在线数据增强，验证集和测试集只使用确定性预处理。
- 增强只覆盖亮度、对比度、轻微颜色扰动、轻微噪声、轻微模糊、轻微压缩等不改变空间语义的传统增强。
- 所有增强都不修改 steering angle 标签。
- 使用 `WeightedRandomSampler` 缓解直行样本多、转弯样本少导致的训练偏置。
- 保持当前工程历史输入习惯：OpenCV 读取 BGR 后，增强链在 RGB 上做颜色扰动，最后默认转为 HSV tensor 输入模型。

## 新增/修改文件

- `augmentations.py`：构建训练/验证/测试 transform。
- `sampler_utils.py`：按转向角分布计算采样权重，并构建 `WeightedRandomSampler`。
- `debug_utils.py`：增强样本可视化与角度/权重分布统计。
- `datasets.py`：兼容 Albumentations transform，并暴露 `dataset.angles` 给 sampler 使用。
- `train.py`：接入安全增强、非均匀采样、验证/测试确定性预处理。

## 安装依赖

如果当前环境尚未安装 Albumentations，请在项目使用的 Python 环境中安装：

```powershell
pip install -r .\e2e_self-driving\Net_Web\requirements.txt
```

或只安装训练所需新增依赖：

```powershell
pip install albumentations opencv-python matplotlib
```

## 默认训练行为

`train.py` 默认行为如下：

- `VENET_BATCH_SIZE=16`，更适合小显存环境。
- `VENET_DISABLE_TRAIN_AUG=0`，训练集启用在线增强。
- `VENET_USE_WEIGHTED_SAMPLER=1`，训练集启用非均匀采样。
- 验证/测试集只做 `Resize + RGB/HSV转换 + Normalize + ToTensorV2`。
- 如果存在 `test.txt`，训练结束后会用独立测试集输出 `finalTestMAE`。

示例：

```powershell
cd E:\桌面\项目\e2e_self-driving\Net
$env:VENET_DATA_FOLDER="E:\桌面\data"
$env:VENET_BATCH_SIZE="16"
$env:VENET_EPOCHS="100"
$env:VENET_OUTPUT_DIR="E:\桌面\项目\training_runs\net_reg_aug"
python train.py
```

## 常用环境变量

- `VENET_INPUT_HEIGHT` / `VENET_INPUT_WIDTH`：输入尺寸，默认 `120 / 160`。
- `VENET_AUG_COLOR_SPACE`：模型输入色彩空间，默认 `hsv`；除非整条链路都改成 RGB，否则建议保持默认。
- `VENET_DISABLE_TRAIN_AUG=1`：关闭训练随机增强，只保留确定性预处理。
- `VENET_USE_WEIGHTED_SAMPLER=0`：关闭非均匀采样，训练 DataLoader 会使用 `shuffle=True`。
- `VENET_SAMPLER_NUM_BINS`：角度分桶数量，默认 `9`。
- `VENET_SAMPLER_USE_ABS_ANGLE`：是否按 `|angle|` 分桶，默认开启。
- `VENET_SAMPLER_MAX_WEIGHT`：样本权重裁剪上限，默认 `8.0`。
- `VENET_SAMPLER_STRAIGHT_THRESHOLD`：直行样本阈值，默认 `0.08`。
- `VENET_SAMPLER_STRAIGHT_WEIGHT_SCALE`：直行样本额外降权比例，默认 `0.45`。

## 可视化调试

建议在正式训练前抽样检查增强结果：

```python
from augmentations import AugConfig, build_train_transforms
from datasets import AutoDriveDataset
from debug_utils import visualize_augmented_samples, summarize_angle_distribution
from sampler_utils import SamplerConfig

cfg = AugConfig(height=120, width=160, output_color_space="hsv")
dataset = AutoDriveDataset(r"E:\桌面\data", mode="train", transform=build_train_transforms(cfg))

visualize_augmented_samples(dataset, num_samples=8, save_path="aug_samples.png")
summarize_angle_distribution(dataset.angles, sampler_config=SamplerConfig(), save_path="angle_weights.png")
```

## 为什么适合当前小数据集回归任务

本方案把泛化增强集中在光照、颜色、画质和轻微相机噪声域偏移上，不改变道路结构、车道线位置和视角几何，因此标签可以保持原始连续转向角。非均匀采样让训练阶段更频繁看到低频转弯角度，减少模型被直行样本主导的倾向；同时通过 smoothing、权重裁剪和直行降权比例控制过采样强度，避免小数据集里少数样本被重复学习到过拟合。
