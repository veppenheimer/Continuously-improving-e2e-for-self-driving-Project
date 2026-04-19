# 泛化优先重构说明

## 目标
本次改造把训练与推理默认契约统一为：

- 全图输入
- `HSV`
- `Resize(120, 160)`
- 默认 `ROI=False`

同时把三条模型链路都切到轻量预训练骨干，并引入 `val_stress` 作为主模型选择指标，优先提升陌生环境和强风格扰动下的稳定性。

## 共享模块
新增根目录共享模块：

- `steering_preprocess.py`
- `steering_models.py`
- `steering_augmentations.py`

### 共享预处理
`steering_preprocess.PreprocessConfig` 为统一输入契约：

- `color_space='hsv'`
- `input_size=(120, 160)`
- `use_roi=False`

以下链路现在默认复用这套契约：

- `e2e_self-driving/Net/train.py`
- `e2e_competition/Net_class/train.py`
- `e2e_competition/Net_improve/train.py`
- `e2e_self-driving/Net_Web/app/services/inference.py`
- `joint_infer_compare.py`
- `joint_infer_dataset_compare.py`
- `e2e_self-driving/trtCrun.py`

## 三条模型链路

### 1. 回归主线 `e2e_self-driving/Net`
模型改为轻量预训练骨干 + 双头：

- 主头：连续角度回归
- 辅助头：基于真实角度词表的辅助分类

训练目标：

- `SmoothL1(angle_pred, angle_gt)`
- `+ 0.3 * CrossEntropy(aux_logits, angle_vocab_idx)`

训练策略：

- 前 `5` 个 epoch 只训练 head
- 后续解冻 backbone
- backbone 学习率默认是 head 的 `0.1x`
- 继续保留按角度分布的 `WeightedRandomSampler`
- 检查点按 `val_stress_mae` 选择，而不是普通 `val_mae`

### 2. 分类残差主线 `e2e_competition/Net_class`
保留：

- 9 个硬编码原型
- grouped soft decode
- residual head

新增：

- 轻量预训练骨干
- 删除几何增强
- 类别频次加权 `CrossEntropy(label_smoothing=0.05)`
- `SmoothL1(residual)`
- 按 `val_stress_angle_mae` 选最优 checkpoint

### 3. 对照线 `e2e_competition/Net_improve`
该模型现在只承担参考线作用，策略与主线对齐：

- 轻量预训练骨干
- 删除几何增强
- 使用相同的全图 HSV 预处理
- 新增 `val_stress_angle_mae`
- 按 `val_stress_angle_mae` 选最优 checkpoint

## 数据增强
训练增强改为混合式风格增强，不包含任何会破坏转向语义的几何变换。

默认风格比例：

- `50%` clean
- `30%` moderate photometric
- `20%` strong style

增强内容只包含：

- brightness / contrast
- gamma
- HSV / RGB channel shift
- CLAHE
- local exposure / local shadow
- 轻微 blur / noise / compression / sharpen

验证与测试只做确定性预处理。

## val_stress
新增固定 `val_stress`：

- 使用 `val` 图像
- 套用强风格 photometric 增强
- 用固定 seed 保证每轮评估一致

训练摘要中新增：

- `modelSelectionMetric`
- `finalValStressMAE` 或 `finalValStressAngleMAE`
- `pretrainedLoaded`
- `preprocess`

## 兼容性
为了避免旧权重完全失效，以下模型文件保留了旧结构回退：

- `e2e_self-driving/Net/models.py`
- `e2e_self-driving/Net_Web/models.py`
- `e2e_competition/Net_class/models.py`
- `e2e_competition/Net_improve/models.py`

联调和 Web 推理会优先尝试新结构；若 checkpoint 是旧结构，会自动回退到 legacy 模型类加载。

## 默认环境变量
新的训练配置默认值：

- `VENET_USE_PRETRAINED=1`
- `VENET_FREEZE_BACKBONE_EPOCHS=5`
- `VENET_BACKBONE_LR_FACTOR=0.1`
- `VENET_AUX_CLS_WEIGHT=0.3` 仅回归主线使用
- `VENET_STYLE_MIX_RATIO=0.5,0.3,0.2`
- `VENET_PREPROCESS_COLOR_SPACE=hsv`

## 运行示例

### 回归主线
```powershell
cd E:\桌面\项目\e2e_self-driving\Net
$env:VENET_DATA_FOLDER = 'E:\桌面\data'
$env:VENET_BATCH_SIZE = '16'
$env:VENET_EPOCHS = '100'
$env:VENET_OUTPUT_DIR = 'E:\桌面\项目\training_runs\generalization_reg'
python train.py
```

### 分类残差主线
```powershell
cd E:\桌面\项目\e2e_competition\Net_class
$env:VENET_DATA_FOLDER = 'E:\桌面\data'
$env:VENET_BATCH_SIZE = '16'
$env:VENET_EPOCHS = '100'
$env:VENET_OUTPUT_DIR = 'E:\桌面\项目\training_runs\generalization_class'
python train.py
```

### 对照线 Net_improve
```powershell
cd E:\桌面\项目\e2e_competition\Net_improve
$env:VENET_DATA_FOLDER = 'E:\桌面\data'
$env:VENET_BATCH_SIZE = '16'
$env:VENET_EPOCHS = '80'
$env:VENET_OUTPUT_DIR = 'E:\桌面\项目\training_runs\generalization_improve'
python train.py
```

### 联调默认行为
联调脚本现在默认：

- 使用全图
- `ROI=False`

如果要启用诊断 ROI：

```powershell
E:\桌面\VeNet\ve_env\Scripts\python.exe E:\桌面\项目\joint_infer_dataset_compare.py E:\桌面\data --test-number 100 --enable-roi --roi-bottom-ratio 0.6
```

## 说明
本次重构的核心不是继续追随机切分精度，而是把“陌生环境鲁棒性”直接纳入训练与选模流程。后续如果继续迭代，建议优先看：

- `val_stress_mae`
- `data1` 强风格测试集表现
- `kunmingr2` 外部 benchmark 表现

而不是只盯普通 `val/test` 的随机切分指标。
