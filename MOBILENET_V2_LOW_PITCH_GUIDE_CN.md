# MobileNet 回归主线 v2 低俯角正式视角说明

## 1. 目标定位

本轮优化不再追求“泛化到任意陌生地图”，而是明确服务于当前比赛的真实约束：

- 地图固定
- 路径基本固定
- 真正需要对抗的是亮度、曝光、白平衡、压缩、屏幕显示状态等风格变化
- 正式部署视角改为更低俯角，让赛道成为画面主体，而不是让模型替摄像头视角误差兜底

因此，当前主线只保留 `e2e_self-driving/Net` 下的 MobileNet 回归模型，并把它升级为更适合“同图高精度 + 风格稳健”的 `MobileNet v2`。

## 2. 正式输入契约

当前主线默认输入契约固定为：

- `Full Image -> HSV -> Resize(144, 192) -> Tensor`
- `ROI = False`

推荐正式摄像头视角目标：

- 近处赛道线 + 中距离弯道趋势占画面 `75% - 85%`
- 上方非赛道背景控制在 `20%` 左右，最多不超过 `25%`

说明：

- 旧版本回归模型、legacy CNN 仍可能使用 `120x160`
- 新脚本已经改为按 checkpoint 内的 `preprocess` 元数据自动推理，不再强制所有模型共用同一输入尺寸

## 3. 模型结构改动

文件：`E:\桌面\项目\steering_models.py`

新主线 `RegressionSteeringNet` 的关键变化：

- backbone 继续使用 `MobileNetV3-Small` 预训练，不增加重型主干
- 不再只依赖最后一层特征做全局平均池化
- 新增中间层高分辨率特征接入，用于保留赛道线位置与弯道趋势
- 通过以下轻量结构融合：
  - `1x1` 降维
  - 上采样对齐
  - 深度可分离 `3x3` 融合
  - 空间注意力加权池化
- 主输出仍为单一连续转向角回归值，不改变部署输出语义
- 辅助头仍保留，但仅用于训练期，不增加部署开销

当前冒烟统计下：

- `MobileNet v1` 参数量约 `1,080,300`
- `MobileNet v2` 参数量约 `1,019,953`

说明：新结构仍处于当前推理预算内，没有超出“轻量主线”的边界。

## 4. 训练策略改动

文件：`E:\桌面\项目\e2e_self-driving\Net\train.py`

当前默认训练策略：

- `batch_size = 16`
- `max_epochs = 80`
- `freeze_backbone_epochs = 3`
- 头部学习率 `1e-4`
- backbone 学习率通过 `VENET_BACKBONE_LR_FACTOR=0.2` 控制为头部的 `0.2x`
- `early_stop_patience = 10`
- `EMA decay = 0.999`

损失设计：

- 主损失：`SmoothL1(angle_pred, angle_gt)`
- 辅助损失：真实角度词表上的软标签分布损失
- 教师蒸馏：仅在 clean batch 上使用 legacy CNN 回归输出作为软目标

默认权重：

- `aux_cls_weight = 0.15`
- `teacher_weight = 0.20`

说明：

- 教师蒸馏只作用于 clean 样本，避免把 legacy 在极端风格扰动下的脆弱性直接蒸给学生
- 导出最佳权重时默认保存 EMA 权重

## 5. 数据增强策略改动

文件：`E:\桌面\项目\steering_augmentations.py`

当前训练增强不再使用几何增强，只保留与真实风格漂移相关的 photometric / image-quality 增强。

默认混合比例：

- `60% clean`
- `25% moderate style`
- `15% strong style`

增强内容包括：

- 亮度 / 对比度
- gamma
- 轻微 RGB shift
- 轻度 Hue / Saturation 扰动
- CLAHE
- 局部曝光增强 / 局部压暗
- 轻微噪声、压缩、模糊、锐化

特别说明：

- 已显式关闭 Albumentations 在线版本检查，避免训练日志中出现无关联网告警
- 原先 `Lambda` 带来的多进程兼容警告也已移除

## 6. 数据集组织改动

新增脚本：`E:\桌面\项目\scripts\prepare_formal_view_run_dataset.py`

作用：

- 按采集 run 明确切分 `train_clean / val_clean / test_clean / val_style_real`
- 避免同一路径相邻帧在 train/val/test 之间泄漏
- 为正式“低俯角数据集”建立单独的数据组织方式

支持输出：

- `train_clean.txt`
- `val_clean.txt`
- `test_clean.txt`
- `val_style_real.txt`
- 同时保留兼容的 `train.txt / val.txt / test.txt`

## 7. 推理与联调改动

### 7.1 Net_Web 推理

文件：`E:\桌面\项目\e2e_self-driving\Net_Web\app\services\inference.py`

改动：

- `load_checkpoint_model()` 会自动读取 checkpoint 中的 `preprocess` 元数据
- `predict_image()` 会按模型自带的预处理配置进行推理
- `load_image_path()` 改为 `np.fromfile + cv2.imdecode`，解决 Windows 中文路径下 `cv2.imread` 失效的问题

### 7.2 回归模型对比脚本

文件：`E:\桌面\项目\scripts\compare_regression_models.py`

改动：

- 支持多个模型同时对比，不再局限于两模型 A/B
- 每个模型按各自 checkpoint 的 `preprocess` 独立推理
- 可公平对比：
  - `legacy CNN`
  - 旧版 `MobileNet v1`
  - 新版 `MobileNet v2`
- 输出继续使用更适合 review 的聚合图：
  - 总体 MAE 柱状图
  - 绝对误差箱线图
  - 按真实角度分组 MAE
  - 按真实角度的平均预测值

## 8. 冒烟验证结果

本轮已完成的本地验证：

1. 语法检查
- 关键修改文件共 `16` 个，均已通过语法编译检查

2. 旧权重 / 新权重自动识别
- 旧 improved MobileNet checkpoint 可自动识别为 `AutoDriveNetV1`
- legacy CNN checkpoint 可自动识别为 `AutoDriveLegacyNet`
- 新 v2 checkpoint 可自动识别为 `AutoDriveNet`

3. 单轮训练冒烟
- 输出目录：`E:\桌面\项目\training_runs\debug_mobilenet_v2_smoke_20260418_02`
- 已成功生成：
  - `best_ve2_debug_mobilenet_v2_smoke.pth`
  - `ve2_debug_mobilenet_v2_smoke.pth`
  - `training_summary.json`
  - `smoke_train.log`

4. 三模型联调冒烟
- 输出目录：`E:\桌面\项目\training_runs\debug_compare_smoke_20260418`
- 已成功生成对比图、CSV、JSON、中文报告

## 9. 当前阶段结论

这次改动的核心，不是继续堆更多“常规增强”，而是把下列事情彻底对齐：

- 正式视角目标：低俯角、赛道占主体
- 训练输入契约：HSV + 144x192 + 全图
- 模型表征能力：保留中间层空间信息，不再过早全局平均
- 训练目标：高精度回归 + 连续角度软标签 + clean 蒸馏
- 评测方式：按模型自身输入契约公平比较

也就是说，模型现在不再被迫替旧摄像头视角和推理接线不一致的问题背锅。下一轮真正决定效果上限的关键，将是：

- 是否按新低俯角重新采集正式数据
- 是否按 run 切分出独立的 `val_style_real`
- 是否用这套新数据重新完整训练并评测 `MobileNet v2`

## 10. 建议的下一步

1. 先按正式低俯角重新采集 `train_clean / val_clean / test_clean / val_style_real` 四类 run
2. 用 `prepare_formal_view_run_dataset.py` 生成新正式数据集标签
3. 用当前 `train.py` 跑一轮完整 `MobileNet v2` 正式训练
4. 再用 `compare_regression_models.py` 对比：legacy / MobileNet v1 / MobileNet v2
5. 最终以 `val_style_real_mae` 和 `test_clean_mae` 共同决定是否替换部署主线
