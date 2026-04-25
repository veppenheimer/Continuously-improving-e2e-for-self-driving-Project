# 3-frame Temporal V2 网络逐层拆解与工作流说明

## 1. 名称与实际结构说明

当前项目中常说的 **3-frame MobileV2 / 3-frame Temporal V2**，这里的 `V2` 指的是项目里的第二版 MobileNet 回归主线，并不是 torchvision 的 `MobileNetV2` 网络。代码中的实际骨干为 **MobileNetV3-Small**，外层增加了轻量 3 帧时序适配器、多尺度特征融合和空间注意力池化。

- 模型变体：`mobilenet_v2_temporal3`
- 输入颜色空间：`hsv`
- 输入尺寸：`[144, 192]`
- ROI：`False`
- 帧数：`3`
- 帧间隔：`1`
- Legacy CNN 教师蒸馏：`teacherEnabled=false`，当前最终 3 帧模型未使用 Legacy CNN 作为教师
- 参数量：`1,019,968` 参数

## 2. 总体结构图

![3-frame Temporal V2 architecture](E:/桌面/项目/output/model_visualization/temporal3_mobilenet_v2/temporal3_mobilenet_v2_architecture.png)

## 3. 输入输出契约

模型输入不是单张 RGB 图，而是连续三帧图像经过同一预处理后拼接：

1. 读取 `t-2`、`t-1`、`t` 三帧。
2. 每帧转为 HSV。
3. 每帧 Resize 到 `144x192`。
4. 三帧按通道维拼接，形成 `[B, 9, 144, 192]`。
5. 模型输出 `[B, 1]`，表示连续转向角。

这种设计的意义是：在车辆高速行驶或弯道入口处，单帧图像可能无法判断接下来是否需要急转；三帧输入可以给模型提供短时运动趋势，降低单帧歧义。

## 4. 逐层 Shape Trace

完整表格已导出到：`E:\桌面\项目\output\model_visualization\temporal3_mobilenet_v2\temporal3_mobilenet_v2_layer_shapes.csv`

| 阶段 | 输入尺寸 | 输出尺寸 | 参数量 | 作用 |
|------|----------|----------|--------|------|
| `input` | `-` | `1x9x144x192` | `0` | 三帧 HSV 图像按通道拼接，保留短时时序上下文。 |
| `temporal_adapter` | `1x9x144x192` | `1x3x144x192` | `15` | 把 9 通道时序输入压回 3 通道，使后续预训练 MobileNet stem 可复用。 |
| `backbone.0` | `1x3x144x192` | `1x16x72x96` | `464` | MobileNetV3-Small backbone feature extraction |
| `backbone.1` | `1x16x72x96` | `1x16x36x48` | `744` | MobileNetV3-Small backbone feature extraction |
| `backbone.2` | `1x16x36x48` | `1x24x18x24` | `3,864` | MobileNetV3-Small backbone feature extraction |
| `backbone.3` | `1x24x18x24` | `1x24x18x24` | `5,416` | MobileNetV3-Small backbone feature extraction |
| `backbone.4` | `1x24x18x24` | `1x40x9x12` | `13,736` | MobileNetV3-Small backbone feature extraction |
| `backbone.5` | `1x40x9x12` | `1x40x9x12` | `57,264` | MobileNetV3-Small backbone feature extraction |
| `backbone.6` | `1x40x9x12` | `1x40x9x12` | `57,264` | MobileNetV3-Small backbone feature extraction |
| `backbone.7` | `1x40x9x12` | `1x48x9x12` | `21,968` | MobileNetV3-Small backbone feature extraction |
| `backbone.8` | `1x48x9x12` | `1x48x9x12` | `29,800` | 中间高分辨率特征，保留更多赛道线空间位置。 |
| `backbone.9` | `1x48x9x12` | `1x96x5x6` | `91,848` | MobileNetV3-Small backbone feature extraction |
| `backbone.10` | `1x96x5x6` | `1x96x5x6` | `294,096` | MobileNetV3-Small backbone feature extraction |
| `backbone.11` | `1x96x5x6` | `1x96x5x6` | `294,096` | MobileNetV3-Small backbone feature extraction |
| `backbone.12` | `1x96x5x6` | `1x576x5x6` | `56,448` | 最终语义特征，提供更强的全局/抽象表征。 |
| `mid_reduce` | `1x48x9x12` | `1x48x9x12` | `2,400` | 压缩/整理中层空间特征，保留车道线几何细节。 |
| `final_reduce` | `1x576x5x6` | `1x80x5x6` | `46,240` | 降低最终语义特征通道数，减少融合计算量。 |
| `upsample` | `1x80x5x6` | `1x80x9x12` | `0` | 把低分辨率语义特征对齐到中层空间网格，准备融合。 |
| `concat` | `1x48x9x12 + 1x80x9x12` | `1x128x9x12` | `0` | 合并空间几何信息和高层语义信息。 |
| `fuse` | `1x128x9x12` | `1x112x9x12` | `15,968` | 低成本融合多尺度特征，控制推理开销。 |
| `spatial_pool` | `1x112x9x12` | `1x112` | `3,221` | 让模型学习关注赛道关键区域，而不是简单平均所有位置。 |
| `reg_head` | `1x112` | `1x1` | `13,201` | 输出最终连续转向角，推理阶段只使用这个分支。 |
| `aux_head` | `1x112` | `1x11` | `11,915` | 训练期辅助约束角度词表连续性；推理阶段不参与输出。 |

## 5. 模块级工作流拆解

### 5.1 三帧输入与 Temporal Adapter

三帧 HSV 图像拼接后是 `[B, 9, 144, 192]`。如果直接修改 MobileNet 第一层为 9 通道，会破坏预训练 stem 的输入分布。当前模型采用更轻量的做法：先用 `Temporal Adapter` 将 9 通道压回 3 通道，再送入预训练 MobileNet。

`Temporal Adapter` 的核心是 `groups=3` 的 `1x1 Conv`：

- H 通道只融合三帧中的 H；
- S 通道只融合三帧中的 S；
- V 通道只融合三帧中的 V；
- 不在适配层混合不同颜色语义，避免引入过强扰动。

因此它的计算量很小，但可以学习“当前帧”和“历史帧”之间的融合权重。

### 5.2 MobileNetV3-Small 预训练骨干

适配后的 `[B, 3, 144, 192]` 输入进入 MobileNetV3-Small 的 `features`。模型会保留两类特征：

- `backbone[8]` 的中层特征：分辨率更高，更适合表达赛道线位置、车道边界和局部几何。
- `backbone[12]` 的最终特征：语义更强，但空间分辨率更低。

这样做是为了避免纯全局平均池化过早抹掉赛道线的空间位置。

### 5.3 多尺度特征融合

中层特征经过 `mid_reduce: 48 -> 48`，最终特征经过 `final_reduce: 576 -> 80`，然后最终特征被上采样到中层特征的空间尺寸。两者拼接后得到 `128` 通道，再通过深度可分离卷积融合为 `112` 通道。

这个模块的设计目标是：

- 保留空间几何信息；
- 引入高层语义上下文；
- 用深度可分离卷积控制推理成本。

### 5.4 空间注意力池化

融合后的特征不是直接 `AdaptiveAvgPool2d(1)`，而是经过空间注意力池化。它会对 `H*W` 个空间位置生成权重，然后做加权求和。

这样可以让模型更关注赛道线、弯道趋势、关键边界等位置，而不是把背景区域和赛道区域平均对待。

### 5.5 回归头与辅助头

回归头为 `112 -> 96 -> 24 -> 1`，输出连续转向角，是部署时真正使用的输出。

辅助头为 `112 -> 96 -> 11`，对应当前数据中的真实角度词表：

`[-1.62, -1.6, -1.58, -1.56, -1.5, 0.0, 1.5, 1.51, 1.511, 1.64, 1.72]`

辅助头只在训练时使用，用来约束特征对角度分布的表达；推理时不使用，不增加实际部署输出复杂度。

## 6. 推理阶段完整流程

1. 从视频流或图像序列中取连续三帧。
2. 对每帧执行同样的 HSV + Resize 预处理。
3. 拼接成 `[1, 9, 144, 192]`。
4. Temporal Adapter 压缩为 `[1, 3, 144, 192]`。
5. MobileNetV3-Small 提取中层和最终特征。
6. 多尺度融合得到兼顾空间几何与语义的特征图。
7. 空间注意力池化得到 `[1, 112]` 的全局特征向量。
8. 回归头输出最终连续转向角。

推理阶段仅使用回归头输出连续转向角；辅助头只服务训练约束。

## 7. 相比 Legacy CNN 的关键变化

| 维度 | Legacy CNN | 3-frame Temporal V2 |
|------|------------|---------------------|
| 输入 | 单帧图像 | 三帧时序输入 |
| 骨干 | 从零训练的小 CNN | ImageNet 预训练 MobileNetV3-Small |
| 空间信息 | 卷积后直接拉平/全连接 | 中层 + 最终特征融合 |
| 池化方式 | 无显式注意力 | 空间注意力加权池化 |
| 输出 | 单连续角度 | 单连续角度，训练期附加角度词表辅助头 |
| 部署成本 | 很低 | 小幅增加，但仍是轻量单模型推理 |
| 主要收益 | 同源数据拟合强 | 更好的风格鲁棒性和短时连续理解 |

## 8. 为什么这版结构适合当前任务

当前任务的核心不是识别任意开放道路，而是在固定赛道/近似固定轨迹下稳定输出转向角。真正困难点主要来自光照、显示风格、摄像头画面差异以及单帧急转弯前兆不足。因此当前结构的收益集中在三点：

- 预训练 MobileNet 提供更稳健的底层视觉表征；
- 三帧输入提供短时运动趋势，降低单帧标签跳变带来的不确定性；
- 多尺度融合和空间注意力尽量保留赛道线几何位置，提高对转向控制关键区域的敏感性。

