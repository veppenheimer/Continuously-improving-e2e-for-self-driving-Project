# 3 帧轻量时序版实验评审

## 结论

本轮已回退到 `Δ=0`，并在 `MobileNet v2` 主线上完成了 `3` 帧堆叠的轻量时序版本训练与对比评测。

结论是：

- `test_clean` 上，时序版优于当前单帧 `MobileNet v2`
- `data1` 强风格扰动集上，时序版显著优于当前单帧 `MobileNet v2`
- `kunmingr2` 外部场景上，时序版也小幅优于当前单帧 `MobileNet v2`

这说明在当前任务里，引入短时序信息是有效的，而且不是只提升同源测试，而是对风格变化和外部数据都带来了正收益。

## 本轮时序版设置

- 标签策略：`Δ=0`
- 模型：`mobilenet_v2_temporal3`
- 输入：`3` 帧堆叠，`frame_stride=1`
- 预处理：`HSV + 144x192 + Full Image`
- batch size：`16`
- teacher distillation：关闭
- backbone freeze：关闭
- EMA：`0.995`
- 加权采样：开启

## 训练结果

权重与日志：

- 最佳权重：`best_ve2_mobilenet_v2_temporal3_adapter_proxy_existing_source.pth`
- 训练摘要：`training_summary.json`
- 终端日志：`terminal.log`

关键指标：

| 模型 | test_clean MAE | val_clean MAE | best val_stress MAE |
|---|---:|---:|---:|
| 单帧 MobileNet v2 | 0.073537 | 0.054234 | 0.079761 |
| 3 帧 temporal v2 | 0.068950 | 0.055092 | 0.052761 |

说明：

- `test_clean`：时序版相对单帧版下降约 `6.2%`
- `best val_stress`：时序版相对单帧版下降约 `33.9%`

## 外部评测结果

对比模型：

- `legacy`
- `mobilenet_v2_single`
- `mobilenet_v2_temporal3`

结果：

| 数据集 | legacy | 单帧 MobileNet v2 | 3 帧 temporal v2 |
|---|---:|---:|---:|
| `data1` | 0.193069 | 0.160019 | 0.060090 |
| `kunmingr2` | 1.580122 | 0.563371 | 0.537475 |

说明：

- `data1` 上，时序版相对单帧版下降约 `62.4%`
- `kunmingr2` 上，时序版相对单帧版下降约 `4.6%`

## 对结果的判断

这轮结果说明：

- 之前“标签前移”不适合当前设定，但“短时序输入”是对症的
- 模型对强风格扰动的稳定性提升非常明显
- 对外部场景 `kunmingr2` 也有正收益，虽然幅度没有 `data1` 那么夸张
- 时序信息确实缓解了“单帧看起来很像，但控制需求突然变化”的一部分问题

## 本轮关键工程改动

- 将时序输入接入 `e2e_self-driving/Net/datasets.py`
- 支持 `3` 帧堆叠与 `frame_stride`
- 为小数据集增加懒加载图像缓存，避免三帧训练时重复磁盘读图拖慢训练
- 为 `MobileNet v2` 增加轻量时序适配器，而不是直接把 stem 改成重的 `9` 通道版本
- 推理/评测链路补齐对 temporal checkpoint 的识别

## 结果文件

- 对比报告：`compare_temporal3_vs_baselines_20260419_01/REGRESSION_COMPARE_REVIEW_CN.md`
- 对比索引：`compare_temporal3_vs_baselines_20260419_01/regression_compare_index.json`
- `data1` 图表：`compare_temporal3_vs_baselines_20260419_01/data1/regression_compare.png`
- `kunmingr2` 图表：`compare_temporal3_vs_baselines_20260419_01/kunmingr2/recursive_regression_folder_mae.png`

## 当前建议

基于本轮结果，`3` 帧轻量时序版值得保留为下一阶段主候选。

后续更值得继续做的是：

1. 在新低俯角正式视角上继续验证这套 temporal 方案
2. 检查 `kunmingr2` 各子目录中时序版收益最大的场景，判断它到底帮到了哪类弯道
3. 在不明显增加推理成本的前提下，尝试 `current frame + 2 个历史差分` 这类更强约束的轻时序输入
