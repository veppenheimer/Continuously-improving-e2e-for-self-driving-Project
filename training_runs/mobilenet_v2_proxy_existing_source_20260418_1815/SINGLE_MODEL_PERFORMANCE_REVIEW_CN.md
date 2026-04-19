# 最新 MobileNet v2 单模型性能分析

## 1. 分析对象

- 模型权重: `E:\桌面\项目\training_runs\mobilenet_v2_proxy_existing_source_20260418_1815\best_ve2_mobilenet_v2_proxy_existing_source.pth`
- 训练目录: `E:\桌面\项目\training_runs\mobilenet_v2_proxy_existing_source_20260418_1815`
- 训练数据: `E:\桌面\项目\dataset\formal_view_proxy_existing_source_20260418_1815`
- 输入契约: `Full Image -> HSV -> Resize(144,192) -> Tensor`
- 教师蒸馏: `legacy CNN`, 仅 clean batch

## 2. 训练结果概览

来自 `training_summary.json` 的核心指标：

- `bestEpoch = 80`
- `earlyStopped = false`
- `modelSelectionMetric = val_stress_mae_fallback`
- `best val_stress MAE = 0.079761`
- `final val_clean MAE = 0.054234`
- `final test_clean MAE = 0.073537`
- `final train MAE = 0.036965`

训练过程观察：

- 前 `1-3` epoch 主要在 head 冻结阶段完成初步对齐
- 第 `4` epoch 开始全模型微调后，验证误差持续稳定下降
- 第 `75` epoch 学习率从 `1e-4` 降到 `5e-5`
- 最优点出现在最后一轮，说明这次训练没有出现明显过拟合回升

当前判断：

- 在同源 clean test 上，这个模型已经达到了比较强的精度
- `val_clean -> val_stress` 的上升幅度可控，说明“风格增强 + EMA + 预训练骨干”确实起到了作用
- 但这还不等于真正跨视角稳健，外部场景还需要继续看

## 3. 外部数据集评测

额外评测输出目录：

- `E:\桌面\项目\training_runs\mobilenet_v2_proxy_existing_source_20260418_1815\single_model_analysis_20260418_01`

### 3.1 clean / style / 外部场景总览

| 数据集 | 类型 | 图像数 | MAE |
|---|---|---:|---:|
| proxy test clean | 同源独立测试集 | 262 | 0.073537 |
| data1 | 同源强风格扰动集 | 1743 | 0.160019 |
| kunmingr2 | 外部场景递归测试集 | 1078 | 0.563371 |

误差放大量级：

- `data1 / clean_test = 2.18x`
- `kunmingr2 / clean_test = 7.66x`

结论：

- 对“同图但强风格变化”，模型已经具备一定鲁棒性，但误差仍会明显放大
- 对 `kunmingr2` 这类外部视角/画面分布变化，模型仍然存在较大退化
- 这说明当前主线已经明显更贴近“抗风格变化”目标，但真正决定上限的仍然是正式低俯角数据本身

## 4. data1 细分诊断

`data1` 为强风格扰动集，按真实角度分组后的主要结果如下：

| 真实角度 | 样本数 | MAE | 平均预测 |
|---|---:|---:|---:|
| -1.62 | 50 | 0.375503 | -1.245452 |
| -1.60 | 108 | 0.196078 | -1.412192 |
| -1.58 | 88 | 0.163132 | -1.425031 |
| -1.56 | 87 | 0.118301 | -1.458541 |
| -1.50 | 333 | 0.158428 | -1.362959 |
| 0.00 | 534 | 0.152127 | -0.022286 |
| 1.50 | 76 | 0.174887 | 1.330926 |
| 1.51 | 150 | 0.151481 | 1.381700 |
| 1.511 | 173 | 0.128748 | 1.408216 |
| 1.64 | 72 | 0.119697 | 1.523073 |
| 1.72 | 72 | 0.186328 | 1.534482 |

主要现象：

- 极端左转 `-1.62` 是最脆弱点，MAE 明显高于其他角度
- 极端右转 `1.72` 也有较明显退化
- 大部分正负转角的平均预测都在向 `0` 收缩
- 也就是说，强风格扰动下，模型会出现“转得不够狠”的保守偏差

这对实车控制的含义很直接：

- 在强风格变化下，模型不太容易突然给出离谱大角度
- 但更容易在本该大转的时候转向不足，最终表现为压线、出弯偏晚或贴边不够

## 5. kunmingr2 分目录诊断

`kunmingr2` 的总 MAE 为 `0.563371`，各子目录差异较大。

表现最好的一组：

| 子目录 | 图像数 | MAE |
|---|---:|---:|
| `kunmingr2/5/1` | 97 | 0.346056 |
| `kunmingr2/3/1` | 161 | 0.392282 |
| `kunmingr2/2/2` | 91 | 0.398837 |

表现最差的一组：

| 子目录 | 图像数 | MAE |
|---|---:|---:|
| `kunmingr2/1/1` | 108 | 0.808065 |
| `kunmingr2/1/2` | 107 | 0.781367 |
| `kunmingr2/4/1` | 83 | 0.702918 |

说明：

- 这个模型在 `kunmingr2` 上不是“全面崩”，而是对某些目录特别敏感
- 这通常意味着问题更接近“画面构成/背景占比/视角落差”而不是单纯亮度变化
- 其中 `1/1`、`1/2` 的高误差，非常值得后续回看对应图像内容，确认是否存在：
  - 上部背景占比过高
  - 赛道主导区域被压缩
  - 近处赛道线在画面中的相对位置变化较大

## 6. 当前模型的优点与短板

优点：

- clean test 精度已经不错，`0.0735` 说明同源控制能力较强
- `val_stress` 与 `data1` 结果证明模型已经比“只会记场景模板”的状态更稳
- 训练过程平滑，没有明显过拟合抖动
- 新主线推理契约已经固定，训练/推理口径一致

短板：

- 外部场景误差仍大，`kunmingr2` 说明旧视角差异依然会造成显著退化
- 强风格条件下，大转角存在明显“向中心收缩”的保守偏差
- 当前模型虽然比以前更稳，但距离“高精度正式部署”还差最后一段关键数据契约: 新低俯角正式数据

## 7. 结论

如果只看这一个最新模型，本轮可以下一个比较明确的结论：

- 它已经具备了不错的“同源 clean 精度”
- 对同图强风格变化也有一定鲁棒性
- 但真正限制它上线表现的，不再主要是训练链细节，而是训练数据视角和正式部署视角还没有完全统一

换句话说：

- 这条 `MobileNet v2` 主线本身已经值得继续投资源
- 但下一步最有收益的动作，不是继续在旧视角数据上死抠，而是尽快把正式低俯角数据接进来

## 8. 关键产物路径

- 训练摘要: `E:\桌面\项目\training_runs\mobilenet_v2_proxy_existing_source_20260418_1815\training_summary.json`
- 训练日志: `E:\桌面\项目\training_runs\mobilenet_v2_proxy_existing_source_20260418_1815\terminal.log`
- 单模型外部分析索引: `E:\桌面\项目\training_runs\mobilenet_v2_proxy_existing_source_20260418_1815\single_model_analysis_20260418_01\regression_compare_index.json`
- `data1` 图表: `E:\桌面\项目\training_runs\mobilenet_v2_proxy_existing_source_20260418_1815\single_model_analysis_20260418_01\data1\regression_compare.png`
- `kunmingr2` 目录级图表: `E:\桌面\项目\training_runs\mobilenet_v2_proxy_existing_source_20260418_1815\single_model_analysis_20260418_01\kunmingr2\recursive_regression_folder_mae.png`
