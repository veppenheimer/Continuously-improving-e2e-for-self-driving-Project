# 回归模型 A/B 对比 Review

- 训练结果目录: `E:\桌面\项目\training_runs\generalization_real_data_20260418_115800`
- 对比输出目录: `E:\桌面\项目\training_runs\generalization_real_data_20260418_115800\regression_ab_legacy_vs_mobilenet_20260418_1500`
- 推理设置: `全图 / HSV / Resize(120,160) / 无 ROI`
- 改进模型: `improved_mobilenet_aux`，权重 `Net_regression/best_ve2_generalization_regression.pth`
- 原始模型: `legacy_original_cnn`，权重 `Net_regression_legacy_original/best_ve2_legacy_original_regression.pth`

## 训练内指标

| model | structure | best epoch | completed epoch | early stop | val_stress_mae(best) | test_mae |
|---|---|---:|---:|---|---:|---:|
| improved_mobilenet_aux | MobileNetV3-Small pretrained + regression head + aux class head | 90 | 100 | False | 0.153702 | 0.113320 |
| legacy_original_cnn | original small CNN regression net | 60 | 72 | True, stopped at 72 | 0.118541 | 0.108537 |

训练内随机划分与 `val_stress` 上，legacy 原始 CNN 的数值更好；这说明它对当前数据分布的拟合能力不差，甚至更贴合当前源数据风格。

## 外部/扰动评测指标

| dataset | images | improved_mobilenet_aux MAE | legacy_original_cnn MAE | improved 相对 legacy |
|---|---:|---:|---:|---:|
| data1 强风格扰动集 | 1743 | 0.219801 | 0.193069 | -13.846% |
| kunmingr2 外部场景 | 1078 | 0.739452 | 1.580122 | +53.203% |

`data1` 是从源数据做强风格扰动得到的集合，因此它仍保留源数据的几何、场地和取景分布；legacy 在这里更好，说明原始 CNN 对“同场景强风格变化”仍有一定优势。

`kunmingr2` 是更接近真实陌生环境泛化的外部 benchmark；改进模型在这里把 MAE 从 `1.580122` 降到 `0.739452`，相对 legacy 降低约 `53.20%`，这是本轮结构改动最关键的收益。

## kunmingr2 子目录对比

| folder | images | improved MAE | legacy MAE | improved 相对 legacy |
|---|---:|---:|---:|---:|
| kunmingr2/1/1 | 108 | 1.104633 | 2.148971 | +48.60% |
| kunmingr2/1/2 | 107 | 1.125279 | 2.126687 | +47.09% |
| kunmingr2/2/1 | 91 | 0.873494 | 2.075771 | +57.92% |
| kunmingr2/2/2 | 91 | 0.787831 | 2.102554 | +62.53% |
| kunmingr2/3/1 | 161 | 0.442590 | 1.431780 | +69.09% |
| kunmingr2/3/2 | 159 | 0.535948 | 1.565677 | +65.77% |
| kunmingr2/4/1 | 83 | 0.711008 | 1.215778 | +41.52% |
| kunmingr2/4/2 | 83 | 0.745514 | 1.176006 | +36.61% |
| kunmingr2/5/1 | 97 | 0.632224 | 0.917468 | +31.09% |
| kunmingr2/5/2 | 98 | 0.689321 | 0.984975 | +30.02% |

改进模型在 `kunmingr2` 的 10 个子目录中全部优于 legacy，说明提升不是单个子目录偶然拉高，而是比较稳定的跨场景收益。

## 结论

- 如果只看当前源数据随机划分或源数据派生的 `data1`，legacy 原始 CNN 并没有输，甚至更好。
- 如果目标是陌生环境泛化，改进的 MobileNetV3-Small 预训练回归模型明显更值得作为主线。
- 这也解释了之前的矛盾现象：随机测试集表现不一定代表实车陌生场景表现，小 CNN 更容易记住当前场地模板，而预训练骨干在外部环境上保留了更强的视觉表征迁移能力。

## 关键文件

- legacy 训练目录: `E:\桌面\项目\training_runs\generalization_real_data_20260418_115800\Net_regression_legacy_original`
- legacy 最佳权重: `E:\桌面\项目\training_runs\generalization_real_data_20260418_115800\Net_regression_legacy_original\best_ve2_legacy_original_regression.pth`
- improved 最佳权重: `E:\桌面\项目\training_runs\generalization_real_data_20260418_115800\Net_regression\best_ve2_generalization_regression.pth`
- A/B 总索引: `E:\桌面\项目\training_runs\generalization_real_data_20260418_115800\regression_ab_legacy_vs_mobilenet_20260418_1500\regression_ab_test_index.json`
- data1 对比图: `E:\桌面\项目\training_runs\generalization_real_data_20260418_115800\regression_ab_legacy_vs_mobilenet_20260418_1500\data1\regression_ab_compare.png`
- kunmingr2 子目录对比图: `E:\桌面\项目\training_runs\generalization_real_data_20260418_115800\regression_ab_legacy_vs_mobilenet_20260418_1500\kunmingr2\recursive_regression_ab_folder_mae.png`
