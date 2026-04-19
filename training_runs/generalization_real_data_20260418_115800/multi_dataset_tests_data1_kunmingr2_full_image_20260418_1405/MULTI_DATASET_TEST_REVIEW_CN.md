# 多数据集联调测试 Review

- 训练结果目录: `E:\桌面\项目\training_runs\generalization_real_data_20260418_115800`
- 测试输出目录: `E:\桌面\项目\training_runs\generalization_real_data_20260418_115800\multi_dataset_tests_data1_kunmingr2_full_image_20260418_1405`
- 推理设置: `全图 / HSV / Resize(120,160) / 无 ROI`
- 权重来源: 训练目录下最新 `best_ve2_generalization_*.pth`

## 汇总

| dataset | kind | images | folders ok/failed | Net_class MAE | Net_improve MAE | Net_regression MAE | status |
|---|---|---:|---:|---:|---:|---:|---|
| data1 | flat | 1743 |  | 0.267559 | 0.269615 | 0.219801 | OK |
| kunmingr2 | recursive | 1078 | 10/0 | 1.229754 | 1.174720 | 0.739452 | OK |

## 结果文件

- `data1` 输出目录: `E:\桌面\项目\training_runs\generalization_real_data_20260418_115800\multi_dataset_tests_data1_kunmingr2_full_image_20260418_1405\data1`
- `data1` JSON: `E:\桌面\项目\training_runs\generalization_real_data_20260418_115800\multi_dataset_tests_data1_kunmingr2_full_image_20260418_1405\data1\joint_dataset_compare.json`
- `data1` 曲线图: `E:\桌面\项目\training_runs\generalization_real_data_20260418_115800\multi_dataset_tests_data1_kunmingr2_full_image_20260418_1405\data1\joint_dataset_compare.png`
- `data1` 终端日志: `E:\桌面\项目\training_runs\generalization_real_data_20260418_115800\multi_dataset_tests_data1_kunmingr2_full_image_20260418_1405\data1\terminal.log`
- `kunmingr2` 输出目录: `E:\桌面\项目\training_runs\generalization_real_data_20260418_115800\multi_dataset_tests_data1_kunmingr2_full_image_20260418_1405\kunmingr2`
- `kunmingr2` 递归索引: `E:\桌面\项目\training_runs\generalization_real_data_20260418_115800\multi_dataset_tests_data1_kunmingr2_full_image_20260418_1405\kunmingr2\recursive_joint_index.json`
- `kunmingr2` 终端日志: `E:\桌面\项目\training_runs\generalization_real_data_20260418_115800\multi_dataset_tests_data1_kunmingr2_full_image_20260418_1405\kunmingr2\terminal.log`

