# 泛化优先三模型训练 Review

- 运行目录: `E:\桌面\项目\training_runs\generalization_real_data_20260418_115800`
- 数据集目录: `E:\桌面\项目\dataset\real_steering_data_20260418_115032`
- 源数据集: `E:/桌面/data`
- 批大小: `16`
- 输入契约: `全图 -> HSV -> Resize(120,160) -> Tensor`
- ROI: `False`
- 训练增强: `50% clean / 30% moderate / 20% strong style`
- 选模指标: `val_stress_mae` 或 `val_stress_angle_mae`

## Net_class
- requestedEpochs: `100`
- completedEpochs: `36`
- bestEpoch: `24`
- stoppedEpoch: `36`
- earlyStopped: `True`
- modelSelectionMetric: `val_stress_mae`
- steeringError: `0.13553566718563237`
- finalTrainLoss: `0.66430462110238`
- finalValLoss: `0.6194414697387666`
- finalTrainAngleMAE: `0.08839747122504184`
- finalValAngleMAE: `0.09894524940639249`
- finalValStressAngleMAE: `0.1576053243526525`
- finalTestLoss: `0.704051842216317`
- finalTestAngleMAE: `0.11206118762137667`
- finalTestAcc: `0.9083969465648855`
- usedDedicatedTestSplit: `True`
- pretrainedLoaded: `True`
- usePretrained: `True`
- freezeBackboneEpochs: `5`

## Net_improve
- requestedEpochs: `80`
- completedEpochs: `34`
- bestEpoch: `22`
- stoppedEpoch: `34`
- earlyStopped: `True`
- modelSelectionMetric: `val_stress_mae`
- steeringError: `0.12091571526509134`
- finalTrainLoss: `0.6508948017339238`
- finalValLoss: `0.6118993909879663`
- finalTrainAngleMAE: `0.09870164335751143`
- finalValAngleMAE: `0.07907663405626669`
- finalValStressAngleMAE: `0.13087740118019425`
- testBestAngleMAE: `0.10345802416328255`
- testBestAcc: `0.916030534351145`
- usedDedicatedTestSplit: `True`
- pretrainedLoaded: `True`
- usePretrained: `True`
- freezeBackboneEpochs: `5`

## Net_regression
- requestedEpochs: `100`
- completedEpochs: `100`
- bestEpoch: `90`
- stoppedEpoch: `None`
- earlyStopped: `False`
- modelSelectionMetric: `val_stress_mae`
- steeringError: `0.1537022560663607`
- finalTrainLoss: `0.06354930428330038`
- finalValLoss: `0.06557160293169577`
- finalTrainMAE: `0.1551279185248203`
- finalValMAE: `0.10350695677758176`
- finalValStressMAE: `0.17458234252087002`
- finalTestLoss: `0.06529391692055546`
- finalTestMAE: `0.11332042560777592`
- usedDedicatedTestSplit: `True`
- pretrainedLoaded: `True`
- usePretrained: `True`
- freezeBackboneEpochs: `5`

## 汇总表

| model | bestEpoch | steeringError/valStress | testMAE | testAcc | earlyStopped | completedEpochs | pretrainedLoaded |
|---|---:|---:|---:|---:|---|---:|---|
| Net_class | 24 | 0.135536 | 0.112061 | 0.908397 | True | 36 | True |
| Net_improve | 22 | 0.120916 | 0.103458 | 0.916031 | True | 34 | True |
| Net_regression | 90 | 0.153702 | 0.113320 |  | False | 100 | True |

## 重要日志位置

- `Net_class`: `E:\桌面\项目\training_runs\generalization_real_data_20260418_115800\Net_class\terminal.log`
- `Net_improve`: `E:\桌面\项目\training_runs\generalization_real_data_20260418_115800\Net_improve\terminal.log`
- `Net_regression`: `E:\桌面\项目\training_runs\generalization_real_data_20260418_115800\Net_regression\terminal.log`

