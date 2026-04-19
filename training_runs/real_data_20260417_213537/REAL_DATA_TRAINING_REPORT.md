# ???????????

- ?????2026-04-17 22:24:52
- ????`E:\桌面\data`
- ?????`E:\桌面\项目\training_runs\real_data_20260417_213537`
- ?? batch size?`16`
- ?????`train=70%`?`val=15%`?`test=15%`

## ????

- `train.txt`: 1220
- `val.txt`: 261
- `test.txt`: 262

## ????

### Net_class ????

- ?????`E:\桌面\项目\training_runs\real_data_20260417_213537\Net_class`
- ???????`E:\桌面\项目\training_runs\real_data_20260417_213537\Net_class\terminal.log`
- `requestedEpochs`: `100`
- `completedEpochs`: `31`
- `bestEpoch`: `21`
- `stoppedEpoch`: `31`
- `earlyStopped`: `True`
- `finalTrainLoss`: `0.3846924974293005`
- `finalValLoss`: `0.3797136122696244`
- `steeringError`: `0.05152441454887162`
- `finalTrainAcc`: `0.9663934426229508`
- `finalValAcc`: `0.9540229885057471`
- `finalTrainAngleMAE`: `0.04938323964349559`
- `finalValAngleMAE`: `0.07861992984410675`
- `finalTestLoss`: `0.417170045030026`
- `finalTestAcc`: `0.9351145047267885`
- `finalTestAngleMAE`: `0.08701288791103218`
- `usedDedicatedTestSplit`: `True`

?????????

```text
epoch:22  CE_Loss:0.411761  Train_Acc:0.955738  Val_CE_Loss:0.380026  Val_Acc:0.961686  Train_Angle_MAE:0.059813  Val_Angle_MAE:0.063103  LR:0.000050
epoch:23  CE_Loss:0.404249  Train_Acc:0.953279  Val_CE_Loss:0.393691  Val_Acc:0.950192  Train_Angle_MAE:0.067273  Val_Angle_MAE:0.067737  LR:0.000050
epoch:24  CE_Loss:0.397192  Train_Acc:0.954918  Val_CE_Loss:0.376679  Val_Acc:0.965517  Train_Angle_MAE:0.067614  Val_Angle_MAE:0.059783  LR:0.000050
epoch:25  CE_Loss:0.412386  Train_Acc:0.950820  Val_CE_Loss:0.393092  Val_Acc:0.946360  Train_Angle_MAE:0.066225  Val_Angle_MAE:0.066282  LR:0.000025
epoch:26  CE_Loss:0.401427  Train_Acc:0.954098  Val_CE_Loss:0.390468  Val_Acc:0.950192  Train_Angle_MAE:0.063219  Val_Angle_MAE:0.071963  LR:0.000025
epoch:27  CE_Loss:0.401529  Train_Acc:0.953279  Val_CE_Loss:0.374721  Val_Acc:0.961686  Train_Angle_MAE:0.066104  Val_Angle_MAE:0.066168  LR:0.000025
epoch:28  CE_Loss:0.396209  Train_Acc:0.960656  Val_CE_Loss:0.385503  Val_Acc:0.954023  Train_Angle_MAE:0.066099  Val_Angle_MAE:0.066857  LR:0.000025
epoch:29  CE_Loss:0.393327  Train_Acc:0.959836  Val_CE_Loss:0.371683  Val_Acc:0.957854  Train_Angle_MAE:0.063850  Val_Angle_MAE:0.073199  LR:0.000013
epoch:30  CE_Loss:0.397763  Train_Acc:0.958197  Val_CE_Loss:0.374546  Val_Acc:0.961686  Train_Angle_MAE:0.077173  Val_Angle_MAE:0.060472  LR:0.000013
epoch:31  CE_Loss:0.383957  Train_Acc:0.966393  Val_CE_Loss:0.378967  Val_Acc:0.954023  Train_Angle_MAE:0.049383  Val_Angle_MAE:0.078620  LR:0.000013
EarlyStopping triggered at epoch 31 | best_epoch 21 | best_val_angle_mae 0.051524
TrainingSummary requested_epochs=100 completed_epochs=31 best_epoch=21 early_stopped=1 stopped_epoch=31
```

### Net_improve ??????

- ?????`E:\桌面\项目\training_runs\real_data_20260417_213537\Net_improve`
- ???????`E:\桌面\项目\training_runs\real_data_20260417_213537\Net_improve\terminal.log`
- `requestedEpochs`: `80`
- `completedEpochs`: `80`
- `bestEpoch`: `77`
- `stoppedEpoch`: `None`
- `earlyStopped`: `False`
- `finalTrainLoss`: `0.6518579150809616`
- `finalValLoss`: `0.647191399130328`
- `finalTrainAcc`: `0.8672131147540983`
- `finalValAcc`: `0.8735632186191749`
- `testBestLoss`: `0.5664241514133133`
- `testBestAcc`: `0.9198473291542694`
- `usedDedicatedTestSplit`: `True`

?????????

```text
��֤����ʧ�½�����������ģ�͵� E:\����\��Ŀ\training_runs\real_data_20260417_213537\Net_improve\best_ve2_real_improve.pth
Epoch 075 | TrainLoss 0.6455 | TrainAcc 0.8664 | ValLoss 0.6245 | ValAcc 0.8659 | LR 0.000250
��֤����ʧ�½�����������ģ�͵� E:\����\��Ŀ\training_runs\real_data_20260417_213537\Net_improve\best_ve2_real_improve.pth
Epoch 076 | TrainLoss 0.6618 | TrainAcc 0.8623 | ValLoss 0.6231 | ValAcc 0.8774 | LR 0.000250
��֤����ʧ�½�����������ģ�͵� E:\����\��Ŀ\training_runs\real_data_20260417_213537\Net_improve\best_ve2_real_improve.pth
Epoch 077 | TrainLoss 0.6594 | TrainAcc 0.8631 | ValLoss 0.5941 | ValAcc 0.8812 | LR 0.000250
��֤����ʧ�½�����������ģ�͵� E:\����\��Ŀ\training_runs\real_data_20260417_213537\Net_improve\best_ve2_real_improve.pth
Epoch 078 | TrainLoss 0.6599 | TrainAcc 0.8566 | ValLoss 0.5946 | ValAcc 0.8851 | LR 0.000250
Epoch 079 | TrainLoss 0.6303 | TrainAcc 0.8926 | ValLoss 0.6115 | ValAcc 0.8736 | LR 0.000250
Epoch 080 | TrainLoss 0.6519 | TrainAcc 0.8672 | ValLoss 0.6472 | ValAcc 0.8736 | LR 0.000250
���ģ�� TestLoss 0.5664 | TestAcc 0.9198
TrainingSummary requested_epochs=80 completed_epochs=80 best_epoch=77 early_stopped=0 stopped_epoch=0
```

### e2e_self-driving/Net ????

- ?????`E:\桌面\项目\training_runs\real_data_20260417_213537\Net_regression`
- ???????`E:\桌面\项目\training_runs\real_data_20260417_213537\Net_regression\terminal.log`
- `requestedEpochs`: `100`
- `completedEpochs`: `100`
- `bestEpoch`: `95`
- `stoppedEpoch`: `None`
- `earlyStopped`: `False`
- `finalTrainLoss`: `0.041843719972959784`
- `finalValLoss`: `0.06613534121146118`
- `steeringError`: `0.06404695280923925`
- `finalTestLoss`: `0.07568484107530071`
- `finalTestMAE`: `0.08835434142983596`
- `usedDedicatedTestSplit`: `True`

?????????

```text
epoch:90  MSE_Loss:0.040907  Val_MSE_Loss:0.058775  Train_MAE:0.078774  Val_MAE:0.072813  LR:0.000050
epoch:91  MSE_Loss:0.025092  Val_MSE_Loss:0.061452  Train_MAE:0.069059  Val_MAE:0.069614  LR:0.000050
epoch:92  MSE_Loss:0.031560  Val_MSE_Loss:0.065131  Train_MAE:0.073806  Val_MAE:0.077780  LR:0.000025
epoch:93  MSE_Loss:0.041996  Val_MSE_Loss:0.059005  Train_MAE:0.079344  Val_MAE:0.070183  LR:0.000025
epoch:94  MSE_Loss:0.034504  Val_MSE_Loss:0.064398  Train_MAE:0.072748  Val_MAE:0.067291  LR:0.000025
epoch:95  MSE_Loss:0.031692  Val_MSE_Loss:0.051091  Train_MAE:0.072956  Val_MAE:0.064047  LR:0.000025
epoch:96  MSE_Loss:0.024661  Val_MSE_Loss:0.053854  Train_MAE:0.068342  Val_MAE:0.066539  LR:0.000025
epoch:97  MSE_Loss:0.030667  Val_MSE_Loss:0.052396  Train_MAE:0.072088  Val_MAE:0.072607  LR:0.000025
epoch:98  MSE_Loss:0.033275  Val_MSE_Loss:0.057974  Train_MAE:0.071239  Val_MAE:0.068601  LR:0.000025
epoch:99  MSE_Loss:0.029849  Val_MSE_Loss:0.069226  Train_MAE:0.068197  Val_MAE:0.069729  LR:0.000025
epoch:100  MSE_Loss:0.041844  Val_MSE_Loss:0.066135  Train_MAE:0.076842  Val_MAE:0.075308  LR:0.000013
TrainingSummary requested_epochs=100 completed_epochs=100 best_epoch=95 early_stopped=0 stopped_epoch=0
```

## ????

- ????????? `terminal.log`?`training_summary.json`?TensorBoard `runs/`?latest checkpoint ? best checkpoint?
- `Net_class` ??? `Net` ? `steeringError` ? best validation MAE??????? test ???
- `Net_improve` ? `testBestAcc` / `testBestLoss` ?? best checkpoint ??? test split ?????