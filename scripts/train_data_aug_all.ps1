param()

$ErrorActionPreference = 'Stop'
$python = 'E:\桌面\VeNet\ve_env\Scripts\python.exe'
$repo = 'E:\桌面\项目'
$dataDir = 'E:\桌面\data_aug'
$timestamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$runDir = Join-Path $repo (Join-Path 'training_runs' "data_aug_$timestamp")

New-Item -ItemType Directory -Force -Path $runDir | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $runDir 'Net_class') | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $runDir 'Net_improve') | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $runDir 'Net_regression') | Out-Null

$splitter = @"
from __future__ import annotations
import random
from pathlib import Path

root = Path(r"E:\桌面\data_aug")
random.seed(256)
imgs = sorted([p for p in root.iterdir() if p.is_file() and p.suffix.lower() == '.jpg'])
if len(imgs) < 3:
    raise RuntimeError('at least 3 jpg files are required')
items = []
for p in imgs:
    angle = p.stem.rsplit('_', 1)[-1]
    items.append((p.as_posix(), angle))
random.shuffle(items)
n_total = len(items)
n_train = max(1, int(n_total * 0.7))
n_val = max(1, int(n_total * 0.15))
n_test = n_total - n_train - n_val
if n_test < 1:
    deficit = 1 - n_test
    take_from_train = min(deficit, max(0, n_train - 1))
    n_train -= take_from_train
    deficit -= take_from_train
    take_from_val = min(deficit, max(0, n_val - 1))
    n_val -= take_from_val
    deficit -= take_from_val
    if deficit > 0:
        raise RuntimeError('dataset too small to keep non-empty train/val/test')
    n_test = n_total - n_train - n_val
splits = {
    'train': items[:n_train],
    'val': items[n_train:n_train+n_val],
    'test': items[n_train+n_val:],
}
for mode, rows in splits.items():
    with (root / f'{mode}.txt').open('w', encoding='utf-8') as f:
        for img_path, angle in rows:
            f.write(f'{img_path} {angle}\n')
print(f"Generated splits for {n_total} images: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")
"@

& $python -c $splitter | Tee-Object -FilePath (Join-Path $runDir 'split_generation.log')

$commonEnv = @{
    'VENET_DATA_FOLDER' = $dataDir
    'VENET_BATCH_SIZE' = '16'
    'VENET_DISABLE_TRAIN_AUG' = '1'
    'VENET_EARLY_STOP_PATIENCE' = '10'
    'VENET_EARLY_STOP_MIN_DELTA' = '0.0001'
}

function Invoke-Train {
    param(
        [string]$WorkDir,
        [hashtable]$EnvMap,
        [string]$LogPath
    )
    Push-Location $WorkDir
    try {
        foreach ($kv in $EnvMap.GetEnumerator()) {
            Set-Item -Path ("Env:" + $kv.Key) -Value $kv.Value
        }
        & $python 'train.py' 2>&1 | Tee-Object -FilePath $LogPath
        if ($LASTEXITCODE -ne 0) {
            throw "Training failed in $WorkDir"
        }
    }
    finally {
        Pop-Location
    }
}

Invoke-Train -WorkDir (Join-Path $repo 'e2e_competition\Net_class') -EnvMap ($commonEnv + @{
    'VENET_OUTPUT_DIR' = (Join-Path $runDir 'Net_class')
    'VENET_LOG_DIR' = (Join-Path $runDir 'Net_class\tb')
    'VENET_SAVE_NAME' = 've2_data_aug_class.pth'
    'VENET_BEST_SAVE_NAME' = 'best_ve2_data_aug_class.pth'
    'VENET_EPOCHS' = '60'
    'VENET_LR' = '0.0001'
    'VENET_WEIGHT_DECAY' = '0.0001'
    'VENET_REG_LOSS_WEIGHT' = '2.0'
}) -LogPath (Join-Path $runDir 'Net_class\terminal.log')

Invoke-Train -WorkDir (Join-Path $repo 'e2e_competition\Net_improve') -EnvMap ($commonEnv + @{
    'VENET_OUTPUT_DIR' = (Join-Path $runDir 'Net_improve')
    'VENET_LOG_DIR' = (Join-Path $runDir 'Net_improve\tb')
    'VENET_SAVE_NAME' = 've2_data_aug_improve.pth'
    'VENET_BEST_SAVE_NAME' = 'best_ve2_data_aug_improve.pth'
    'VENET_EPOCHS' = '60'
    'VENET_LR' = '0.001'
    'VENET_WEIGHT_DECAY' = '0.0001'
    'VENET_LABEL_SMOOTHING' = '0.05'
}) -LogPath (Join-Path $runDir 'Net_improve\terminal.log')

Invoke-Train -WorkDir (Join-Path $repo 'e2e_self-driving\Net') -EnvMap ($commonEnv + @{
    'VENET_OUTPUT_DIR' = (Join-Path $runDir 'Net_regression')
    'VENET_LOG_DIR' = (Join-Path $runDir 'Net_regression\tb')
    'VENET_SAVE_NAME' = 've2_data_aug_regression.pth'
    'VENET_BEST_SAVE_NAME' = 'best_ve2_data_aug_regression.pth'
    'VENET_EPOCHS' = '80'
    'VENET_LR' = '0.0001'
    'VENET_WEIGHT_DECAY' = '0.0001'
    'VENET_USE_WEIGHTED_SAMPLER' = '1'
}) -LogPath (Join-Path $runDir 'Net_regression\terminal.log')

$reporter = @"
from __future__ import annotations
import json
from pathlib import Path

run_dir = Path(r"__RUN_DIR__")
models = [
    ('Net_class', run_dir / 'Net_class' / 'training_summary.json'),
    ('Net_improve', run_dir / 'Net_improve' / 'training_summary.json'),
    ('Net_regression', run_dir / 'Net_regression' / 'training_summary.json'),
]
lines = []
lines.append('# DATA_AUG 训练报告')
lines.append('')
lines.append(f'- 训练目录: `{run_dir}`')
lines.append('- 数据集: `E:/桌面/data_aug`')
lines.append('- 批大小: `16`')
lines.append('- 在线增强: `关闭`')
lines.append('- 划分方式: 简单随机 70/15/15')
lines.append('')
for name, path in models:
    summary = json.loads(path.read_text(encoding='utf-8'))
    lines.append(f'## {name}')
    for key in [
        'requestedEpochs', 'completedEpochs', 'bestEpoch', 'stoppedEpoch', 'earlyStopped',
        'steeringError', 'finalTrainLoss', 'finalValLoss', 'finalTrainMAE', 'finalValMAE',
        'finalTestLoss', 'finalTestMAE', 'finalTestAcc', 'finalValAcc', 'usedDedicatedTestSplit'
    ]:
        if key in summary:
            lines.append(f'- {key}: `{summary[key]}`')
    lines.append('')
(report_path := run_dir / 'DATA_AUG_TRAINING_REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
print(report_path)
"@
$reporter = $reporter.Replace('__RUN_DIR__', $runDir.Replace('\', '\\'))
& $python -c $reporter | Tee-Object -FilePath (Join-Path $runDir 'report_generation.log')

Set-Content -Path (Join-Path $repo 'training_runs\latest_data_aug_run.txt') -Value $runDir -Encoding UTF8
Write-Host "RUN_DIR=$runDir"
