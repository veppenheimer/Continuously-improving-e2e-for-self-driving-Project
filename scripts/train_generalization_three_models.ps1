param(
    [string]$DatasetDir = 'E:\桌面\项目\dataset\real_steering_data_20260418_115032',
    [string]$RunName = ''
)

$ErrorActionPreference = 'Stop'
$python = 'E:\桌面\VeNet\ve_env\Scripts\python.exe'
$repo = 'E:\桌面\项目'
if ([string]::IsNullOrWhiteSpace($RunName)) {
    $timestamp = Get-Date -Format 'yyyyMMdd_HHmmss'
    $RunName = "generalization_real_data_$timestamp"
}
$runDir = Join-Path (Join-Path $repo 'training_runs') $RunName
New-Item -ItemType Directory -Force -Path $runDir | Out-Null

$branches = @('Net_class','Net_improve','Net_regression')
foreach ($branch in $branches) {
    New-Item -ItemType Directory -Force -Path (Join-Path $runDir $branch) | Out-Null
}

$commonEnv = @{
    'VENET_DATA_FOLDER' = $DatasetDir
    'VENET_BATCH_SIZE' = '16'
    'VENET_EARLY_STOP_PATIENCE' = '12'
    'VENET_EARLY_STOP_MIN_DELTA' = '0.0001'
    'VENET_WEIGHT_DECAY' = '0.0001'
    'VENET_GRAD_CLIP' = '3.0'
    'VENET_USE_PRETRAINED' = '1'
    'VENET_FREEZE_BACKBONE_EPOCHS' = '5'
    'VENET_BACKBONE_LR_FACTOR' = '0.1'
    'VENET_STYLE_MIX_RATIO' = '0.5,0.3,0.2'
    'VENET_PREPROCESS_COLOR_SPACE' = 'hsv'
    'PYTHONUNBUFFERED' = '1'
}

function Invoke-Train {
    param(
        [string]$Name,
        [string]$WorkDir,
        [hashtable]$ExtraEnv,
        [string]$LogPath
    )
    Write-Host "========== START $Name =========="
    Push-Location $WorkDir
    try {
        foreach ($kv in $commonEnv.GetEnumerator()) {
            Set-Item -Path ("Env:" + $kv.Key) -Value $kv.Value
        }
        foreach ($kv in $ExtraEnv.GetEnumerator()) {
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
    Write-Host "========== END $Name =========="
}

Invoke-Train -Name 'Net_class' -WorkDir (Join-Path $repo 'e2e_competition\Net_class') -ExtraEnv @{
    'VENET_OUTPUT_DIR' = (Join-Path $runDir 'Net_class')
    'VENET_LOG_DIR' = (Join-Path $runDir 'Net_class\tb')
    'VENET_SAVE_NAME' = 've2_generalization_class.pth'
    'VENET_BEST_SAVE_NAME' = 'best_ve2_generalization_class.pth'
    'VENET_EPOCHS' = '100'
    'VENET_LR' = '0.0001'
    'VENET_REG_LOSS_WEIGHT' = '2.0'
} -LogPath (Join-Path $runDir 'Net_class\terminal.log')

Invoke-Train -Name 'Net_improve' -WorkDir (Join-Path $repo 'e2e_competition\Net_improve') -ExtraEnv @{
    'VENET_OUTPUT_DIR' = (Join-Path $runDir 'Net_improve')
    'VENET_LOG_DIR' = (Join-Path $runDir 'Net_improve\tb')
    'VENET_SAVE_NAME' = 've2_generalization_improve.pth'
    'VENET_BEST_SAVE_NAME' = 'best_ve2_generalization_improve.pth'
    'VENET_EPOCHS' = '80'
    'VENET_LR' = '0.0001'
    'VENET_LABEL_SMOOTHING' = '0.05'
} -LogPath (Join-Path $runDir 'Net_improve\terminal.log')

Invoke-Train -Name 'Net_regression' -WorkDir (Join-Path $repo 'e2e_self-driving\Net') -ExtraEnv @{
    'VENET_OUTPUT_DIR' = (Join-Path $runDir 'Net_regression')
    'VENET_LOG_DIR' = (Join-Path $runDir 'Net_regression\tb')
    'VENET_SAVE_NAME' = 've2_generalization_regression.pth'
    'VENET_BEST_SAVE_NAME' = 'best_ve2_generalization_regression.pth'
    'VENET_EPOCHS' = '100'
    'VENET_LR' = '0.0001'
    'VENET_AUX_CLS_WEIGHT' = '0.3'
    'VENET_USE_WEIGHTED_SAMPLER' = '1'
} -LogPath (Join-Path $runDir 'Net_regression\terminal.log')

$reportScript = @"
from __future__ import annotations
import json
from pathlib import Path

run_dir = Path(r'''$runDir''')
dataset_dir = Path(r'''$DatasetDir''')
models = [
    ('Net_class', run_dir / 'Net_class' / 'training_summary.json'),
    ('Net_improve', run_dir / 'Net_improve' / 'training_summary.json'),
    ('Net_regression', run_dir / 'Net_regression' / 'training_summary.json'),
]
lines = []
lines.append('# 泛化优先三模型训练 Review')
lines.append('')
lines.append(f'- 运行目录: `{run_dir}`')
lines.append(f'- 数据集目录: `{dataset_dir}`')
lines.append('- 源数据集: `E:/桌面/data`')
lines.append('- 批大小: `16`')
lines.append('- 输入契约: `全图 -> HSV -> Resize(120,160) -> Tensor`')
lines.append('- ROI: `False`')
lines.append('- 训练增强: `50% clean / 30% moderate / 20% strong style`')
lines.append('- 选模指标: `val_stress_mae` 或 `val_stress_angle_mae`')
lines.append('')
summary_table = []
for name, path in models:
    lines.append(f'## {name}')
    if not path.is_file():
        lines.append(f'- summary: `missing: {path}`')
        lines.append('')
        continue
    summary = json.loads(path.read_text(encoding='utf-8'))
    keys = [
        'requestedEpochs', 'completedEpochs', 'bestEpoch', 'stoppedEpoch', 'earlyStopped',
        'modelSelectionMetric', 'steeringError',
        'finalTrainLoss', 'finalValLoss', 'finalTrainMAE', 'finalValMAE',
        'finalTrainAngleMAE', 'finalValAngleMAE', 'finalValStressAngleMAE', 'finalValStressMAE',
        'finalTestLoss', 'finalTestMAE', 'finalTestAcc', 'testBestAngleMAE', 'testBestAcc',
        'usedDedicatedTestSplit', 'pretrainedLoaded', 'usePretrained', 'freezeBackboneEpochs'
    ]
    for key in keys:
        if key in summary:
            lines.append(f'- {key}: `{summary[key]}`')
    lines.append('')
    summary_table.append({
        'model': name,
        'bestEpoch': summary.get('bestEpoch'),
        'steeringError': summary.get('steeringError'),
        'finalTestMAE': summary.get('finalTestMAE') or summary.get('finalTestAngleMAE') or summary.get('testBestAngleMAE'),
        'finalTestAcc': summary.get('finalTestAcc') or summary.get('testBestAcc'),
        'earlyStopped': summary.get('earlyStopped'),
        'completedEpochs': summary.get('completedEpochs'),
    })

lines.append('## 汇总表')
lines.append('')
lines.append('| model | bestEpoch | steeringError/valStress | testMAE | testAcc | earlyStopped | completedEpochs |')
lines.append('|---|---:|---:|---:|---:|---|---:|')
for row in summary_table:
    def fmt(value):
        if value is None:
            return ''
        if isinstance(value, float):
            return f'{value:.6f}'
        return str(value)
    lines.append(
        f"| {row['model']} | {fmt(row['bestEpoch'])} | {fmt(row['steeringError'])} | "
        f"{fmt(row['finalTestMAE'])} | {fmt(row['finalTestAcc'])} | {fmt(row['earlyStopped'])} | {fmt(row['completedEpochs'])} |"
    )
lines.append('')
lines.append('## 重要日志位置')
lines.append('')
for name, _ in models:
    lines.append(f'- `{name}`: `{run_dir / name / "terminal.log"}`')
lines.append('')
report_path = run_dir / 'TRAINING_REVIEW_CN.md'
report_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
print(report_path)
"@
& $python -c $reportScript | Tee-Object -FilePath (Join-Path $runDir 'report_generation.log')

Set-Content -Path (Join-Path $repo 'training_runs\latest_generalization_real_data_run.txt') -Value $runDir -Encoding UTF8
Write-Host "RUN_DIR=$runDir"
