$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

python .\scripts\compare_regression_models.py `
  --model temporal3=.\runs\temporal3\best_temporal3.pth `
  --flat-dataset test=.\dataset\temporal3_data `
  --output-dir .\output\temporal3_eval
