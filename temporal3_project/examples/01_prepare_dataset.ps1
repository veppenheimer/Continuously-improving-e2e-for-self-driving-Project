$ProjectRoot = Split-Path -Parent $PSScriptRoot
$SourceData = "E:\桌面\data"

Set-Location $ProjectRoot
python .\scripts\prepare_real_dataset.py `
  --src $SourceData `
  --dst-root .\dataset `
  --name temporal3_data `
  --label-shift 0
