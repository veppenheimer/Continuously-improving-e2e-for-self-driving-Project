$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$env:VENET_DATA_FOLDER = ".\dataset\temporal3_data"
$env:VENET_MODEL_VARIANT = "temporal3"
$env:VENET_NUM_FRAMES = "3"
$env:VENET_FRAME_STRIDE = "1"
$env:VENET_OUTPUT_DIR = ".\runs\temporal3"
$env:VENET_SAVE_NAME = "temporal3.pth"
$env:VENET_BEST_SAVE_NAME = "best_temporal3.pth"
$env:VENET_EPOCHS = "80"

python .\train.py
