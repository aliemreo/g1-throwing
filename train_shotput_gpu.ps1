$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot
$Python = "C:\Users\25954\anaconda3\envs\mujo_torch\python.exe"

if (-not (Test-Path $Python)) {
    throw "mujo_torch Python was not found at $Python"
}

Set-Location $ProjectRoot
& $Python "$ProjectRoot\train\train_shotput_td3.py" `
    --device cuda @args
