# USER-OPERATED. Does not install a driver, download packages, or start training.
$ErrorActionPreference = 'Stop'
Write-Host 'Install the NVIDIA Windows production driver manually before continuing.'
Write-Host 'Do not install a Linux NVIDIA display driver inside WSL.'
wsl.exe --status
wsl.exe --list --verbose
Write-Host 'If WSL/Ubuntu is absent, the USER should run:'
Write-Host '  wsl.exe --install -d Ubuntu-24.04'
Write-Host '  wsl.exe --update'
Write-Host 'Keep project/data under /home in WSL, not /mnt/c.'
Write-Host 'Next: create a Python 3.11 environment in WSL and follow docs/runbook.md.'
