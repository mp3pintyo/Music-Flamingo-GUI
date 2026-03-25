param(
    [switch]$Install,
    [string]$PythonCommand = "python"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $venvPython)) {
    if (-not (Get-Command $PythonCommand -ErrorAction SilentlyContinue)) {
        throw "Nem talalhato Python parancs. Hasznald a telepitett python.exe-t vagy aktivald az Anaconda/base kornyezetet elotte."
    }

    & $PythonCommand -m venv .venv
}

if ($Install) {
    & $venvPython -m pip install --upgrade pip
    Write-Host "Elobb telepits egy CUDA-s torch buildet, majd a tobbi fuggoseget a requirements.txt-bol."
    & $venvPython -m pip install -r requirements.txt
}

& $venvPython app.py
