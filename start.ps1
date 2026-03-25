param(
    [switch]$Install,
    [string]$PythonCommand = "python"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
$defaultModelId = "nvidia/music-flamingo-think-2601-hf"

function Write-Status {
    param([string]$Message)

    Write-Host $Message -ForegroundColor Cyan
}

function Get-CudaTorchIndexUrl {
    while ($true) {
        $selection = (Read-Host "Melyik CUDA van a gépen? Írd be: 12 vagy 13").Trim()

        switch -Regex ($selection) {
            '^(12|12\.4|124|cu124)$' { return "https://download.pytorch.org/whl/cu124" }
            '^(13|13\.0|130|cu130)$' { return "https://download.pytorch.org/whl/cu130" }
            default {
                Write-Host "Kérlek 12 vagy 13 értéket adj meg." -ForegroundColor Yellow
            }
        }
    }
}

function Invoke-VenvPython {
    param(
        [string[]]$Arguments
    )

    & $venvPython @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "A Python parancs hibaval zarult."
    }
}

function Invoke-PythonCheck {
    param([string]$Script)

    $tempBase = [System.IO.Path]::Combine([System.IO.Path]::GetTempPath(), [System.Guid]::NewGuid().ToString())
    $scriptPath = "$tempBase.py"
    $stdoutPath = "$tempBase.stdout.txt"
    $stderrPath = "$tempBase.stderr.txt"

    try {
        Set-Content -Path $scriptPath -Value $Script -Encoding UTF8

        $startInfo = @{
            FilePath               = $venvPython
            ArgumentList           = @($scriptPath)
            NoNewWindow            = $true
            Wait                   = $true
            PassThru               = $true
            RedirectStandardOutput = $stdoutPath
            RedirectStandardError  = $stderrPath
        }

        $process = Start-Process @startInfo

        $stdout = if (Test-Path $stdoutPath) { Get-Content -Path $stdoutPath -Raw } else { "" }
        $stderr = if (Test-Path $stderrPath) { Get-Content -Path $stderrPath -Raw } else { "" }
        $combinedOutput = (@($stdout, $stderr) | Where-Object { $_ }) -join [Environment]::NewLine

        return [pscustomobject]@{
            ExitCode = $process.ExitCode
            Output   = $combinedOutput.Trim()
        }
    }
    finally {
        Remove-Item -Path $scriptPath, $stdoutPath, $stderrPath -ErrorAction SilentlyContinue
    }
}

function Install-TorchForCuda {
    param([string]$IndexUrl)

    Write-Status "CPU verzio cseréje vagy hianyzo PyTorch telepitese indul a megadott CUDA valtozatra..."
    Invoke-VenvPython -Arguments @("-m", "pip", "uninstall", "-y", "torch", "torchvision", "torchaudio")
    Invoke-VenvPython -Arguments @("-m", "pip", "install", "--upgrade", "pip")
    Invoke-VenvPython -Arguments @("-m", "pip", "install", "torch", "torchvision", "torchaudio", "--index-url", $IndexUrl)
}

function Invoke-VenvPythonScript {
    param([string]$Script)

    $scriptPath = [System.IO.Path]::Combine([System.IO.Path]::GetTempPath(), ([System.Guid]::NewGuid().ToString() + ".py"))

    try {
        Set-Content -Path $scriptPath -Value $Script -Encoding UTF8
        Invoke-VenvPython -Arguments @($scriptPath)
    }
    finally {
        Remove-Item -Path $scriptPath -ErrorAction SilentlyContinue
    }
}

if (-not (Test-Path $venvPython)) {
    if (-not (Get-Command $PythonCommand -ErrorAction SilentlyContinue)) {
        throw "Nem talalhato Python parancs. Hasznald a telepitett python.exe-t vagy aktivald az Anaconda/base kornyezetet elotte."
    }

    Write-Status "A virtualis kornyezet hianyzik, letrehozom..."
    & $PythonCommand -m venv .venv

    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $venvPython)) {
        throw "Nem sikerult letrehozni a virtualis kornyezetet."
    }
}

Write-Status "A virtualis kornyezet rendben van, ellenorzom a fuggosegeket..."

$torchCheckScript = @'
try:
    import torch
    if torch.cuda.is_available():
        print("READY")
    else:
        print("MISSING")
except ImportError:
    print("MISSING")
'@

$torchState = Invoke-PythonCheck -Script $torchCheckScript
if ($torchState.ExitCode -ne 0 -or $torchState.Output -notmatch "READY") {
    $torchIndexUrl = Get-CudaTorchIndexUrl
    Install-TorchForCuda -IndexUrl $torchIndexUrl
}

$dependencyCheckScript = @'
import importlib.util

missing = []

for module_name in [
    "accelerate",
    "bitsandbytes",
    "gradio",
    "huggingface_hub",
    "librosa",
    "soundfile",
    "transformers",
]:
    if importlib.util.find_spec(module_name) is None:
        missing.append(module_name)

if missing:
    print("MISSING")
    for item in missing:
        print(item)
    raise SystemExit(1)

print("READY")
'@
$dependencyState = Invoke-PythonCheck -Script $dependencyCheckScript
if ($dependencyState.ExitCode -ne 0 -or $dependencyState.Output -notmatch "READY") {
    Write-Status "Hianyzo fuggosegek talalhatok, telepitem a requirements.txt tartalmat..."
    Invoke-VenvPython -Arguments @("-m", "pip", "install", "-r", "requirements.txt")

    $dependencyState = Invoke-PythonCheck -Script $dependencyCheckScript
    if ($dependencyState.ExitCode -ne 0 -or $dependencyState.Output -notmatch "READY") {
        throw "A fuggosegek ellenorzese tovabbra sem sikerult. Rendszergazdai telepitesre vagy ujratesitesre lehet szukseg."
    }
}

if ($Install) {
    Write-Status "A -Install kapcsolo aktiv, frissitem a csomagokat is..."
    Invoke-VenvPython -Arguments @("-m", "pip", "install", "--upgrade", "pip")
    Invoke-VenvPython -Arguments @("-m", "pip", "install", "-r", "requirements.txt")
}

Write-Status "Ellenorzom, hogy a Music Flamingo modell elerheto-e lokalis cache-bol..."

$modelCheckScript = @'
from huggingface_hub import hf_hub_download

DEFAULT_MODEL_ID = "__MODEL_ID__"

try:
    hf_hub_download(repo_id=DEFAULT_MODEL_ID, filename="model.safetensors", local_files_only=True)
except Exception:
    print("MISSING")
else:
    print("READY")
'@

$modelCheckScript = $modelCheckScript.Replace('__MODEL_ID__', $defaultModelId)

$modelState = Invoke-PythonCheck -Script $modelCheckScript
if ($modelState.ExitCode -ne 0 -or $modelState.Output -notmatch "READY") {
    Write-Status "A modell nincs teljesen helyi cache-ben, letoltes/folytatas indul. Ez az elso alkalommal sok ideig is tarthat, mert a teljes csomag nagyjabol 16.5 GB..."
    Write-Status "(Megjegyzes: Ha korabban megszakadt a letoltes, akkor is 0%-tol indul a meroszam, de csak a HATRALEVO reszt fogja letolteni!)" -ForegroundColor Yellow

    $modelDownloadScript = @'
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
from huggingface_hub import snapshot_download

DEFAULT_MODEL_ID = "__MODEL_ID__"

snapshot_download(repo_id=DEFAULT_MODEL_ID)
print("READY")
'@

    $modelDownloadScript = $modelDownloadScript.Replace('__MODEL_ID__', $defaultModelId)

    try {
        Invoke-VenvPythonScript -Script $modelDownloadScript
    }
    catch {
        throw "A modell letoltese nem sikerult. A fenti Python hiba alapjan ellenorizd a halozatot, a szabad lemezteruletet es a Hugging Face hozzaferest. Ha a repo gated lenne, akkor kulon bejelentkezes vagy licencelfogadas is szukseges lehet."
    }
}

Write-Status "Minden rendben van, inditom az alkalmazast..."
& $venvPython app.py
