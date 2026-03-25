param(
    [switch]$Install,
    [string]$PythonCommand = "python"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"

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

    $output = & $venvPython -c $Script 2>&1
    [pscustomobject]@{
        ExitCode = $LASTEXITCODE
        Output   = (($output | Out-String).Trim())
    }
}

function Install-TorchForCuda {
    param([string]$IndexUrl)

    Write-Status "A megfelelő CUDA-s PyTorch telepítése indul..."
    Invoke-VenvPython -Arguments @("-m", "pip", "install", "--upgrade", "pip")
    Invoke-VenvPython -Arguments @("-m", "pip", "install", "torch", "torchvision", "torchaudio", "--index-url", $IndexUrl)
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
import importlib.util

if importlib.util.find_spec("torch") is None:
    print("MISSING")
else:
    print("READY")
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
from huggingface_hub import snapshot_download

from music_flamingo_gui.inference import DEFAULT_MODEL_ID

try:
    snapshot_download(repo_id=DEFAULT_MODEL_ID, local_files_only=True)
except Exception:
    print("MISSING")
else:
    print("READY")
'@

$modelState = Invoke-PythonCheck -Script $modelCheckScript
if ($modelState.ExitCode -ne 0 -or $modelState.Output -notmatch "READY") {
    Write-Status "A modell nincs helyi cache-ben, letoltes indul..."

    $modelDownloadScript = @'
from huggingface_hub import snapshot_download

from music_flamingo_gui.inference import DEFAULT_MODEL_ID

snapshot_download(repo_id=DEFAULT_MODEL_ID)
print("READY")
'@

    $downloadState = Invoke-PythonCheck -Script $modelDownloadScript
    if ($downloadState.ExitCode -ne 0 -or $downloadState.Output -notmatch "READY") {
        throw "A modell letoltese nem sikerult. Ha a repository gated, futtasd elotte a huggingface-cli login parancsot, es fogadd el a modell licencet a Hugging Face oldalan."
    }
}

Write-Status "Minden rendben van, inditom az alkalmazast..."
& $venvPython app.py
