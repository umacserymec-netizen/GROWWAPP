param(
    [int]$Port = 8501
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

$local = Join-Path $root ".venv\Scripts\python.exe"
$sibling = Join-Path $root "..\groww-mf-faq-assistant\.venv\Scripts\python.exe"

if (Test-Path $local) {
    $python = $local
} elseif (Test-Path $sibling) {
    $python = $sibling
} else {
    throw "No Python environment found. Create .venv and run: pip install -r requirements.txt"
}

$env:PYTHONIOENCODING = "utf-8"
Write-Host "Using $python"
& $python -m streamlit run app/streamlit_app.py --server.port $Port
