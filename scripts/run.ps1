param (
    [Parameter(Mandatory=$true)]
    [ValidateSet("install", "train-quick", "train-full", "evaluate", "explain", "test", "api", "clean")]
    [string]$Action
)

$VENV_PY = ".\.venv\Scripts\python.exe"
$VENV_PIP = ".\.venv\Scripts\pip.exe"
$VENV_UVICORN = ".\.venv\Scripts\uvicorn.exe"
$VENV_PYTEST = ".\.venv\Scripts\pytest.exe"

$env:PYTHONPATH = "src"

switch ($Action) {
    "install" {
        Write-Host "Setting up virtual environment and installing dependencies..."
        python -m venv .venv
        & $VENV_PIP install --upgrade pip
        & $VENV_PIP install -r requirements.txt
    }
    "train-quick" {
        Write-Host "Running quick baseline training..."
        & $VENV_PY -m ticketlens.train_baseline --mode quick
        Write-Host "Running quick transformer training..."
        & $VENV_PY -m ticketlens.train_transformer --mode quick
    }
    "train-full" {
        Write-Host "Running full baseline training..."
        & $VENV_PY -m ticketlens.train_baseline --mode full
        Write-Host "Running full transformer training..."
        & $VENV_PY -m ticketlens.train_transformer --mode full
    }
    "evaluate" {
        Write-Host "Evaluating models on test set..."
        & $VENV_PY -m ticketlens.evaluate
    }
    "explain" {
        Write-Host "Generating explainability and bias report..."
        & $VENV_PY -m ticketlens.explain
    }
    "test" {
        Write-Host "Running test suite..."
        & $VENV_PYTEST tests/
    }
    "api" {
        Write-Host "Starting TicketLens FastAPI inference server..."
        & $VENV_UVICORN ticketlens.api:app --reload --host 0.0.0.0 --port 8000
    }
    "clean" {
        Write-Host "Cleaning cache files..."
        Get-ChildItem -Path . -Recurse -Filter "__pycache__" | Remove-Item -Recurse -Force
        Get-ChildItem -Path . -Recurse -Filter "*.pyc" | Remove-Item -Force
    }
}
