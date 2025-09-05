# =============================================================================
# NLP Model Training Script
# Trains natural language processing models with progress monitoring
# =============================================================================

param(
    [switch]$Monitor,
    [switch]$Help
)

# Configuration
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
$DataDir = Join-Path $ProjectRoot "data"
$ModelDir = Join-Path $ProjectRoot "models"
$LogDir = Join-Path $ProjectRoot "logs"

# Training parameters
$ModelType = "nlp"
$BatchSize = 32
$Epochs = 10
$LearningRate = 0.001
$ValidationSplit = 0.2

# Color output functions
function Write-Log {
    param([string]$Message)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logMessage = "[$timestamp] $Message"
    Write-Host $logMessage -ForegroundColor Green
    Add-Content -Path (Join-Path $LogDir "nlp_training.log") -Value $logMessage
}

function Write-Error-Log {
    param([string]$Message)
    $errorMessage = "[ERROR] $Message"
    Write-Host $errorMessage -ForegroundColor Red
    Add-Content -Path (Join-Path $LogDir "nlp_training.log") -Value $errorMessage
    exit 1
}

# Create required directories
function New-RequiredDirectories {
    Write-Log "Creating required directories..."
    $directories = @(
        (Join-Path $ModelDir "nlp"),
        $LogDir,
        (Join-Path $DataDir "datasets\nlp_datasets")
    )
    
    foreach ($dir in $directories) {
        if (-not (Test-Path $dir)) {
            New-Item -ItemType Directory -Path $dir -Force | Out-Null
        }
    }
}

# Check Python dependencies
function Test-Dependencies {
    Write-Log "Checking Python dependencies..."
    try {
        python -c "import torch, transformers, numpy, pandas" 2>$null
        if ($LASTEXITCODE -ne 0) { throw }
    }
    catch {
        Write-Error-Log "Missing required Python packages. Install with: pip install torch transformers numpy pandas"
    }
}

# Data validation
function Test-TrainingData {
    Write-Log "Validating training data..."
    $nlpDataDir = Join-Path $DataDir "datasets\nlp_datasets"
    
    if (-not (Test-Path $nlpDataDir)) {
        Write-Error-Log "NLP datasets directory not found: $nlpDataDir"
    }
    
    $dataFiles = Get-ChildItem -Path $nlpDataDir -Include "*.txt", "*.json", "*.csv" -Recurse
    if ($dataFiles.Count -eq 0) {
        Write-Error-Log "No training data files found in nlp_datasets directory"
    }
    
    Write-Log "Found $($dataFiles.Count) training data files"
}

# Start training process
function Start-NLPTraining {
    Write-Log "Starting NLP model training..."
    Write-Log "Configuration:"
    Write-Log "  - Model Type: $ModelType"
    Write-Log "  - Batch Size: $BatchSize"
    Write-Log "  - Epochs: $Epochs"
    Write-Log "  - Learning Rate: $LearningRate"
    Write-Log "  - Validation Split: $ValidationSplit"
    
    $trainerPath = Join-Path $ProjectRoot "core\model_trainer.py"
    $nlpDataPath = Join-Path $DataDir "datasets\nlp_datasets"
    $nlpModelPath = Join-Path $ModelDir "nlp"
    $logFile = Join-Path $LogDir "nlp_training.log"
    
    $arguments = @(
        $trainerPath,
        "--model_type", $ModelType,
        "--data_dir", $nlpDataPath,
        "--output_dir", $nlpModelPath,
        "--batch_size", $BatchSize,
        "--epochs", $Epochs,
        "--learning_rate", $LearningRate,
        "--validation_split", $ValidationSplit,
        "--log_file", $logFile
    )
    
    try {
        python @arguments
        if ($LASTEXITCODE -ne 0) { throw }
    }
    catch {
        Write-Error-Log "Training failed. Check logs for details."
    }
}

# Monitor training progress
function Watch-TrainingProgress {
    $logFile = Join-Path $LogDir "nlp_training.log"
    Write-Host "Monitoring training progress..." -ForegroundColor Blue
    Write-Host "Press Ctrl+C to stop monitoring (training will continue)" -ForegroundColor Yellow
    
    Get-Content -Path $logFile -Wait | ForEach-Object {
        if ($_ -match "Epoch") {
            Write-Host $_ -ForegroundColor Green
        }
        elseif ($_ -match "Loss") {
            Write-Host $_ -ForegroundColor Blue
        }
        elseif ($_ -match "ERROR") {
            Write-Host $_ -ForegroundColor Red
        }
    }
}

# Main execution
function Invoke-Main {
    Write-Log "=== NLP Model Training Started ==="
    
    New-RequiredDirectories
    Test-Dependencies
    Test-TrainingData
    Start-NLPTraining
    
    Write-Log "=== NLP Model Training Completed ==="
    Write-Log "Model saved to: $(Join-Path $ModelDir 'nlp')"
    Write-Log "Logs available at: $(Join-Path $LogDir 'nlp_training.log')"
}

# Handle command line arguments
if ($Help) {
    Write-Host "Usage: .\train_nlp.ps1 [-Monitor] [-Help]"
    Write-Host "  -Monitor: Monitor training progress"
    Write-Host "  -Help:    Show this help message"
    exit 0
}
elseif ($Monitor) {
    Watch-TrainingProgress
}
else {
    Invoke-Main
}