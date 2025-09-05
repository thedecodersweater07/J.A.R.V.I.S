# =====================================================================
# TTS Model File Handler
# Moves only .pkl files to the model directory
# =====================================================================

param(
    [string]$InputDir = "data\data_sets\speech_dataset",
    [string]$OutputDir = "data\models\speech"
)

function Write-Log {
    param([string]$Message)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "[$timestamp] $Message" -ForegroundColor Green
}

function New-RequiredDirectories {
    if (-not (Test-Path $OutputDir)) {
        New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
        Write-Log "Created model directory: $OutputDir"
    }
}

function Move-PKLFiles {
    $pklFiles = Get-ChildItem -Path $InputDir -Filter *.pkl -Recurse

    if ($pklFiles.Count -eq 0) {
        Write-Log "No .pkl files found in $InputDir"
        return
    }

    foreach ($file in $pklFiles) {
        $destFile = Join-Path $OutputDir $file.Name
        Copy-Item -Path $file.FullName -Destination $destFile -Force
        Write-Log "Copied: $($file.Name) -> $destFile"
    }
}

# Run
New-RequiredDirectories
Move-PKLFiles
Write-Log "Only .pkl files have been moved successfully."
Write-Log "TTS Model File Handler completed."