# =====================================================================
# Audio Dataset Builder
# Converts raw audio files into a clean dataset (PowerShell + FFmpeg)
# =====================================================================

param(
    [string]$InputDir = "data\raw\audio",
    [string]$OutputDir = "data\data_sets\speech_dataset",
    [int]$SampleRate = 16000
)

function Write-Log {
    param([string]$Message)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "[$timestamp] $Message" -ForegroundColor Green
}

function New-RequiredDirectories {
    if (-not (Test-Path $OutputDir)) {
        New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
        Write-Log "Created dataset directory: $OutputDir"
    }
}

function Start-AudioDatasetBuild {
    Write-Log "Starting dataset conversion..."
    $files = Get-ChildItem -Path $InputDir -Include *.wav, *.mp3, *.flac -Recurse

    if ($files.Count -eq 0) {
        Write-Log "No audio files found in $InputDir"
        exit 1
    }

    $manifest = @()

    foreach ($file in $files) {
        $outFile = Join-Path $OutputDir ($file.BaseName + "_16k.wav")
        & ffmpeg -y -i $file.FullName -ar $SampleRate -ac 1 $outFile 2>$null

        if (Test-Path $outFile) {
            $manifest += [PSCustomObject]@{
                OriginalFile = $file.FullName
                DatasetFile  = $outFile
                SampleRate   = $SampleRate
            }
            Write-Log "Converted: $($file.Name) -> $($outFile)"
        }
    }

    # Save manifest to CSV
    $manifestFile = Join-Path $OutputDir "dataset_manifest.csv"
    $manifest | Export-Csv -Path $manifestFile -NoTypeInformation -Encoding UTF8
    Write-Log "Dataset build complete. Manifest saved to $manifestFile"
}

# Run
New-RequiredDirectories
Start-AudioDatasetBuild
