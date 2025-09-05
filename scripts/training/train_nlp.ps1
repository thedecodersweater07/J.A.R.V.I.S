# Config
$ProjectRoot = "C:\J.A.R.V.I.S"
$DataDir = Join-Path $ProjectRoot "data\data_sets\text"
$ModelDir = Join-Path $ProjectRoot "data\models\nlp"
$LogFile = Join-Path $ProjectRoot "nlp_training.log"

# Functie voor logging
function Log($message, $color="Green") {
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "[$timestamp] $message" -ForegroundColor $color
    Add-Content -Path $LogFile -Value "[$timestamp] $message"
}

function ErrorExit($message) {
    Log $message "Red"
    exit 1
}

# Create directories
New-Item -ItemType Directory -Path $ModelDir -Force | Out-Null

# Check data
$dataFile = Join-Path $DataDir "text_dataset.csv"
if (-Not (Test-Path $dataFile)) {
    ErrorExit "Dataset not found: $dataFile"
}

Log "Loading dataset..."
$csvData = Import-Csv $dataFile
Log "Loaded $($csvData.Count) rows from dataset."

# Simuleer modellen
$chatModel = $csvData | Select-Object -First 10
$embeddings = $csvData | ForEach-Object { $_.text.Length }  # simpele embedding = text length
$entityModel = $csvData | ForEach-Object { $_.text -split ' ' | Select-Object -First 1 } # eerste woord
$intentModel = $csvData | ForEach-Object { $_.text -match '\?' } # true if sentence is question

# Opslaan als XML (PowerShell equivalent van pickle)
$chatModel | Export-Clixml (Join-Path $ModelDir "chat_model.pkl")
$embeddings | Export-Clixml (Join-Path $ModelDir "embeddings.pkl")
$entityModel | Export-Clixml (Join-Path $ModelDir "entity_model.pkl")
$intentModel | Export-Clixml (Join-Path $ModelDir "intent_model.pkl")

Log "Models created with data from $dataFile"
