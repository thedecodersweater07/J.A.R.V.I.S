$ErrorActionPreference = "Stop"

# Function to write colored output
function Write-ColorOutput {
    param([string]$Message, [string]$ForegroundColor = "White")
    Write-Host $Message -ForegroundColor $ForegroundColor
}

# Function to run a command and capture output
function Invoke-CommandWithOutput {
    param(
        [string]$Command,
        [string[]]$Arguments,
        [string]$WorkingDir = $PWD,
        [switch]$ShowOutput = $true
    )
    
    $processInfo = New-Object System.Diagnostics.ProcessStartInfo
    $processInfo.FileName = $Command
    $processInfo.Arguments = $Arguments -join ' '
    $processInfo.WorkingDirectory = $WorkingDir
    $processInfo.RedirectStandardOutput = $true
    $processInfo.RedirectStandardError = $true
    $processInfo.UseShellExecute = $false
    $processInfo.CreateNoWindow = $true
    
    $process = New-Object System.Diagnostics.Process
    $process.StartInfo = $processInfo
    
    $output = [System.Text.StringBuilder]::new()
    $errorOutput = [System.Text.StringBuilder]::new()
    
    $outputEvent = Register-ObjectEvent -InputObject $process -Action {
        $event.MessageData.AppendLine($EventArgs.Data) | Out-Null
        if ($ShowOutput) { Write-Host $EventArgs.Data }
    } -EventName 'OutputDataReceived' -MessageData $output
    
    $errorEvent = Register-ObjectEvent -InputObject $process -Action {
        $event.MessageData.AppendLine($EventArgs.Data) | Out-Null
        if ($ShowOutput) { Write-Host $EventArgs.Data -ForegroundColor Red }
    } -EventName 'ErrorDataReceived' -MessageData $errorOutput
    
    $process.Start() | Out-Null
    $process.BeginOutputReadLine()
    $process.BeginErrorReadLine()
    $process.WaitForExit()
    
    Unregister-Event -SourceIdentifier $outputEvent.Name
    Unregister-Event -SourceIdentifier $errorEvent.Name
    
    return @{
        ExitCode = $process.ExitCode
        Output = $output.ToString()
        Error = $errorOutput.ToString()
    }
}

# Find Python
$pythonExe = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $pythonExe) {
    $pythonExe = (Get-Command python3 -ErrorAction SilentlyContinue).Source
}
if (-not $pythonExe) {
    Write-ColorOutput "Python not found. Please install Python 3.7 or higher and ensure it's in your PATH." -ForegroundColor Red
    exit 1
}

Write-ColorOutput "Using Python at: $pythonExe" -ForegroundColor Green

# Get Python info
$pythonInfo = & $pythonExe -c "import sys, sysconfig; print(f'Version: {sys.version}'); print(f'Executable: {sys.executable}'); print(f'Include: {sysconfig.get_path("include")}'); print(f'Library: {sysconfig.get_config_var("LIBDIR")}')"
Write-ColorOutput $pythonInfo -ForegroundColor Cyan

# Check for required packages
$requiredPackages = @("cmake", "pybind11")
foreach ($pkg in $requiredPackages) {
    $installed = & $pythonExe -c "try: import $pkg; print('OK'); except: print('MISSING')" 2>&1
    Write-ColorOutput "Checking for $pkg : $installed" -ForegroundColor $(if ($installed -eq 'OK') { 'Green' } else { 'Red' })
    
    if ($installed -ne 'OK') {
        Write-ColorOutput "Installing $pkg..." -ForegroundColor Yellow
        & $pythonExe -m pip install $pkg --user
        if ($LASTEXITCODE -ne 0) {
            Write-ColorOutput "Failed to install $pkg" -ForegroundColor Red
            exit 1
        }
    }
}

# Set up build directories
$buildDir = "$PSScriptRoot\build"
$installDir = "$PSScriptRoot\install"

# Clean previous build if it exists
if (Test-Path $buildDir) {
    Write-ColorOutput "Cleaning previous build directory..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force $buildDir
}

# Create build directory
New-Item -ItemType Directory -Path $buildDir -Force | Out-Null

# Detect Visual Studio
$vsPath = "${env:ProgramFiles}\Microsoft Visual Studio\Installer\vswhere.exe"
if (Test-Path $vsPath) {
    $vsInfo = & $vsPath -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -format json | ConvertFrom-Json
    if ($vsInfo) {
        $vsVersion = $vsInfo[0].installationVersion.Split('.')[0]
        $generator = "Visual Studio ${vsVersion} 20$($vsVersion - 2)"
        Write-ColorOutput "Detected $generator at $($vsInfo[0].installationPath)" -ForegroundColor Green
    }
}

if (-not $generator) {
    $generator = "Ninja"
    Write-ColorOutput "Using $generator as fallback generator" -ForegroundColor Yellow
}

# Configure CMake
$cmakeArgs = @(
    "-G", "$generator",
    "-A", "x64",
    "-DCMAKE_BUILD_TYPE=Release",
    "-DCMAKE_INSTALL_PREFIX=$installDir",
    "-DPYTHON_EXECUTABLE=$pythonExe"
)

Write-ColorOutput "Configuring CMake with: $($cmakeArgs -join ' ')" -ForegroundColor Cyan

# Run CMake configuration
$result = Invoke-CommandWithOutput -Command "cmake" -Arguments $cmakeArgs -WorkingDir $buildDir -ShowOutput $true

if ($result.ExitCode -ne 0) {
    Write-ColorOutput "CMake configuration failed with exit code $($result.ExitCode)" -ForegroundColor Red
    exit $result.ExitCode
}

# Build the project
Write-ColorOutput "Building the project..." -ForegroundColor Green

if ($generator -match "Visual Studio") {
    $buildArgs = @("--build", ".", "--config", "Release", "--target", "INSTALL", "--", "/m:8")
} else {
    $buildArgs = @("--build", ".", "--config", "Release", "--target", "install", "--", "-j8")
}

$result = Invoke-CommandWithOutput -Command "cmake" -Arguments $buildArgs -WorkingDir $buildDir -ShowOutput $true

if ($result.ExitCode -ne 0) {
    Write-ColorOutput "Build failed with exit code $($result.ExitCode)" -ForegroundColor Red
    exit $result.ExitCode
}

Write-ColorOutput "Build completed successfully!" -ForegroundColor Green
Write-ColorOutput "Installed to: $installDir" -ForegroundColor Green
