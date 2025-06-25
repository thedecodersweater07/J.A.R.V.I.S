$ErrorActionPreference = "Stop"

# Function to write colored output
function Write-ColorOutput {
    param(
        [Parameter(Mandatory=$true)]
        [string]$Message,
        [string]$ForegroundColor = "White"
    )
    Write-Host $Message -ForegroundColor $ForegroundColor
}

# Function to find the latest Visual Studio installation
function Get-VisualStudioPath {
    $vswhere = "${env:ProgramFiles}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path $vswhere) {
        $vsInfo = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -format json | ConvertFrom-Json
        if ($vsInfo) {
            return $vsInfo[0].installationPath
        }
    }
    return $null
}

# Function to get the latest Visual Studio generator
function Get-VisualStudioGenerator {
    $vsPath = Get-VisualStudioPath
    if ($vsPath) {
        $versionFile = "$vsPath\VC\Auxiliary\Build\Microsoft.VCToolsVersion.default.txt"
        if (Test-Path $versionFile) {
            $vsVersion = (Get-Content $versionFile -ErrorAction SilentlyContinue).Split('.')[0]
            if ($vsVersion -eq '17') { return "Visual Studio 17 2022" }
            if ($vsVersion -eq '16') { return "Visual Studio 16 2019" }
            if ($vsVersion -eq '15') { return "Visual Studio 15 2017" }
        }
    }
    return "Ninja"  # Fallback to Ninja if VS not found
}

# Find Python executable and paths
$pythonExe = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $pythonExe) {
    $pythonExe = (Get-Command python3 -ErrorAction SilentlyContinue).Source
}
if (-not $pythonExe) {
    Write-Error "Python not found. Please install Python 3.7 or higher and ensure it's in your PATH."
    exit 1
}

Write-ColorOutput "Using Python at: $pythonExe" -ForegroundColor Green

# Get Python version
$pythonVersion = & $pythonExe -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"

# Set build directory
$buildDir = "$PSScriptRoot\build"
$installDir = "$PSScriptRoot\install"

# Clean previous build if it exists
if (Test-Path $buildDir) {
    Write-ColorOutput "Cleaning previous build directory..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force $buildDir
}

# Create build directory
New-Item -ItemType Directory -Path $buildDir -Force | Out-Null

# Check for available build tools
$generator = $null
$arch = "x64"  # Default to x64
$architecture = "x64"

# Check for Visual Studio 2022 first (preferred on Windows)
$vsPath = Get-VisualStudioPath
if ($vsPath -and (Test-Path "$vsPath\VC\Auxiliary\Build\vcvars64.bat")) {
    $generator = "Visual Studio 17 2022"
    Write-ColorOutput "Using Visual Studio 2022 generator" -ForegroundColor Green
    
    # Set up the environment
    Write-ColorOutput "Setting up Visual Studio environment..." -ForegroundColor Cyan
    cmd /c "`"$vsPath\VC\Auxiliary\Build\vcvars64.bat`" && set > `"$env:TEMP\vcvars.tmp`""
    Get-Content "$env:TEMP\vcvars.tmp" | ForEach-Object {
        if ($_ -match '^([^=]+)=(.*)') {
            $varName = $matches[1]
            $varValue = $matches[2]
            [System.Environment]::SetEnvironmentVariable($varName, $varValue, 'Process')
        }
    }
    Remove-Item "$env:TEMP\vcvars.tmp" -ErrorAction SilentlyContinue
    
    # Verify we have the compiler
    $clPath = (Get-Command "cl" -ErrorAction SilentlyContinue).Source
    if ($null -eq $clPath) {
        Write-ColorOutput "Failed to find MSVC compiler. Please ensure 'Desktop development with C++' is installed in Visual Studio." -ForegroundColor Red
        exit 1
    }
    Write-ColorOutput "Found MSVC compiler at: $clPath" -ForegroundColor Green
}
else {
    # Fall back to Ninja if available
    $ninjaPath = (Get-Command "ninja" -ErrorAction SilentlyContinue).Source
    if ($null -ne $ninjaPath) {
        $generator = "Ninja"
        Write-ColorOutput "Using Ninja build system at: $ninjaPath" -ForegroundColor Green
    } else {
        Write-ColorOutput "No suitable build system found. Please install one of the following:" -ForegroundColor Red
        Write-ColorOutput "1. Visual Studio 2022 with 'Desktop development with C++' workload" -ForegroundColor Yellow
        Write-ColorOutput "2. Ninja build system (https://ninja-build.org/)" -ForegroundColor Yellow
        exit 1
    }
}

Write-ColorOutput "Detected generator: $generator" -ForegroundColor Green
Write-ColorOutput "Target architecture: $arch" -ForegroundColor Green

# Configure CMake
$cmakeArgs = @(
    "-S", ".",
    "-B", $buildDir,
    "-G", $generator,
    "-DCMAKE_BUILD_TYPE=Release",
    "-DCMAKE_INSTALL_PREFIX=$installDir",
    "-DPYTHON_EXECUTABLE=$pythonExe"
)

# Add architecture for Visual Studio generators
if ($generator -match "Visual Studio") {
    $cmakeArgs += @("-A", $architecture)
}

# Get Python configuration
$pythonConfig = & $pythonExe -c @"
import sys
import sysconfig
import os

def get_python_lib():
    if sys.platform == 'win32':
        return os.path.join(sys.base_prefix, 'libs', f'python{sys.version_info.major}{sys.version_info.minor}.lib')
    return sysconfig.get_config_var('LIBRARY')

print(f'PYTHON_INCLUDE_DIR={sysconfig.get_path("include")}')
print(f'PYTHON_LIBRARY={get_python_lib()}')
print(f'PYTHON_VERSION={sys.version_info.major}.{sys.version_info.minor}')
print(f'PYTHON_SITE_PACKAGES={sysconfig.get_path("purelib")}')
"@ | ConvertFrom-String -Delimiter '=' -Property Name,Value

# Convert to hashtable
$pythonVars = @{}
$pythonConfig | ForEach-Object { $pythonVars[$_.Name] = $_.Value }

# Add Python paths to CMake arguments
$cmakeArgs += @(
    "-DPYTHON_INCLUDE_DIR=$($pythonVars.PYTHON_INCLUDE_DIR)",
    "-DPYTHON_LIBRARY=$($pythonVars.PYTHON_LIBRARY)",
    "-DPYTHON_VERSION=$($pythonVars.PYTHON_VERSION)",
    "-DPYTHON_SITE_PACKAGES=$($pythonVars.PYTHON_SITE_PACKAGES)"
)

# Add platform-specific arguments
if ($generator -match "Visual Studio") {
    $cmakeArgs += @("-Thost=x64")
}

Write-ColorOutput "Configuring CMake with arguments: $($cmakeArgs -join ' ')" -ForegroundColor Cyan

# Run CMake configuration
Push-Location $buildDir
$cmakeLog = "$PSScriptRoot\cmake_configure.log"

function Show-CMakeError {
    param([string]$logFile, [string]$errorFile, [int]$exitCode)
    
    Write-ColorOutput "=== CMake configuration failed with exit code $exitCode ===" -ForegroundColor Red
    
    if (Test-Path $logFile) {
        Write-ColorOutput "=== Last 20 lines of CMake output: ===" -ForegroundColor Yellow
        Get-Content $logFile -Tail 20 | ForEach-Object { Write-Host $_ }
    }
    
    if (Test-Path $errorFile) {
        Write-ColorOutput "=== Error output: ===" -ForegroundColor Red
        Get-Content $errorFile | ForEach-Object { Write-Host $_ -ForegroundColor Red }
    }
}

function Invoke-CMakeBuild {
    try {
        Write-ColorOutput "Running CMake configuration (log: $cmakeLog)..." -ForegroundColor Cyan
        
        # Prepare CMake arguments
        $cmakeArgsList = @(
            "-S", "$PSScriptRoot",
            "-B", $buildDir,
            "-G", $generator,
            "-DCMAKE_BUILD_TYPE=Release",
            "-DCMAKE_INSTALL_PREFIX=$installDir",
            "-DPYTHON_EXECUTABLE=$pythonExe"
        )
        
        # Add architecture for Visual Studio generators
        if ($generator -match "Visual Studio") {
            $cmakeArgsList += @("-A", $architecture)
        }
        
        # Add Python paths
        $pythonConfig = & $pythonExe -c @"
import sys
import sysconfig
import os

def get_python_lib():
    if sys.platform == 'win32':
        return os.path.join(sys.base_prefix, 'libs', f'python{sys.version_info.major}{sys.version_info.minor}.lib')
    return sysconfig.get_config_var('LIBRARY')

# Get Python include path
include_path = sysconfig.get_path('include')
print(f'PYTHON_INCLUDE_DIR={include_path}' if include_path else 'PYTHON_INCLUDE_DIR=')

# Get Python library
lib_path = get_python_lib()
print(f'PYTHON_LIBRARY={lib_path}' if lib_path else 'PYTHON_LIBRARY=')

# Get Python version
print(f'PYTHON_VERSION={sys.version_info.major}.{sys.version_info.minor}')

# Get site packages
site_packages = sysconfig.get_path('purelib')
print(f'PYTHON_SITE_PACKAGES={site_packages}' if site_packages else 'PYTHON_SITE_PACKAGES=')
"@ | ConvertFrom-String -Delimiter '=' -Property Name,Value | Where-Object { $_.Name -and $_.Value }

        # Convert to hashtable
        $pythonVars = @{}
        $pythonConfig | ForEach-Object { $pythonVars[$_.Name] = $_.Value }
        
        # Add Python paths to CMake arguments
        $cmakeArgsList += @(
            "-DPYTHON_INCLUDE_DIR=$($pythonVars.PYTHON_INCLUDE_DIR)",
            "-DPYTHON_LIBRARY=$($pythonVars.PYTHON_LIBRARY)",
            "-DPYTHON_VERSION=$($pythonVars.PYTHON_VERSION)",
            "-DPYTHON_SITE_PACKAGES=$($pythonVars.PYTHON_SITE_PACKAGES)"
        )
        
        # Run CMake and capture output
        $cmakeCommand = "cmake $($cmakeArgsList -join ' ')"
        Write-ColorOutput "Running: $cmakeCommand" -ForegroundColor Cyan
        
        $process = Start-Process -FilePath "cmake.exe" -ArgumentList $cmakeArgsList -NoNewWindow -PassThru -Wait `
            -RedirectStandardOutput $cmakeLog -RedirectStandardError "$cmakeLog.errors"
        
        if ($process.ExitCode -ne 0) {
            Show-CMakeError -logFile $cmakeLog -errorFile "$cmakeLog.errors" -exitCode $process.ExitCode
            exit $process.ExitCode
        }
        
        # Build the project
        Write-ColorOutput "Building the project..." -ForegroundColor Green
        $buildArgs = @("--build", ".", "--config", "Release", "--target", "INSTALL")
        Write-ColorOutput "Running: cmake $($buildArgs -join ' ')" -ForegroundColor Cyan
        
        $buildLog = "$PSScriptRoot\cmake_build.log"
        $buildProcess = Start-Process -FilePath "cmake.exe" -ArgumentList $buildArgs -NoNewWindow -PassThru -Wait -RedirectStandardOutput $buildLog -RedirectStandardError "$buildLog.errors"
        
        if ($buildProcess.ExitCode -ne 0) {
            Write-ColorOutput "=== Build failed with exit code $($buildProcess.ExitCode) ===" -ForegroundColor Red
            if (Test-Path $buildLog) {
                Write-ColorOutput "=== Last 20 lines of build output: ===" -ForegroundColor Yellow
                Get-Content $buildLog -Tail 20 | ForEach-Object { Write-Host $_ }
            }
            if (Test-Path "$buildLog.errors") {
                Write-ColorOutput "=== Build errors: ===" -ForegroundColor Red
                Get-Content "$buildLog.errors" | ForEach-Object { Write-Host $_ -ForegroundColor Red }
            }
            exit $buildProcess.ExitCode
        }
        
        Write-ColorOutput "Build completed successfully!" -ForegroundColor Green
        Write-ColorOutput "Installed to: $installDir" -ForegroundColor Green
        Pop-Location
    } catch {
        Write-ColorOutput "An error occurred during build: $_" -ForegroundColor Red
        Write-ColorOutput $_.ScriptStackTrace -ForegroundColor Red
        exit 1
    }
}
# Main script execution
try {
    # Call the build function
    Invoke-CMakeBuild
} catch {
    Write-Error "An error occurred: $_"
    Write-Error $_.ScriptStackTrace
    exit 1
} finally {
    Pop-Location
}

# Set Python environment variables for CMake
$pythonBasePath = Split-Path (Split-Path $pythonExe)
$env:Python3_ROOT_DIR = $pythonBasePath
$env:Python3_EXECUTABLE = $pythonExe

# Function to check if a command exists
function Test-CommandExists {
    param($command)
    $exists = $null -ne (Get-Command $command -ErrorAction SilentlyContinue)
    return $exists
}

# Function to check Python version
function Test-PythonVersion {
    $version = [version]$pythonVersionMajorMinor
    if ($version.Major -lt 3 -or ($version.Major -eq 3 -and $version.Minor -lt 7)) {
        Write-ColorOutput "Python 3.7 or higher is required. Found Python $pythonVersion" -ForegroundColor Red
        return $false
    }
    Write-ColorOutput "Found Python $pythonVersion" -ForegroundColor Green
    return $true
}

# Function to ensure pip is installed and up to date
function Initialize-Pip {
    Write-ColorOutput "Checking pip installation..." -ForegroundColor Cyan
    python -m ensurepip --upgrade
    python -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) {
        Write-ColorOutput "Failed to initialize pip. Please install pip manually." -ForegroundColor Red
        exit 1
    }
}

# Function to install build dependencies
function Install-BuildDependencies {
    Write-ColorOutput "Installing build dependencies..." -ForegroundColor Cyan
    python -m pip install cmake ninja
    if ($LASTEXITCODE -ne 0) {
        Write-ColorOutput "Failed to install build dependencies." -ForegroundColor Red
        exit 1
    }
}

# Function to install pybind11
function Install-PyBind11 {
    Write-ColorOutput "Installing pybind11..." -ForegroundColor Cyan
    python -m pip install pybind11
    if ($LASTEXITCODE -ne 0) {
        Write-ColorOutput "Failed to install pybind11. Please install it manually with: pip install pybind11" -ForegroundColor Red
        exit 1
    }
}

# Main script starts here
Write-ColorOutput "=== JARVIS Build System ===" -ForegroundColor Magenta

# Check Python installation
if (-not (Test-CommandExists "python")) {
    Write-ColorOutput "Python is not found in PATH. Please install Python 3.7 or higher and add it to PATH." -ForegroundColor Red
    exit 1
}

# Check Python version
if (-not (Test-PythonVersion)) {
    exit 1
}

# Initialize pip
Initialize-Pip

# Install build dependencies
Install-BuildDependencies

# Install pybind11 if not found
$pybind11Path = $null
try {
    $pybind11Path = python -c "import pybind11; print(pybind11.get_include())" 2>$null
} catch {
    Write-ColorOutput "pybind11 not found. Installing..." -ForegroundColor Yellow
    python -m pip install pybind11 --user
    try {
        $pybind11Path = python -c "import pybind11; print(pybind11.get_include())" 2>$null
    } catch {
        Write-ColorOutput "Failed to install pybind11. Please install it manually with: pip install pybind11" -ForegroundColor Red
        exit 1
    }
}

if (-not $pybind11Path) {
    Write-ColorOutput "pybind11 not found. Please install it with: pip install pybind11" -ForegroundColor Red
    exit 1
}

Write-ColorOutput "Found pybind11 at: $pybind11Path" -ForegroundColor Green

# Get Python paths
$pythonExecutable = (Get-Command python).Source
$pythonIncludeDir = python -c "import sysconfig; print(sysconfig.get_path('include'))" 2>$null
$pythonVersion = python -c "import sys; print(f'{sys.version_info.major}{sys.version_info.minor}')"

Write-ColorOutput "=== Build Configuration ===" -ForegroundColor Magenta
Write-ColorOutput "Python Executable: $pythonExecutable" -ForegroundColor Cyan
Write-ColorOutput "Python Include Dir: $pythonIncludeDir" -ForegroundColor Cyan
Write-ColorOutput "Python Version: $pythonVersion" -ForegroundColor Cyan
Write-ColorOutput "pybind11 Path: $pybind11Path" -ForegroundColor Cyan

# Set build directory
$BUILD_DIR = "build"

# Clean build directory if it exists
if (Test-Path $BUILD_DIR) {
    Write-ColorOutput "Cleaning build directory..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force $BUILD_DIR
}

# Create build directory
New-Item -ItemType Directory -Path $BUILD_DIR | Out-Null

# Get Python library directory and other paths
$pythonLibDir = python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR') or '')" 2>$null
if (-not $pythonLibDir) {
    $pythonLibDir = [System.IO.Path]::GetDirectoryName($pythonExecutable)
}

# Get Python version and paths
$pythonVersion = python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
$pythonIncludeDir = python -c "import sysconfig; print(sysconfig.get_path('include'))"
try {
    $pythonSitePackages = python -c "import site; print(site.getsitepackages()[0])" 2>$null
} catch {
    # Fallback for virtualenv
    $pythonSitePackages = python -c "import sysconfig; print(sysconfig.get_path('purelib'))"
}

# Ensure we have pybind11 installed
$pybind11Path = python -c "import pybind11, os; print(os.path.dirname(pybind11.__file__))" 2>$null
if (-not $pybind11Path) {
    Write-ColorOutput "pybind11 not found. Installing..." -ForegroundColor Yellow
    python -m pip install pybind11 --user
    $pybind11Path = python -c "import pybind11, os; print(os.path.dirname(pybind11.__file__))"
}

Write-ColorOutput "Python Configuration:" -ForegroundColor Cyan
Write-ColorOutput "  Executable: $pythonExecutable" -ForegroundColor Cyan
Write-ColorOutput "  Version: $pythonVersion" -ForegroundColor Cyan
Write-ColorOutput "  Include: $pythonIncludeDir" -ForegroundColor Cyan
Write-ColorOutput "  Site Packages: $pythonSitePackages" -ForegroundColor Cyan
Write-ColorOutput "  pybind11 Path: $pybind11Path" -ForegroundColor Cyan

# Configure with CMake
Write-ColorOutput "`nConfiguring CMake..." -ForegroundColor Magenta

# At this point, we should have a valid generator
if ($null -eq $generator) {
    Write-ColorOutput "FATAL: Failed to detect a valid build system. This should not happen." -ForegroundColor Red
    exit 1
}

# Set up environment based on the selected generator
if ($generator -match "Visual Studio") {
    # Set up Visual Studio environment
    $vsPath = Get-VisualStudioPath
    if ($vsPath) {
        # First try the new vcvarsall.bat
        $vcvarsall = "$vsPath\VC\Auxiliary\Build\vcvars64.bat"
        if (Test-Path $vcvarsall) {
            Write-ColorOutput "Setting up Visual Studio environment using vcvars64.bat..." -ForegroundColor Cyan
            cmd /c "`"$vcvarsall`" && set > `"$env:TEMP\vcvars.tmp`""
            Get-Content "$env:TEMP\vcvars.tmp" | ForEach-Object {
                if ($_ -match '^([^=]+)=(.*)') {
                    $varName = $matches[1]
                    $varValue = $matches[2]
                    [System.Environment]::SetEnvironmentVariable($varName, $varValue, 'Process')
                }
            }
            Remove-Item "$env:TEMP\vcvars.tmp" -ErrorAction SilentlyContinue
        } else {
            # Fall back to VsDevCmd if vcvarsall is not found
            $vsdevcmd = "$vsPath\Common7\Tools\VsDevCmd.bat"
            if (Test-Path $vsdevcmd) {
                Write-ColorOutput "Setting up Visual Studio environment using VsDevCmd.bat..." -ForegroundColor Cyan
                cmd /c "`"$vsdevcmd`" -arch=x64 -no_logo && set > `"$env:TEMP\vcvars.tmp`""
                Get-Content "$env:TEMP\vcvars.tmp" | ForEach-Object {
                    if ($_ -match '^([^=]+)=(.*)') {
                        $varName = $matches[1]
                        $varValue = $matches[2]
                        [System.Environment]::SetEnvironmentVariable($varName, $varValue, 'Process')
                    }
                }
                Remove-Item "$env:TEMP\vcvars.tmp" -ErrorAction SilentlyContinue
            }
        }
    }
    
    # Set the generator to the specific Visual Studio version
    $generator = "Visual Studio 17 2022"
    
} elseif ($generator -eq "MSBuild") {
    # Ensure MSBuild is in PATH
    $msbuildPath = (Get-Command "msbuild" -ErrorAction SilentlyContinue).Source
    if ($null -eq $msbuildPath) {
        Write-ColorOutput "MSBuild not found in PATH. Please ensure .NET SDK is installed." -ForegroundColor Red
        exit 1
    }
}

# Set environment variables for MSVC if using Visual Studio
if ($generator -match "Visual Studio") {
    $vsPath = Get-VisualStudioPath
    if ($vsPath) {
        $vcvarsall = "$vsPath\VC\Auxiliary\Build\vcvars64.bat"
        if (Test-Path $vcvarsall) {
            Write-ColorOutput "Setting up MSVC environment..." -ForegroundColor Cyan
            cmd /c "`"$vcvarsall`" && set > %TEMP%\vcvars.tmp"
            Get-Content "$env:TEMP\vcvars.tmp" | ForEach-Object {
                if ($_ -match '^([^=]+)=(.*)') {
                    $varName = $matches[1]
                    $varValue = $matches[2]
                    [System.Environment]::SetEnvironmentVariable($varName, $varValue, 'Process')
                }
            }
            Remove-Item "$env:TEMP\vcvars.tmp"
        }
    }
}

# Check available generators
$availableGenerators = cmake --help | Select-String -Pattern "^  \*?\s*(.*?)\s*=\s" | ForEach-Object { $_.Matches.Groups[1].Value.Trim() }

# Try to find the best available generator
$preferredGenerators = @(
    @{ Name = "Visual Studio 17 2022"; Arch = "x64" },
    @{ Name = "Visual Studio 16 2019"; Arch = "x64" },
    @{ Name = "Visual Studio 15 2017"; Arch = "x64" },
    @{ Name = "Ninja"; Arch = $null }
)

foreach ($gen in $preferredGenerators) {
    if ($availableGenerators -contains $gen.Name) {
        $generator = $gen.Name
        $generatorArch = $gen.Arch
        Write-ColorOutput "Found generator: $generator" -ForegroundColor Green
        
        if ($generator -eq "Ninja") {
            $buildArgs = @("--build", $buildDir, "--config", "Release")
        } else {
            $buildArgs = @(
                "--build", $buildDir,
                "--config", "Release",
                "--", "/m:$env:NUMBER_OF_PROCESSORS"
            )
        }
        break
    }
}

if (-not $generator) {
    Write-ColorOutput "No suitable CMake generator found. Please install Visual Studio or Ninja build system." -ForegroundColor Red
    exit 1
}

# Prepare CMake arguments
$cmakeArgs = @(
    "-S", ".",
    "-B", $BUILD_DIR,
    "-G", "$generator",
    "-DCMAKE_BUILD_TYPE=Release",
    "-DPYTHON_EXECUTABLE=$pythonExe",
    "-DPYTHON_INCLUDE_DIR=$pythonIncludeDir",
    "-DPYTHON_LIBRARY=$pythonLibDir",
    "-DPYTHON_SITE_PACKAGES=$pythonSitePackages",
    "-DPython3_ROOT_DIR=$pythonBasePath",
    "-DPython3_EXECUTABLE=$pythonExe",
    "-DPython3_INCLUDE_DIR=$pythonIncludeDir",
    "-DPython3_LIBRARY=$pythonLibDir",
    "-Dpybind11_DIR=$pybind11Path",
    "-DCMAKE_VERBOSE_MAKEFILE:BOOL=ON"
)

# Add architecture for Visual Studio generators
if ($generator -ne "Ninja" -and $generatorArch) {
    $cmakeArgs += @("-A", $generatorArch)
    $cmakeArgs += "-Thost=x64"
}

# Set Python environment variables
$env:Python3_ROOT_DIR = $pythonBasePath
$env:Python3_EXECUTABLE = $pythonExe
$env:Python3_INCLUDE_DIR = $pythonIncludeDir
$env:Python3_LIBRARY = $pythonLibDir

Write-ColorOutput "Using generator: $generator" -ForegroundColor Cyan
if ($generatorArch) {
    Write-ColorOutput "Architecture: $generatorArch" -ForegroundColor Cyan
}

# Function to run CMake and handle output
function Invoke-CMake {
    param (
        [Parameter(Mandatory=$true)]
        [array]$Arguments
    )
    
    $process = Start-Process -FilePath "cmake" -ArgumentList $Arguments -NoNewWindow -PassThru -RedirectStandardOutput "$BUILD_DIR/cmake_stdout.log" -RedirectStandardError "$BUILD_DIR/cmake_stderr.log"
    $process.WaitForExit()
    
    $stdout = Get-Content "$BUILD_DIR/cmake_stdout.log" -Raw -ErrorAction SilentlyContinue
    $stderr = Get-Content "$BUILD_DIR/cmake_stderr.log" -Raw -ErrorAction SilentlyContinue
    
    return @{
        ExitCode = $process.ExitCode
        StdOut = $stdout
        StdErr = $stderr
    }
}

# Run CMake configuration
Write-ColorOutput "Running: cmake $($cmakeArgs -join ' ')" -ForegroundColor Cyan
$cmakeResult = Invoke-CMake -Arguments $cmakeArgs

# Save combined output for debugging
$cmakeOutput = "=== STDOUT ===`n$($cmakeResult.StdOut)`n=== STDERR ===`n$($cmakeResult.StdErr)"
$cmakeOutput | Out-File -FilePath "$BUILD_DIR/cmake_configure.log" -Force

# Check for common CMake errors
if ($cmakeResult.ExitCode -ne 0 -or $cmakeOutput -match "CMake Error") {
    Write-ColorOutput "CMake configuration failed. Checking for common issues..." -ForegroundColor Yellow
    
    # Display error details
    if (-not [string]::IsNullOrEmpty($cmakeResult.StdErr)) {
        Write-ColorOutput "CMake Error Output:" -ForegroundColor Red
        $cmakeResult.StdErr.Trim() -split "`n" | ForEach-Object { Write-ColorOutput "  $_" -ForegroundColor Red }
    }
    
    # Check for common issues
    if ($cmakeOutput -match "Could NOT find pybind11" -or $cmakeOutput -match "pybind11-config.cmake") {
        Write-ColorOutput "pybind11 configuration issue detected. Reinstalling pybind11..." -ForegroundColor Yellow
        
        # Force reinstall pybind11
        python -m pip install --force-reinstall pybind11 --user
        
        # Try running CMake again
        $cmakeResult = Invoke-CMake -Arguments $cmakeArgs
        $cmakeOutput = "=== STDOUT ===`n$($cmakeResult.StdOut)`n=== STDERR ===`n$($cmakeResult.StdErr)"
        $cmakeOutput | Out-File -FilePath "$BUILD_DIR/cmake_configure.log" -Append
        
        if ($cmakeResult.ExitCode -ne 0) {
            Write-ColorOutput "CMake still failing after pybind11 reinstall." -ForegroundColor Red
            Write-ColorOutput "Please check the following paths:" -ForegroundColor Red
            Write-ColorOutput "  - Python executable: $pythonExecutable" -ForegroundColor Red
            Write-ColorOutput "  - Python version: $pythonVersion" -ForegroundColor Red
            Write-ColorOutput "  - Python include: $pythonIncludeDir" -ForegroundColor Red
            Write-ColorOutput "  - pybind11 path: $pybind11Path" -ForegroundColor Red
            Write-ColorOutput "  - CMake logs: $BUILD_DIR/cmake_*.log" -ForegroundColor Red
            exit 1
        }
    } else {
        Write-ColorOutput "CMake configuration failed. See $BUILD_DIR/cmake_*.log for details." -ForegroundColor Red
        exit 1
    }
}

# Build the project
Write-ColorOutput "`nBuilding project..." -ForegroundColor Magenta
$buildResult = Start-Process -FilePath "cmake" -ArgumentList "--build", "$BUILD_DIR", "--config", "Release" -NoNewWindow -PassThru -Wait

if ($buildResult.ExitCode -ne 0) {
    $buildLog = Get-Content -Path "$BUILD_DIR/CMakeFiles/CMakeError.log" -ErrorAction SilentlyContinue
    if (-not $buildLog) {
        $buildLog = Get-Content -Path "$BUILD_DIR/CMakeFiles/CMakeOutput.log" -ErrorAction SilentlyContinue
    }
    
    Write-ColorOutput "Build failed with exit code $($buildResult.ExitCode)" -ForegroundColor Red
    if ($buildLog) {
        Write-ColorOutput "Build log:" -ForegroundColor Red
        $buildLog | Select-Object -First 50 | ForEach-Object { Write-ColorOutput "  $_" -ForegroundColor Red }
        if ($buildLog.Count -gt 50) {
            Write-ColorOutput "  ... and $($buildLog.Count - 50) more lines" -ForegroundColor Red
        }
    }
    exit $buildResult.ExitCode
}

# Install the project
Write-ColorOutput "`nInstalling project..." -ForegroundColor Magenta
$installArgs = @(
    "--install", $BUILD_DIR,
    "--config", "Release",
    "--prefix", "$PWD/install"
)

$installResult = Start-Process -FilePath "cmake" -ArgumentList $installArgs -NoNewWindow -PassThru -Wait

if ($installResult.ExitCode -ne 0) {
    Write-ColorOutput "Installation failed with exit code $($installResult.ExitCode)" -ForegroundColor Red
    exit $installResult.ExitCode
}

# Copy Python modules to site-packages
$pythonSitePackages = python -c "import site; print(site.getsitepackages()[0])"
$installDir = "$PWD/install"

if (Test-Path "$installDir/nlp") {
    Write-ColorOutput "`nCopying Python modules to site-packages..." -ForegroundColor Cyan
    
    $targetDir = "$pythonSitePackages/jarvis"
    if (-not (Test-Path $targetDir)) {
        New-Item -ItemType Directory -Path $targetDir -Force | Out-Null
    }
    
    Get-ChildItem -Path "$installDir/nlp" -Filter "*.pyd" | ForEach-Object {
        $targetFile = Join-Path $targetDir $_.Name
        Copy-Item -Path $_.FullName -Destination $targetFile -Force
        Write-ColorOutput "  Copied $($_.Name) to $targetFile" -ForegroundColor Green
    }
    
    # Create __init__.py if it doesn't exist
    $initFile = Join-Path $targetDir "__init__.py"
    if (-not (Test-Path $initFile)) {
        Set-Content -Path $initFile -Value "# JARVIS Python Package\n"
        Write-ColorOutput "  Created $initFile" -ForegroundColor Green
    }
}

Write-ColorOutput "`nBuild and installation completed successfully!" -ForegroundColor Green
Write-ColorOutput "Python modules are available in: $pythonSitePackages/jarvis" -ForegroundColor Green

# Build completed successfully
