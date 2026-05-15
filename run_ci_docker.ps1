#!/usr/bin/env pwsh
# AetherArch Local CI Test Runner using Docker
# Runs all GitHub Actions CI jobs in Docker containers (Cross-platform)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

$Branch = git rev-parse --abbrev-ref HEAD 2>$null
$Commit = git rev-parse --short HEAD 2>$null
$Timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$ImageName = "aether-ci-test"
$ContainerName = "aether-ci-$Branch-$Commit"

# Color codes for output
function Write-ColorOutput($ForegroundColor) {
    $fc = $host.UI.RawUI.ForegroundColor
    $host.UI.RawUI.ForegroundColor = $ForegroundColor
    if ($args) {
        Write-Output $args
    }
    $host.UI.RawUI.ForegroundColor = $fc
}

function Write-Header {
    Write-ColorOutput Cyan "╔════════════════════════════════════════════════════════════╗"
    Write-ColorOutput Cyan "║           AetherArch Local CI Test Runner                   ║"
    Write-ColorOutput Cyan "╚════════════════════════════════════════════════════════════╝"
    Write-Output ""
}

function Write-Section($Message) {
    Write-ColorOutput Cyan "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    Write-ColorOutput Cyan "→ $Message"
    Write-ColorOutput Cyan "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}

function Write-Success($Message) {
    Write-ColorOutput Green "✓ $Message"
}

function Write-Error($Message) {
    Write-ColorOutput Red "✗ $Message"
}

function Write-Info($Message) {
    Write-ColorOutput Yellow "ℹ $Message"
}

Write-Header

Write-Output "System Information:"
Write-Output "  Branch:          $Branch"
Write-Output "  Commit:          $Commit"
Write-Output "  Timestamp:       $Timestamp"
Write-Output "  Project Root:    $ProjectRoot"
Write-Output "  Image Name:      $ImageName"
Write-Output "  Container Name:  $ContainerName"
Write-Output ""

# Check Docker availability
Write-Section "Checking Docker"
try {
    $DockerVersion = docker --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Success "Docker found: $DockerVersion"
    } else {
        throw "Docker command failed"
    }
} catch {
    Write-Error "Docker is not installed or not running"
    Write-Output ""
    Write-Output "Installation:"
    Write-Output "  Windows: Install Docker Desktop from https://www.docker.com/products/docker-desktop"
    Write-Output "  Linux:   sudo apt-get install docker.io"
    Write-Output "  macOS:   brew install --cask docker"
    Write-Output ""
    exit 1
}
Write-Output ""

# Check Git availability
Write-Section "Checking Git"
try {
    $GitVersion = git --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Success "Git found: $GitVersion"
    } else {
        throw "Git command failed"
    }
} catch {
    Write-Error "Git is not installed"
    exit 1
}
Write-Output ""

# Build Docker image
Write-Section "Building Docker Image"
Write-Info "Building image: $ImageName"
Write-Info "Command: docker build -f Dockerfile.ci -t $Image_NAME ."
Write-Output ""

$BuildArgs = @("build", "-f", "Dockerfile.ci", "-t", $ImageName, ".")
$BuildResult = & docker @BuildArgs 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Error "Docker build failed"
    Write-Output $BuildResult
    exit 1
}
Write-Success "Docker image built: $ImageName"
Write-Output ""

# Parse command line arguments
$RunAll = $true
$SpecificTest = $args[0]

if ($SpecificTest) {
    $RunAll = $false
    Write-Info "Running specific test: $SpecificTest"
}

# Test functions
function Test-Build {
    Write-Section "1. Build (Release)"
    Write-Info "Command: cargo build --workspace --release"
    $RunArgs = @("run", "--rm", "--name", "$ContainerName-build", "--cpus", "4", "--memory", "8g",
                 "-v", "${ProjectRoot}:/workspace", "-w", "/workspace", $ImageName,
                 "cargo", "build", "--workspace", "--release")
    & docker @RunArgs
    if ($LASTEXITCODE -eq 0) {
        Write-Success "Build test passed"
        return $true
    } else {
        Write-Error "Build test failed"
        return $false
    }
}

function Test-Tests {
    Write-Section "2. Tests (Unit + Integration)"
    Write-Info "Command: cargo test --workspace --release"
    $RunArgs = @("run", "--rm", "--name", "$ContainerName-test", "--cpus", "4", "--memory", "8g",
                 "-v", "${ProjectRoot}:/workspace", "-w", "/workspace", $ImageName,
                 "cargo", "test", "--workspace", "--release")
    & docker @RunArgs
    if ($LASTEXITCODE -eq 0) {
        Write-Success "Test suite passed"
        return $true
    } else {
        Write-Error "Test suite failed"
        return $false
    }
}

function Test-Clippy {
    Write-Section "3. Clippy Linting"
    Write-Info "Command: cargo clippy --workspace -- -D warnings"
    $RunArgs = @("run", "--rm", "--name", "$ContainerName-clippy", "--cpus", "4", "--memory", "8g",
                 "-v", "${ProjectRoot}:/workspace", "-w", "/workspace", $ImageName,
                 "bash", "-c", "cargo clippy --workspace -- -D warnings")
    & docker @RunArgs
    if ($LASTEXITCODE -eq 0) {
        Write-Success "Clippy linting passed"
        return $true
    } else {
        Write-Error "Clippy linting failed"
        return $false
    }
}

function Test-Fmt {
    Write-Section "4. Format Check (rustfmt)"
    Write-Info "Command: cargo fmt --all -- --check"
    $RunArgs = @("run", "--rm", "--name", "$ContainerName-fmt", "--cpus", "4", "--memory", "8g",
                 "-v", "${ProjectRoot}:/workspace", "-w", "/workspace", $ImageName,
                 "bash", "-c", "cargo fmt --all -- --check")
    & docker @RunArgs
    if ($LASTEXITCODE -eq 0) {
        Write-Success "Format check passed"
        return $true
    } else {
        Write-Error "Format check failed"
        return $false
    }
}

function Test-Doc {
    Write-Section "5. Documentation Build"
    Write-Info "Command: cargo doc --workspace --no-deps"
    $RunArgs = @("run", "--rm", "--name", "$ContainerName-doc", "--cpus", "4", "--memory", "8g",
                 "-e", "RUSTDOCFLAGS=-Dwarnings",
                 "-v", "${ProjectRoot}:/workspace", "-w", "/workspace", $ImageName,
                 "bash", "-c", "cargo doc --workspace --no-deps")
    & docker @RunArgs
    if ($LASTEXITCODE -eq 0) {
        Write-Success "Documentation build passed"
        return $true
    } else {
        Write-Error "Documentation build failed"
        return $false
    }
}

function Test-MSRV {
    Write-Section "6. MSRV Check (Rust 1.85.0)"
    Write-Info "Command: cargo +1.85.0 check --workspace"
    $RunArgs = @("run", "--rm", "--name", "$ContainerName-msrv", "--cpus", "4", "--memory", "8g",
                 "-v", "${ProjectRoot}:/workspace", "-w", "/workspace", $ImageName,
                 "bash", "-c", "rustup install 1.85.0 && cargo +1.85.0 check --workspace")
    & docker @RunArgs
    if ($LASTEXITCODE -eq 0) {
        Write-Success "MSRV check passed"
        return $true
    } else {
        Write-Error "MSRV check failed"
        return $false
    }
}

# Run tests
$Results = @{}

if ($RunAll) {
    Write-Section "Running All CI Tests"
    Write-Output ""
    
    # First run the full CI pipeline from Dockerfile CMD
    Write-Info "Running full CI pipeline from Dockerfile..."
    $RunArgs = @("run", "--rm", "--name", $ContainerName, "--cpus", "4", "--memory", "8g",
                 "-v", "${ProjectRoot}:/workspace", $ImageName)
    & docker @RunArgs
    
    if ($LASTEXITCODE -eq 0) {
        Write-Success "Full CI pipeline passed"
        $Results["Build"] = $true
        $Results["Tests"] = $true
        $Results["Clippy"] = $true
        $Results["Fmt"] = $true
        $Results["Doc"] = $true
        $Results["MSRV"] = $true
    } else {
        Write-Error "Full CI pipeline failed, running individual tests..."
        Write-Output ""
        
        # Run individual tests for detailed reporting
        $Results["Build"] = Test-Build
        Write-Output ""
        $Results["Tests"] = Test-Tests
        Write-Output ""
        $Results["Clippy"] = Test-Clippy
        Write-Output ""
        $Results["Fmt"] = Test-Fmt
        Write-Output ""
        $Results["Doc"] = Test-Doc
        Write-Output ""
        $Results["MSRV"] = Test-MSRV
        Write-Output ""
    }
} else {
    # Run specific test
    switch ($SpecificTest) {
        "build" { $Results["Build"] = Test-Build }
        "test" { $Results["Tests"] = Test-Tests }
        "clippy" { $Results["Clippy"] = Test-Clippy }
        "fmt" { $Results["Fmt"] = Test-Fmt }
        "doc" { $Results["Doc"] = Test-Doc }
        "msrv" { $Results["MSRV"] = Test-MSRV }
        default {
            Write-Error "Unknown test: $SpecificTest"
            Write-Output "Available tests: build, test, clippy, fmt, doc, msrv"
            exit 1
        }
    }
}

# Summary
Write-Section "CI Test Summary"
$AllPassed = $true

foreach ($Key in $Results.Keys) {
    if ($Results[$Key]) {
        Write-Success "$Key : PASSED"
    } else {
        Write-Error "$Key : FAILED"
        $AllPassed = $false
    }
}

Write-Output ""

if ($AllPassed) {
    Write-ColorOutput Green "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    Write-ColorOutput Green "✓ All CI checks passed successfully!"
    Write-ColorOutput Green "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    Write-Output ""
    Write-Output "Next Steps:"
    Write-Output "  1. Commit and push to branch: git push origin $Branch"
    Write-Output "  2. Create a pull request to main"
    Write-Output "  3. GitHub Actions will run additional platform tests:"
    Write-Output "     • Ubuntu (Linux)"
    Write-Output "     • Windows"
    Write-Output "     • macOS"
    Write-Output ""
    exit 0
} else {
    Write-ColorOutput Red "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    Write-ColorOutput Red "✗ Some CI checks failed!"
    Write-ColorOutput Red "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    Write-Output ""
    Write-Output "Please fix the failing tests above before pushing."
    Write-Output ""
    exit 1
}
