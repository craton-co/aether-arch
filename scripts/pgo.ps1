#requires -Version 5.1
<#
.SYNOPSIS
    Profile-Guided Optimization (PGO) build for `aet` (aether-cli).

.DESCRIPTION
    One-shot wrapper around `cargo-pgo` that:
      1. Builds an instrumented binary into target/release-pgo/.
      2. Runs a training workload (compress + extract a handful of representative
         fixtures, repeated for sample volume) to produce raw profile data.
      3. Merges the raw profiles with `llvm-profdata`.
      4. Rebuilds with `-Cprofile-use` to produce the final optimized `aet`.

    The standard `cargo build --release` is NOT affected — PGO artifacts live
    in a dedicated `release-pgo` profile / target subdir.

.NOTES
    Requires:
      - cargo-pgo            (cargo install cargo-pgo)
      - llvm-tools-preview   (rustup component add llvm-tools-preview)

.EXAMPLE
    pwsh ./scripts/pgo.ps1
#>

[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$ProgressPreference    = 'SilentlyContinue'

# Resolve repo root from script location (scripts/ is at the workspace root).
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot  = Resolve-Path (Join-Path $ScriptDir '..')

function Write-Step($msg) {
    Write-Host ""
    Write-Host "==> $msg" -ForegroundColor Cyan
}

function Write-Err($msg) {
    Write-Host "ERROR: $msg" -ForegroundColor Red
}

# ─── 1. Tool availability checks ─────────────────────────────────────────────
Write-Step "Checking prerequisites"

$cargoPgoVersion = & cargo pgo --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Err "cargo-pgo is not installed."
    Write-Host ""
    Write-Host "Install it with:"   -ForegroundColor Yellow
    Write-Host "    cargo install cargo-pgo"
    Write-Host ""
    Write-Host "Then ensure llvm-tools-preview is present:" -ForegroundColor Yellow
    Write-Host "    rustup component add llvm-tools-preview"
    exit 1
}
Write-Host "  cargo-pgo: $($cargoPgoVersion -join ' ')"

# llvm-profdata ships with the llvm-tools-preview rustup component.
# It is typically placed under the active toolchain's lib/rustlib/<host>/bin.
$llvmProfdata = $null
$rustcSysroot = (& rustc --print sysroot).Trim()
if ($rustcSysroot) {
    $hostTriple = (& rustc -vV | Select-String '^host:').ToString().Split(':')[1].Trim()
    $candidate  = Join-Path $rustcSysroot ("lib\rustlib\$hostTriple\bin\llvm-profdata.exe")
    if (Test-Path $candidate) { $llvmProfdata = $candidate }
}
if (-not $llvmProfdata) {
    # Fall back to PATH.
    $cmd = Get-Command llvm-profdata -ErrorAction SilentlyContinue
    if ($cmd) { $llvmProfdata = $cmd.Source }
}
if (-not $llvmProfdata) {
    Write-Err "llvm-profdata not found."
    Write-Host ""
    Write-Host "Install the llvm-tools-preview rustup component:" -ForegroundColor Yellow
    Write-Host "    rustup component add llvm-tools-preview"
    exit 1
}
Write-Host "  llvm-profdata: $llvmProfdata"

# ─── 2. Locate training fixtures ─────────────────────────────────────────────
Write-Step "Locating training fixtures"

$FixtureDir = Join-Path $RepoRoot 'tests\fixtures\large'
if (-not (Test-Path $FixtureDir)) {
    Write-Err "Training fixture directory not found: $FixtureDir"
    exit 1
}

$Fixtures = @()
foreach ($name in 'english.txt', 'source.rs', 'mixed.json') {
    $p = Join-Path $FixtureDir $name
    if (Test-Path $p) { $Fixtures += (Resolve-Path $p).Path }
    else { Write-Host "  (skip, not present) $name" -ForegroundColor DarkGray }
}
if ($Fixtures.Count -eq 0) {
    Write-Err "No training fixtures found under $FixtureDir"
    exit 1
}
foreach ($f in $Fixtures) {
    $len = (Get-Item $f).Length
    Write-Host ("  fixture: {0,-12} {1,10} bytes" -f (Split-Path -Leaf $f), $len)
}

# ─── 3. Instrumented build ───────────────────────────────────────────────────
Write-Step "Building instrumented binary (cargo pgo instrument build)"

Push-Location $RepoRoot
try {
    & cargo pgo instrument build -- -p aether-cli --profile release-pgo
    if ($LASTEXITCODE -ne 0) { throw "cargo pgo instrument build failed (exit $LASTEXITCODE)" }
}
finally {
    Pop-Location
}

# cargo-pgo writes the instrumented binary into target/<triple>/release-pgo/.
$HostTriple = (& rustc -vV | Select-String '^host:').ToString().Split(':')[1].Trim()
$InstrumentedBin = Join-Path $RepoRoot "target\$HostTriple\release-pgo\aet.exe"
if (-not (Test-Path $InstrumentedBin)) {
    # Some cargo-pgo versions don't include the triple subdir when host matches default.
    $alt = Join-Path $RepoRoot "target\release-pgo\aet.exe"
    if (Test-Path $alt) { $InstrumentedBin = $alt }
    else {
        Write-Err "Instrumented binary not found at expected paths:"
        Write-Host "    $InstrumentedBin"
        Write-Host "    $alt"
        exit 1
    }
}
Write-Host "  instrumented: $InstrumentedBin"

# cargo-pgo collects raw .profraw under target/pgo-profiles/ by default.
$ProfileDir = Join-Path $RepoRoot 'target\pgo-profiles'
if (-not (Test-Path $ProfileDir)) { New-Item -ItemType Directory -Force -Path $ProfileDir | Out-Null }
$env:LLVM_PROFILE_FILE = (Join-Path $ProfileDir 'aet-%p-%m.profraw')

# ─── 4. Training workload ────────────────────────────────────────────────────
Write-Step "Running training workload"

$WorkDir = Join-Path $env:TEMP ("aether-pgo-train-" + [Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Force -Path $WorkDir | Out-Null

# Repetitions per fixture per direction. Each iteration covers full compress
# pipeline + full extract pipeline, so total instrumented invocations =
# iterations * fixtures * 2.
$Iterations = 3

try {
    foreach ($i in 1..$Iterations) {
        foreach ($fixture in $Fixtures) {
            $name    = Split-Path -Leaf $fixture
            $archive = Join-Path $WorkDir ("{0}.{1}.aet" -f $name, $i)
            $outDir  = Join-Path $WorkDir ("out-{0}-{1}" -f $name, $i)
            New-Item -ItemType Directory -Force -Path $outDir | Out-Null

            Write-Host ("  [iter {0}/{1}] compress {2}" -f $i, $Iterations, $name)
            & $InstrumentedBin compress $fixture -o $archive --force | Out-Null
            if ($LASTEXITCODE -ne 0) { throw "instrumented compress failed for $fixture" }

            Write-Host ("  [iter {0}/{1}] extract  {2}" -f $i, $Iterations, $name)
            & $InstrumentedBin extract $archive -o $outDir | Out-Null
            if ($LASTEXITCODE -ne 0) { throw "instrumented extract failed for $archive" }
        }
    }
}
finally {
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue $WorkDir
}

$RawProfiles = Get-ChildItem -Path $ProfileDir -Filter '*.profraw' -ErrorAction SilentlyContinue
Write-Host ("  raw profiles produced: {0}" -f $RawProfiles.Count)
if ($RawProfiles.Count -eq 0) {
    Write-Err "No .profraw files produced. Did instrumentation actually run?"
    exit 1
}

# ─── 5. Optimized build ──────────────────────────────────────────────────────
Write-Step "Building optimized binary (cargo pgo optimize build)"

Push-Location $RepoRoot
try {
    & cargo pgo optimize build -- -p aether-cli --profile release-pgo
    if ($LASTEXITCODE -ne 0) { throw "cargo pgo optimize build failed (exit $LASTEXITCODE)" }
}
finally {
    Pop-Location
}

# ─── 6. Report ───────────────────────────────────────────────────────────────
Write-Step "PGO build complete"

$OptimizedBin = Join-Path $RepoRoot "target\$HostTriple\release-pgo\aet.exe"
if (-not (Test-Path $OptimizedBin)) {
    $alt = Join-Path $RepoRoot "target\release-pgo\aet.exe"
    if (Test-Path $alt) { $OptimizedBin = $alt }
}

Write-Host "  optimized binary: $OptimizedBin"
if (Test-Path $OptimizedBin) {
    $sz = (Get-Item $OptimizedBin).Length
    Write-Host ("  size:             {0:N0} bytes" -f $sz)
}

# Show merged-profile sample count via llvm-profdata.
# cargo-pgo merges into target/pgo-profiles/merged.profdata (name varies by version);
# fall back to the first .profdata found.
$Merged = Get-ChildItem -Path $ProfileDir -Filter '*.profdata' -ErrorAction SilentlyContinue | Select-Object -First 1
if ($Merged) {
    Write-Host "  merged profile:   $($Merged.FullName)"
    $show = & $llvmProfdata show $Merged.FullName 2>&1
    $summary = $show | Where-Object {
        $_ -match 'Total number of functions' -or
        $_ -match 'Maximum function count'    -or
        $_ -match 'Total number of blocks'    -or
        $_ -match 'sample count'              -or
        $_ -match 'Instrumentation level'
    }
    if ($summary) {
        Write-Host "  profile summary:"
        foreach ($line in $summary) { Write-Host "    $line" }
    }
} else {
    Write-Host "  (no merged .profdata found under $ProfileDir)" -ForegroundColor DarkGray
}

Write-Host ""
Write-Host "Done. Standard `cargo build --release` is unaffected." -ForegroundColor Green
