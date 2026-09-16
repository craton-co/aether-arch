<#
.SYNOPSIS
    Decompression benchmark matrix: workload class x profile x thread count.

.DESCRIPTION
    The compression matrix (benchmark-matrix.ps1) measures one extraction per
    case, which is enough to spot a ratio regression but not enough to reason
    about read speed. This script is the decompression counterpart:

      * Compression runs once per case as *setup* and is not timed.
      * Extraction is repeated and reported as the minimum and median over
        the repetitions. The minimum is the number to compare across builds —
        it is the run least polluted by whatever else the host was doing.
      * Thread counts are swept, so scaling is visible rather than assumed.
      * `verify` (decode without writing files) and single-file extraction
        (random access via the block index) are measured alongside full
        extraction, because they are separate user-visible operations with
        different costs.

    Throughput is reported over the *decompressed* bytes. An archive-size
    rate makes a decompressor look faster the better the compressor did.

.PARAMETER Binary
    Path to the `aet` executable.

.PARAMETER DatasetRoot
    Directory containing one subdirectory per workload class
    (text, logs, binaries, images, tiny).

.PARAMETER Threads
    Thread counts to sweep. 0 means "all cores"; 1 is sequential.
    Values other than 1 need a binary built with the `threading` feature
    (on by default for `aether-cli`).

.EXAMPLE
    ./scripts/decompression-matrix.ps1 -Binary ./target/release/aet.exe `
        -DatasetRoot ./datasets -OutputCsv decompression-matrix.csv
#>
param(
    [Parameter(Mandatory = $true)]
    [string]$Binary,

    [Parameter(Mandatory = $true)]
    [string]$DatasetRoot,

    [string]$OutputCsv = "decompression-matrix.csv",

    [ValidateRange(1, 100)]
    [int]$Repetitions = 5,

    [ValidateSet("archival", "balanced", "fast")]
    [string[]]$Profiles = @("archival", "balanced", "fast"),

    [ValidateRange(0, 1024)]
    [int[]]$Threads = @(1, 0)
)

$ErrorActionPreference = "Stop"
$classes = @("text", "logs", "binaries", "images", "tiny")
$binaryPath = (Resolve-Path -LiteralPath $Binary).Path
$datasetPath = (Resolve-Path -LiteralPath $DatasetRoot).Path
$runId = "aether-dbench-{0}-{1}" -f $PID, [guid]::NewGuid().ToString("N")
$scratch = Join-Path ([IO.Path]::GetTempPath()) $runId
$results = [Collections.Generic.List[object]]::new()

function Invoke-Aet {
    param([string[]]$Arguments, [string]$What)

    $saved = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $binaryPath @Arguments 2>$null | Out-Null
    $code = $LASTEXITCODE
    $ErrorActionPreference = $saved
    if ($code -ne 0) {
        throw "$What failed (exit $code): $binaryPath $($Arguments -join ' ')"
    }
}

function Measure-Repeated {
    param([scriptblock]$Action, [int]$Count)

    $samples = [Collections.Generic.List[double]]::new()
    for ($i = 0; $i -lt $Count; $i++) {
        $watch = [Diagnostics.Stopwatch]::StartNew()
        & $Action
        $watch.Stop()
        $samples.Add($watch.Elapsed.TotalSeconds)
    }
    $sorted = $samples | Sort-Object
    [pscustomobject]@{
        Min = $sorted[0]
        Median = $sorted[[int][math]::Floor($sorted.Count / 2)]
    }
}

New-Item -ItemType Directory -Path $scratch | Out-Null
try {
    foreach ($class in $classes) {
        $inputPath = Join-Path $datasetPath $class
        if (-not (Test-Path -LiteralPath $inputPath -PathType Container)) {
            Write-Warning "Skipping missing workload class: $inputPath"
            continue
        }

        $files = Get-ChildItem -LiteralPath $inputPath -File -Recurse
        $inputBytes = ($files | Measure-Object -Property Length -Sum).Sum
        if ($null -eq $inputBytes) { $inputBytes = 0 }
        if ($inputBytes -eq 0) {
            Write-Warning "Skipping empty workload class: $inputPath"
            continue
        }

        # A representative single file for the random-access measurement:
        # the largest one, so the number is dominated by decoding rather than
        # by opening the archive.
        $largest = $files | Sort-Object Length -Descending | Select-Object -First 1
        $largestRelative = $largest.FullName.Substring($inputPath.Length).TrimStart('\', '/')
        $largestRelative = $largestRelative -replace '\\', '/'

        foreach ($profile in $Profiles) {
            $archive = Join-Path $scratch "$class-$profile.aet"

            # Setup, deliberately untimed.
            Invoke-Aet -Arguments @(
                "compress", $inputPath, "--output", $archive,
                "--predictor", "ssm", "--profile", $profile, "--force"
            ) -What "compress $class/$profile"

            $archiveBytes = (Get-Item -LiteralPath $archive).Length

            foreach ($threadCount in $Threads) {
                $extractDir = Join-Path $scratch "$class-$profile-$threadCount-out"

                $extract = Measure-Repeated -Count $Repetitions -Action {
                    if (Test-Path -LiteralPath $extractDir) {
                        Remove-Item -LiteralPath $extractDir -Recurse -Force
                    }
                    Invoke-Aet -Arguments @(
                        "extract", $archive, "--output", $extractDir, "--threads", $threadCount
                    ) -What "extract $class/$profile/t$threadCount"
                }
                if (Test-Path -LiteralPath $extractDir) {
                    Remove-Item -LiteralPath $extractDir -Recurse -Force
                }

                $verify = Measure-Repeated -Count $Repetitions -Action {
                    Invoke-Aet -Arguments @("verify", $archive) -What "verify $class/$profile"
                }

                $singleDir = Join-Path $scratch "$class-$profile-$threadCount-single"
                $single = Measure-Repeated -Count $Repetitions -Action {
                    if (Test-Path -LiteralPath $singleDir) {
                        Remove-Item -LiteralPath $singleDir -Recurse -Force
                    }
                    Invoke-Aet -Arguments @(
                        "extract", $archive, "--output", $singleDir,
                        "--file", $largestRelative, "--threads", $threadCount
                    ) -What "extract-file $class/$profile"
                }
                if (Test-Path -LiteralPath $singleDir) {
                    Remove-Item -LiteralPath $singleDir -Recurse -Force
                }

                $results.Add([pscustomobject]@{
                    Class = $class
                    Profile = $profile
                    Threads = $threadCount
                    Repetitions = $Repetitions
                    InputBytes = [int64]$inputBytes
                    ArchiveBytes = [int64]$archiveBytes
                    Ratio = $archiveBytes / $inputBytes
                    ExtractMinSeconds = $extract.Min
                    ExtractMedianSeconds = $extract.Median
                    ExtractMiBs = if ($extract.Min -gt 0) { $inputBytes / 1MB / $extract.Min } else { 0 }
                    VerifyMinSeconds = $verify.Min
                    VerifyMiBs = if ($verify.Min -gt 0) { $inputBytes / 1MB / $verify.Min } else { 0 }
                    SingleFile = $largestRelative
                    SingleFileBytes = [int64]$largest.Length
                    SingleFileMinSeconds = $single.Min
                    SingleFileMiBs = if ($single.Min -gt 0) { $largest.Length / 1MB / $single.Min } else { 0 }
                })
            }

            Remove-Item -LiteralPath $archive -Force
        }
    }

    $results | Export-Csv -LiteralPath $OutputCsv -NoTypeInformation
    Write-Host "Wrote $($results.Count) measurements to $OutputCsv"

    $results |
        Sort-Object Class, Profile, Threads |
        Format-Table Class, Profile, Threads,
            @{ Label = "Ratio"; Expression = { "{0:P2}" -f $_.Ratio } },
            @{ Label = "Extract MiB/s"; Expression = { "{0:N2}" -f $_.ExtractMiBs } },
            @{ Label = "Verify MiB/s"; Expression = { "{0:N2}" -f $_.VerifyMiBs } },
            @{ Label = "1-file MiB/s"; Expression = { "{0:N2}" -f $_.SingleFileMiBs } } |
        Out-String |
        Write-Host
}
finally {
    $scratchFull = [IO.Path]::GetFullPath($scratch)
    $tempFull = [IO.Path]::GetFullPath([IO.Path]::GetTempPath())
    if ($scratchFull.StartsWith($tempFull, [StringComparison]::OrdinalIgnoreCase) -and
        (Split-Path -Leaf $scratchFull).StartsWith("aether-dbench-")) {
        Remove-Item -LiteralPath $scratchFull -Recurse -Force -ErrorAction SilentlyContinue
    }
}
