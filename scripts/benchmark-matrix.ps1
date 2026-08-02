param(
    [Parameter(Mandatory = $true)]
    [string]$Binary,

    [Parameter(Mandatory = $true)]
    [string]$DatasetRoot,

    [string]$OutputCsv = "benchmark-matrix.csv",

    [ValidateRange(1, 100)]
    [int]$Iterations = 10,

    [ValidateSet("archival", "balanced", "fast")]
    [string[]]$Profiles = @("archival", "balanced", "fast")
)

$ErrorActionPreference = "Stop"
$classes = @("text", "logs", "binaries", "images", "tiny")
$binaryPath = (Resolve-Path -LiteralPath $Binary).Path
$datasetPath = (Resolve-Path -LiteralPath $DatasetRoot).Path
$runId = "aether-bench-{0}-{1}" -f $PID, [guid]::NewGuid().ToString("N")
$scratch = Join-Path ([IO.Path]::GetTempPath()) $runId
$results = [Collections.Generic.List[object]]::new()

New-Item -ItemType Directory -Path $scratch | Out-Null
try {
    foreach ($class in $classes) {
        $inputPath = Join-Path $datasetPath $class
        if (-not (Test-Path -LiteralPath $inputPath -PathType Container)) {
            Write-Warning "Skipping missing workload class: $inputPath"
            continue
        }

        $inputBytes = (Get-ChildItem -LiteralPath $inputPath -File -Recurse |
            Measure-Object -Property Length -Sum).Sum
        if ($null -eq $inputBytes) {
            $inputBytes = 0
        }

        foreach ($profile in $Profiles) {
            for ($iteration = 1; $iteration -le $Iterations; $iteration++) {
                $caseId = "{0}-{1}-{2}" -f $class, $profile, $iteration
                $archive = Join-Path $scratch "$caseId.aet"
                $extractDir = Join-Path $scratch "$caseId-out"

                $compressWatch = [Diagnostics.Stopwatch]::StartNew()
                $savedErrorPreference = $ErrorActionPreference
                $ErrorActionPreference = "Continue"
                & $binaryPath compress $inputPath --output $archive --predictor ssm `
                    --profile $profile --force 2>$null
                $compressExitCode = $LASTEXITCODE
                $ErrorActionPreference = $savedErrorPreference
                if ($compressExitCode -ne 0) {
                    throw "Compression failed for $caseId"
                }
                $compressWatch.Stop()

                $extractWatch = [Diagnostics.Stopwatch]::StartNew()
                $ErrorActionPreference = "Continue"
                & $binaryPath extract $archive --output $extractDir 2>$null
                $extractExitCode = $LASTEXITCODE
                $ErrorActionPreference = $savedErrorPreference
                if ($extractExitCode -ne 0) {
                    throw "Extraction failed for $caseId"
                }
                $extractWatch.Stop()

                $archiveBytes = (Get-Item -LiteralPath $archive).Length
                $results.Add([pscustomobject]@{
                    Class = $class
                    Profile = $profile
                    Iteration = $iteration
                    InputBytes = [int64]$inputBytes
                    ArchiveBytes = [int64]$archiveBytes
                    Ratio = if ($inputBytes -gt 0) { $archiveBytes / $inputBytes } else { 0 }
                    CompressSeconds = $compressWatch.Elapsed.TotalSeconds
                    ExtractSeconds = $extractWatch.Elapsed.TotalSeconds
                    CompressMiBs = if ($compressWatch.Elapsed.TotalSeconds -gt 0) {
                        $inputBytes / 1MB / $compressWatch.Elapsed.TotalSeconds
                    } else { 0 }
                    ExtractMiBs = if ($extractWatch.Elapsed.TotalSeconds -gt 0) {
                        $inputBytes / 1MB / $extractWatch.Elapsed.TotalSeconds
                    } else { 0 }
                })

                Remove-Item -LiteralPath $archive -Force
                Remove-Item -LiteralPath $extractDir -Recurse -Force
            }
        }
    }

    $results | Export-Csv -LiteralPath $OutputCsv -NoTypeInformation
    Write-Host "Wrote $($results.Count) measurements to $OutputCsv"
}
finally {
    $scratchFull = [IO.Path]::GetFullPath($scratch)
    $tempFull = [IO.Path]::GetFullPath([IO.Path]::GetTempPath())
    if ($scratchFull.StartsWith($tempFull, [StringComparison]::OrdinalIgnoreCase) -and
        (Split-Path -Leaf $scratchFull).StartsWith("aether-bench-")) {
        Remove-Item -LiteralPath $scratchFull -Recurse -Force -ErrorAction SilentlyContinue
    }
}
