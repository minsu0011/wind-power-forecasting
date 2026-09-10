param(
    [string]$RawDir = "data/local/open",
    [string]$ArtifactRoot = "artifacts",
    [string]$OutDir = "artifacts\postgate\raw_spatial_grid_temporal_cnn_scada_multitask_strict_v1"
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repo ".venv\Scripts\python.exe"
$runner = Join-Path $repo "scripts\run_raw_spatial_grid_temporal_cnn_scada_multitask.py"
$guard = Join-Path $repo "artifacts\locks\heavy_cpu_fit.pid.json"
$resolvedOut = [System.IO.Path]::GetFullPath((Join-Path $repo $OutDir))
if (Test-Path -LiteralPath $guard) { throw "shared heavy guard already exists: $guard" }
if (Test-Path -LiteralPath $resolvedOut) { throw "canonical output already exists: $resolvedOut" }
$logDir = Join-Path $repo "artifacts\logs"
New-Item -ItemType Directory -Path $logDir -Force | Out-Null
$stdout = Join-Path $logDir "raw_spatial_grid_temporal_cnn_scada_multitask_v1.stdout.log"
$stderr = Join-Path $logDir "raw_spatial_grid_temporal_cnn_scada_multitask_v1.stderr.log"
if ((Test-Path -LiteralPath $stdout) -or (Test-Path -LiteralPath $stderr)) {
    throw "fresh launcher log path required"
}
$arguments = @(
    $runner, "--stage", "all", "--raw-dir", $RawDir,
    "--artifact-root", $ArtifactRoot, "--out-dir", $resolvedOut
)
$process = Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $repo `
    -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
@{
    pid = $process.Id
    stdout = $stdout
    stderr = $stderr
    out_dir = $resolvedOut
} | ConvertTo-Json -Compress
